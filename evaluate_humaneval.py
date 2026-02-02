#!/usr/bin/env python3
"""
Evaluate meta-llama/Llama-3.1-8B-Instruct on HumanEval benchmark.

Requirements:
    pip install torch transformers accelerate datasets numpy

Usage:
    python evaluate_humaneval.py --num_samples 1 --temperature 0.1
    python evaluate_humaneval.py --num_samples 10 --temperature 0.8  # For pass@10
"""

import argparse
import contextlib
import io
import json
import os
import re
import signal
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# HumanEval evaluation utilities
# ============================================================================

def write_jsonl(filename: str, data: list):
    """Write a list of dicts to a jsonl file."""
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def read_jsonl(filename: str) -> list:
    """Read a jsonl file into a list of dicts."""
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]


def estimate_pass_at_k(num_samples: list, num_correct: list, k: int) -> np.ndarray:
    """Estimates pass@k using the unbiased estimator from the HumanEval paper."""
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    return np.array([
        estimator(int(n), int(c), k)
        for n, c in zip(num_samples, num_correct)
    ])


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager that raises TimeoutException after `seconds`."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def unsafe_execute(problem: dict, completion: str, timeout: float = 3.0) -> str:
    """Execute the completion against the test cases."""
    test_code = problem["test"]
    entry_point = problem["entry_point"]
    full_code = completion + "\n" + test_code + f"\ncheck({entry_point})"
    
    try:
        with time_limit(timeout):
            exec_globals = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(full_code, exec_globals)
        return "passed"
    except TimeoutException:
        return "timed out"
    except Exception as e:
        return f"failed: {type(e).__name__}: {e}"


def check_correctness(problem: dict, completion: str, timeout: float = 3.0) -> dict:
    """Check if a completion is correct for a given problem."""
    result = unsafe_execute(problem, completion, timeout)
    return {
        "task_id": problem["task_id"],
        "passed": result == "passed",
        "result": result,
    }


def evaluate_functional_correctness(
    sample_file: str,
    problems: dict,
    k_values: list = [1, 10, 100],
    timeout: float = 3.0,
) -> dict:
    """Evaluate functional correctness of generated samples."""
    samples = read_jsonl(sample_file)
    
    samples_by_task = defaultdict(list)
    for sample in samples:
        samples_by_task[sample["task_id"]].append(sample["completion"])
    
    results = []
    print(f"Evaluating {len(samples)} samples across {len(samples_by_task)} tasks...")
    
    for task_id, completions in samples_by_task.items():
        problem = problems[task_id]
        for completion in completions:
            result = check_correctness(problem, completion, timeout)
            results.append(result)
    
    task_results = defaultdict(list)
    for result in results:
        task_results[result["task_id"]].append(result["passed"])
    
    total = []
    correct = []
    for task_id in task_results:
        passed = task_results[task_id]
        total.append(len(passed))
        correct.append(sum(passed))
    
    total = np.array(total)
    correct = np.array(correct)
    
    min_samples = total.min()
    valid_k = [k for k in k_values if k <= min_samples]
    
    pass_at_k = {}
    for k in valid_k:
        pass_at_k[f"pass@{k}"] = float(estimate_pass_at_k(total, correct, k).mean())
    
    return pass_at_k


# ============================================================================
# Model loading and generation
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Llama-3.1-8B-Instruct on HumanEval"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/gpfs/users/weiz/ckpts/llama/llama-3.1-8b-instruct",
        help="Path to local model checkpoint or HuggingFace model name",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per problem (for pass@k evaluation)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (use 0.1 for pass@1, 0.8 for pass@k)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="humaneval_samples.jsonl",
        help="Output file for generated samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information for first few problems",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")

    is_local = os.path.isdir(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "local_files_only": is_local,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    print(f"Model loaded successfully on {model.device}")
    return model, tokenizer


def create_prompt(problem: dict, tokenizer) -> str:
    """
    Create a prompt using the tokenizer's chat template.
    
    This ensures we use the exact format the model expects.
    """
    function_signature = problem["prompt"]
    
    # The model tends to output complete functions, so we ask it to do that
    # and then extract just the function body ourselves
    messages = [
        {
            "role": "user", 
            "content": f"Complete this Python function. Output ONLY the completed function, no explanations:\n\n{function_signature}"
        }
    ]
    
    # Use the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return prompt


def extract_function_body(generated_text: str, original_prompt: str, entry_point: str) -> str:
    """
    Extract the function body from the model's generation.
    
    The model typically outputs the complete function, so we need to:
    1. Find the function in the output
    2. Extract just the body (after the docstring)
    3. Return that to be appended to the original prompt
    """
    # First, try to extract code from markdown code blocks
    code_block_match = re.search(r"```python\n?(.*?)```", generated_text, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1).strip()
    else:
        code_block_match = re.search(r"```\n?(.*?)```", generated_text, re.DOTALL)
        if code_block_match:
            code = code_block_match.group(1).strip()
        else:
            # No code block, use the raw text
            code = generated_text.strip()
    
    # Try to find the function definition
    # Pattern: def entry_point(...): followed by docstring, then body
    func_pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)[^:]*:'
    func_match = re.search(func_pattern, code)
    
    if func_match:
        # Found the function, now extract everything after the docstring
        after_def = code[func_match.end():]
        
        # Skip whitespace and find docstring
        after_def = after_def.lstrip()
        
        # Check for docstring (""" or ''')
        docstring_match = re.match(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', after_def)
        if docstring_match:
            # Get everything after the docstring
            body = after_def[docstring_match.end():]
        else:
            # No docstring found, the whole thing after def is the body
            body = after_def
        
        # Clean up the body
        body = body.strip('\n')
        
        # Stop at next function definition or class
        stop_patterns = ['\ndef ', '\nclass ', '\nif __name__']
        for pattern in stop_patterns:
            if pattern in body:
                body = body[:body.index(pattern)]
        
        # Ensure proper indentation (body should be indented)
        lines = body.split('\n')
        if lines and lines[0] and not lines[0][0].isspace():
            # First line not indented, add indentation
            body = '\n'.join('    ' + line if line.strip() else line for line in lines)
        
        return body
    
    # Fallback: couldn't find the function, try to use the code as-is
    # This handles cases where model just outputs the body
    code_lines = code.split('\n')
    
    # Remove any import statements or function definitions at the start
    start_idx = 0
    for i, line in enumerate(code_lines):
        stripped = line.strip()
        if stripped.startswith('from ') or stripped.startswith('import '):
            start_idx = i + 1
        elif stripped.startswith('def '):
            # Skip past the def line
            start_idx = i + 1
            # Also skip the docstring if present
            remaining = '\n'.join(code_lines[start_idx:])
            remaining = remaining.lstrip()
            docstring_match = re.match(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', remaining)
            if docstring_match:
                # Find where in code_lines this ends
                doc_end = docstring_match.end()
                char_count = 0
                for j in range(start_idx, len(code_lines)):
                    line_len = len(code_lines[j]) + 1  # +1 for newline
                    if char_count + line_len > doc_end:
                        start_idx = j + 1
                        break
                    char_count += line_len
            break
    
    body = '\n'.join(code_lines[start_idx:])
    
    # Ensure indentation
    lines = body.split('\n')
    if lines and lines[0] and not lines[0][0].isspace():
        body = '\n'.join('    ' + line if line.strip() else line for line in lines)
    
    return body.rstrip()


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    num_samples: int = 1,
) -> list[str]:
    """Generate code completions for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": num_samples,
    }
    
    if temperature > 0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
        })
    else:
        gen_kwargs["do_sample"] = False
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Decode only the new tokens (not the prompt)
    prompt_length = inputs.input_ids.shape[1]
    completions = []
    for output in outputs:
        new_tokens = output[prompt_length:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(generated_text)
    
    return completions


def evaluate_on_humaneval(
    model,
    tokenizer,
    num_samples: int = 1,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
    output_file: str = "humaneval_samples.jsonl",
    debug: bool = False,
):
    """Evaluate the model on HumanEval benchmark."""
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    
    problems = {}
    for problem in dataset:
        problems[problem["task_id"]] = {
            "task_id": problem["task_id"],
            "prompt": problem["prompt"],
            "entry_point": problem["entry_point"],
            "test": problem["test"],
            "canonical_solution": problem["canonical_solution"],
        }
    
    samples = []
    total = len(dataset)
    
    print(f"Generating completions for {total} problems...")
    print(f"Settings: num_samples={num_samples}, temperature={temperature}")
    
    for idx, problem in enumerate(dataset):
        task_id = problem["task_id"]
        original_prompt = problem["prompt"]
        entry_point = problem["entry_point"]
        
        # Create the chat prompt
        chat_prompt = create_prompt(problem, tokenizer)
        
        # Generate completions
        raw_completions = generate_completion(
            model,
            tokenizer,
            chat_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )
        
        # Extract function bodies and create full solutions
        for raw_completion in raw_completions:
            body = extract_function_body(raw_completion, original_prompt, entry_point)
            full_completion = original_prompt + body
            
            if debug and idx < 3:
                print(f"\n{'='*60}")
                print(f"Problem: {task_id}")
                print(f"{'='*60}")
                print(f"Raw completion:\n{raw_completion[:500]}...")
                print(f"\nExtracted body:\n{body}")
                print(f"\nFull code:\n{full_completion}")
                # Test it
                result = check_correctness(problems[task_id], full_completion)
                print(f"\nTest result: {result['result']}")
            
            samples.append({
                "task_id": task_id,
                "completion": full_completion,
            })
        
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{total} problems completed")
    
    print(f"Writing samples to {output_file}...")
    write_jsonl(output_file, samples)
    
    print("Running functional correctness evaluation...")
    k_values = [1, 10, 100] if num_samples >= 100 else [1, 10] if num_samples >= 10 else [1]
    results = evaluate_functional_correctness(
        output_file,
        problems,
        k_values=k_values,
        timeout=3.0,
    )
    
    return results


def main():
    args = parse_args()
    
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    
    results = evaluate_on_humaneval(
        model,
        tokenizer,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_file=args.output_file,
        debug=args.debug,
    )
    
    print("\n" + "=" * 50)
    print("HumanEval Results")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value * 100:.2f}%")
    
    results_file = args.output_file.replace(".jsonl", "_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
