#!/usr/bin/env python3
"""
Evaluate Llama-3.1-8B-Instruct on HumanEval benchmark using HuggingFace transformers.

Requirements:
    pip install torch transformers accelerate datasets numpy

Usage:
    python evaluate_humaneval_hf.py --debug  # Test first 5 problems
    python evaluate_humaneval_hf.py          # Full evaluation
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
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def read_jsonl(filename: str) -> list:
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]


def estimate_pass_at_k(num_samples: list, num_correct: list, k: int) -> np.ndarray:
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def unsafe_execute(problem: dict, completion: str, timeout: float = 5.0) -> str:
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


def check_correctness(problem: dict, completion: str, timeout: float = 5.0) -> dict:
    result = unsafe_execute(problem, completion, timeout)
    return {"task_id": problem["task_id"], "passed": result == "passed", "result": result}


def evaluate_functional_correctness(sample_file: str, problems: dict, k_values: list = [1]) -> dict:
    samples = read_jsonl(sample_file)
    samples_by_task = defaultdict(list)
    for sample in samples:
        samples_by_task[sample["task_id"]].append(sample["completion"])
    
    results = []
    for task_id, completions in samples_by_task.items():
        for completion in completions:
            results.append(check_correctness(problems[task_id], completion))
    
    task_results = defaultdict(list)
    for result in results:
        task_results[result["task_id"]].append(result["passed"])
    
    total = np.array([len(task_results[tid]) for tid in task_results])
    correct = np.array([sum(task_results[tid]) for tid in task_results])
    
    valid_k = [k for k in k_values if k <= total.min()]
    return {f"pass@{k}": float(estimate_pass_at_k(total, correct, k).mean()) for k in valid_k}


# ============================================================================
# Code extraction - THE KEY PART
# ============================================================================

def extract_code_from_response(response: str, entry_point: str, original_prompt: str) -> str:
    """
    Extract complete, executable code from model response.
    
    Strategy:
    1. Extract code from markdown blocks if present
    2. Find the target function
    3. Include necessary imports
    4. Return complete, runnable code
    """
    # Step 1: Extract from markdown code blocks
    code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
    if code_blocks:
        # Use the first/largest code block that contains our function
        for block in code_blocks:
            if entry_point in block:
                code = block.strip()
                break
        else:
            code = code_blocks[0].strip()
    else:
        # No code blocks, use raw response
        code = response.strip()
    
    # Step 2: Check if code contains the function definition
    if f"def {entry_point}" in code:
        # Extract just the function (stop at next def/class)
        lines = code.split('\n')
        result_lines = []
        in_function = False
        function_indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Capture imports
            if stripped.startswith('from ') or stripped.startswith('import '):
                result_lines.append(line)
                continue
            
            # Start of our function
            if stripped.startswith(f'def {entry_point}'):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                result_lines.append(line)
                continue
            
            if in_function:
                # Check if we've hit a new top-level definition
                if stripped and not line[0].isspace() and (stripped.startswith('def ') or stripped.startswith('class ')):
                    break
                # Check for same-level definition (for nested functions case)
                current_indent = len(line) - len(line.lstrip()) if stripped else function_indent + 1
                if stripped and current_indent <= function_indent and (stripped.startswith('def ') or stripped.startswith('class ')):
                    break
                result_lines.append(line)
        
        code = '\n'.join(result_lines)
    
    # Step 3: Ensure we have necessary imports from original prompt
    if 'List' in original_prompt and 'from typing import' not in code:
        if 'List[' in code or 'List[' in original_prompt:
            code = 'from typing import List\n\n' + code
    if 'Optional' in original_prompt and 'Optional' not in code:
        code = 'from typing import Optional\n\n' + code
    if 'Tuple' in original_prompt and 'from typing import' not in code:
        if 'Tuple[' in code:
            code = 'from typing import Tuple\n\n' + code
    
    # Clean up multiple imports
    lines = code.split('\n')
    imports = []
    other_lines = []
    for line in lines:
        if line.strip().startswith('from typing import') or line.strip().startswith('import '):
            if line.strip() not in imports:
                imports.append(line.strip())
        else:
            other_lines.append(line)
    
    if imports:
        code = '\n'.join(imports) + '\n\n' + '\n'.join(other_lines)
    else:
        code = '\n'.join(other_lines)
    
    # Remove leading/trailing whitespace but preserve internal structure
    code = code.strip()
    
    return code


# ============================================================================
# Model and generation
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on HumanEval")
    parser.add_argument("--model_name", type=str, 
                        default="/gpfs/users/weiz/ckpts/llama/llama-3.1-8b-instruct")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="humaneval_hf_samples.jsonl")
    parser.add_argument("--debug", action="store_true", help="Debug first 5 problems")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, load_in_8bit=False, load_in_4bit=False):
    print(f"Loading model: {model_name}")
    is_local = os.path.isdir(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=is_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": is_local,
    }
    
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int, 
                        temperature: float, num_samples: int) -> list[str]:
    """Generate completions using the model."""
    
    # Build chat messages
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": num_samples,
    }
    
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})
    else:
        gen_kwargs["do_sample"] = False
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Decode only new tokens
    prompt_len = inputs.input_ids.shape[1]
    completions = []
    for output in outputs:
        new_tokens = output[prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(text)
    
    return completions


def main():
    args = parse_args()
    
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.load_in_8bit, args.load_in_4bit
    )
    
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    
    problems = {p["task_id"]: {
        "task_id": p["task_id"], "prompt": p["prompt"], 
        "entry_point": p["entry_point"], "test": p["test"],
        "canonical_solution": p["canonical_solution"],
    } for p in dataset}
    
    samples = []
    total = len(dataset) if not args.debug else 5  # Only 5 problems in debug mode
    passed_count = 0
    
    print(f"Generating completions for {total} problems...")
    print(f"Settings: temperature={args.temperature}, max_new_tokens={args.max_new_tokens}")
    
    for idx, problem in enumerate(dataset):
        if idx >= total:
            break
            
        task_id = problem["task_id"]
        original_prompt = problem["prompt"]
        entry_point = problem["entry_point"]
        
        # Simple, clear prompt
        user_prompt = f"Complete this Python function:\n\n{original_prompt}"
        
        # Generate
        raw_completions = generate_completion(
            model, tokenizer, user_prompt,
            args.max_new_tokens, args.temperature, args.num_samples
        )
        
        for raw in raw_completions:
            # Extract code
            code = extract_code_from_response(raw, entry_point, original_prompt)
            
            # Debug output
            if args.debug:
                print(f"\n{'='*70}")
                print(f"Problem {idx}: {task_id}")
                print(f"{'='*70}")
                print(f"Entry point: {entry_point}")
                print(f"\nOriginal prompt:\n{original_prompt}")
                print(f"\nRaw model output:\n{raw[:600]}{'...' if len(raw) > 600 else ''}")
                print(f"\nExtracted code:\n{code}")
                
                result = check_correctness(problems[task_id], code)
                status = "✓ PASSED" if result["passed"] else f"✗ FAILED: {result['result']}"
                print(f"\nResult: {status}")
                if result["passed"]:
                    passed_count += 1
            
            samples.append({"task_id": task_id, "completion": code})
        
        if not args.debug and (idx + 1) % 20 == 0:
            print(f"Progress: {idx + 1}/{total}")
    
    if args.debug:
        print(f"\n{'='*70}")
        print(f"[Debug Summary] Passed {passed_count}/{total} problems")
        print(f"{'='*70}")
        return
    
    # Save and evaluate
    print(f"\nWriting samples to {args.output_file}...")
    write_jsonl(args.output_file, samples)
    
    print("Evaluating...")
    k_values = [1, 10, 100] if args.num_samples >= 100 else [1, 10] if args.num_samples >= 10 else [1]
    results = evaluate_functional_correctness(args.output_file, problems, k_values)
    
    print("\n" + "=" * 50)
    print("HumanEval Results")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value * 100:.2f}%")
    
    results_file = args.output_file.replace(".jsonl", "_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
