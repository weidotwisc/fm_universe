#!/usr/bin/env python3
"""
Evaluate Llama-3.1-8B-Instruct on HumanEval benchmark using vLLM.

Requirements:
    pip install vllm datasets numpy

Usage:
    python evaluate_humaneval_vllm.py --model_name /path/to/model
    python evaluate_humaneval_vllm.py --num_samples 10 --temperature 0.8  # For pass@10
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
from datasets import load_dataset
from vllm import LLM, SamplingParams


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
    
    return np.array([
        estimator(int(n), int(c), k)
        for n, c in zip(num_samples, num_correct)
    ])


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


def unsafe_execute(problem: dict, completion: str, timeout: float = 3.0) -> str:
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
    
    total = np.array([len(task_results[tid]) for tid in task_results])
    correct = np.array([sum(task_results[tid]) for tid in task_results])
    
    min_samples = total.min()
    valid_k = [k for k in k_values if k <= min_samples]
    
    pass_at_k = {}
    for k in valid_k:
        pass_at_k[f"pass@{k}"] = float(estimate_pass_at_k(total, correct, k).mean())
    
    return pass_at_k


# ============================================================================
# Code extraction
# ============================================================================

def extract_code_from_response(response: str, entry_point: str) -> str:
    """
    Extract the complete function from model response.
    
    Returns the full function code (including signature and docstring).
    """
    # Try to extract from markdown code block first
    code_block_match = re.search(r"```python\n?(.*?)```", response, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1).strip()
    else:
        code_block_match = re.search(r"```\n?(.*?)```", response, re.DOTALL)
        if code_block_match:
            code = code_block_match.group(1).strip()
        else:
            code = response.strip()
    
    # Find the function definition
    func_pattern = rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\)[^:]*:.*?)(?=\ndef\s|\nclass\s|\Z)'
    func_match = re.search(func_pattern, code, re.DOTALL)
    
    if func_match:
        return func_match.group(1).rstrip()
    
    # If we can't find the specific function, return cleaned code
    # Stop at common end markers
    stop_patterns = ['\ndef ', '\nclass ', '\nif __name__', '\n# Test', '\n# Example', '\nassert ']
    for pattern in stop_patterns:
        if pattern in code:
            code = code[:code.index(pattern)]
    
    return code.rstrip()


def build_full_completion(extracted_code: str, original_prompt: str, entry_point: str) -> str:
    """
    Build the full completion for evaluation.
    
    If the extracted code contains the full function, use it directly.
    Otherwise, append it to the original prompt.
    """
    # Check if extracted code already has the function definition
    if f"def {entry_point}" in extracted_code:
        # Check if it has imports that the original prompt has
        if "from typing import" in original_prompt and "from typing import" not in extracted_code:
            # Add the import
            return "from typing import List\n\n" + extracted_code
        return extracted_code
    
    # Otherwise, it's just the body - append to original prompt
    return original_prompt + extracted_code


# ============================================================================
# Main evaluation
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on HumanEval using vLLM")
    parser.add_argument(
        "--model_name",
        type=str,
        default="/gpfs/users/weiz/ckpts/llama/llama-3.1-8b-instruct",
        help="Path to model",
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="humaneval_vllm_samples.jsonl")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
    )
    
    # Sampling parameters
    if args.temperature > 0:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=args.max_tokens,
            n=args.num_samples,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=args.max_tokens,
            n=args.num_samples,
        )
    
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
    
    # Build prompts using vLLM's chat interface
    print("Preparing prompts...")
    prompts = []
    task_ids = []
    
    for problem in dataset:
        task_ids.append(problem["task_id"])
        
        # Simple, direct prompt that asks for the completed function
        user_content = f"""Complete this Python function:

{problem["prompt"]}"""
        
        prompts.append(user_content)
    
    # Generate using vLLM chat interface
    print(f"Generating completions for {len(prompts)} problems...")
    print(f"Settings: num_samples={args.num_samples}, temperature={args.temperature}")
    
    # Use chat completion
    outputs = llm.chat(
        messages=[[{"role": "user", "content": p}] for p in prompts],
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    
    # Process outputs
    samples = []
    debug_count = 0
    
    for task_id, output in zip(task_ids, outputs):
        problem = problems[task_id]
        original_prompt = problem["prompt"]
        entry_point = problem["entry_point"]
        
        for completion_output in output.outputs:
            response = completion_output.text
            
            # Extract code from response
            extracted = extract_code_from_response(response, entry_point)
            
            # Build full completion
            full_completion = build_full_completion(extracted, original_prompt, entry_point)
            
            if args.debug and debug_count < 5:
                print(f"\n{'='*60}")
                print(f"Problem: {task_id}")
                print(f"{'='*60}")
                print(f"Model response:\n{response[:800]}")
                print(f"\nExtracted code:\n{extracted[:500]}")
                print(f"\nFull completion:\n{full_completion}")
                result = check_correctness(problem, full_completion)
                print(f"\nTest result: {result['result']}")
                debug_count += 1
            
            samples.append({
                "task_id": task_id,
                "completion": full_completion,
            })
    
    # Save samples
    print(f"\nWriting samples to {args.output_file}...")
    write_jsonl(args.output_file, samples)
    
    # Evaluate
    print("Running evaluation...")
    k_values = [1, 10, 100] if args.num_samples >= 100 else [1, 10] if args.num_samples >= 10 else [1]
    results = evaluate_functional_correctness(
        args.output_file,
        problems,
        k_values=k_values,
        timeout=5.0,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("HumanEval Results")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value * 100:.2f}%")
    
    # Save results
    results_file = args.output_file.replace(".jsonl", "_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
