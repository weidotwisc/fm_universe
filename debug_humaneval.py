#!/usr/bin/env python3
"""
Debug script to inspect HumanEval generations.
Run this on a few problems to see what's happening.
"""

import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    is_local = os.path.isdir(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=is_local,
    )
    model.eval()
    return model, tokenizer


def create_prompt_v1(problem: dict) -> str:
    """Original prompt - instruction style."""
    function_signature = problem["prompt"]
    
    system_message = (
        "You are an expert Python programmer. Complete the given Python function. "
        "Only output the completion code that should come after the function signature "
        "and docstring. Do not repeat the function signature or docstring. "
        "Do not include any explanation, just the code."
    )
    
    user_message = f"""Complete the following Python function. Only provide the function body (the code that comes after the docstring). Do not repeat the function signature or docstring.

```python
{function_signature}
```

Provide only the completion code:"""

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def create_prompt_v2(problem: dict) -> str:
    """Alternative: Simpler completion-style prompt."""
    function_signature = problem["prompt"]
    
    # Just ask for completion directly
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Complete this Python function:\n\n{function_signature}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def create_prompt_v3(problem: dict) -> str:
    """Alternative: Direct code completion (no chat template)."""
    # Just return the prompt as-is for completion
    return problem["prompt"]


def extract_code_v1(generated_text: str, prompt: str, original_prompt: str) -> str:
    """Original extraction logic."""
    if generated_text.startswith(prompt):
        completion = generated_text[len(prompt):]
    else:
        completion = generated_text
    
    completion = completion.split("<|eot_id|>")[0]
    completion = completion.split("<|end_of_text|>")[0]
    
    # Extract from markdown
    code_block_match = re.search(r"```python\n?(.*?)```", completion, re.DOTALL)
    if code_block_match:
        completion = code_block_match.group(1)
    else:
        code_block_match = re.search(r"```\n?(.*?)```", completion, re.DOTALL)
        if code_block_match:
            completion = code_block_match.group(1)
    
    # Stop at markers
    stop_patterns = ["\ndef ", "\nclass ", "\nif __name__", "\n# Test", "\n# Example", "\nprint(", "\nassert "]
    for pattern in stop_patterns:
        if pattern in completion:
            completion = completion[:completion.index(pattern)]
    
    return completion.rstrip()


def extract_code_v2(generated_text: str, prompt: str, original_prompt: str) -> str:
    """
    Alternative: Look for the function and extract its body.
    """
    if generated_text.startswith(prompt):
        completion = generated_text[len(prompt):]
    else:
        completion = generated_text
    
    completion = completion.split("<|eot_id|>")[0]
    completion = completion.split("<|end_of_text|>")[0]
    
    # Extract from markdown
    code_block_match = re.search(r"```python\n?(.*?)```", completion, re.DOTALL)
    if code_block_match:
        completion = code_block_match.group(1)
    else:
        code_block_match = re.search(r"```\n?(.*?)```", completion, re.DOTALL)
        if code_block_match:
            completion = code_block_match.group(1)
    
    # If the completion contains the function signature, extract just the body
    # Look for where the docstring ends (after """) and take everything after
    if '"""' in original_prompt:
        # Check if completion repeats the function
        lines = completion.strip().split('\n')
        if lines and (lines[0].strip().startswith('def ') or lines[0].strip().startswith('from ') or lines[0].strip().startswith('import ')):
            # Model repeated the function, try to find the body
            in_docstring = False
            body_start = 0
            for i, line in enumerate(lines):
                if '"""' in line:
                    if in_docstring:
                        body_start = i + 1
                        break
                    else:
                        in_docstring = True
            if body_start > 0 and body_start < len(lines):
                completion = '\n'.join(lines[body_start:])
    
    # Stop at markers
    stop_patterns = ["\ndef ", "\nclass ", "\nif __name__", "\n# Test", "\n# Example"]
    for pattern in stop_patterns:
        if pattern in completion:
            completion = completion[:completion.index(pattern)]
    
    return completion.rstrip()


def generate_and_debug(model, tokenizer, problem, prompt_fn, extract_fn):
    """Generate and show what's happening."""
    prompt = prompt_fn(problem)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    completion = extract_fn(generated_text, prompt, problem["prompt"])
    
    return prompt, generated_text, completion


def test_completion(problem, completion):
    """Test if the completion works."""
    full_code = problem["prompt"] + completion
    test_code = problem["test"]
    entry_point = problem["entry_point"]
    
    program = full_code + "\n" + test_code + f"\ncheck({entry_point})"
    
    try:
        exec_globals = {}
        exec(program, exec_globals)
        return True, "PASSED"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    model_name = "/gpfs/users/weiz/ckpts/llama/llama-3.1-8b-instruct"
    
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    print("\nLoading HumanEval...")
    dataset = load_dataset("openai_humaneval", split="test")
    
    # Test on first few problems
    test_problems = [0, 1, 2, 3, 4]
    
    prompt_versions = [
        ("v1 (instruction)", create_prompt_v1, extract_code_v1),
        ("v2 (simpler chat)", create_prompt_v2, extract_code_v2),
    ]
    
    for idx in test_problems:
        problem = dataset[idx]
        print("\n" + "="*80)
        print(f"Problem: {problem['task_id']}")
        print("="*80)
        print(f"\nOriginal prompt:\n{problem['prompt']}")
        print(f"\nCanonical solution:\n{problem['canonical_solution']}")
        print(f"\nEntry point: {problem['entry_point']}")
        
        for version_name, prompt_fn, extract_fn in prompt_versions:
            print(f"\n--- Testing {version_name} ---")
            
            prompt, raw_output, completion = generate_and_debug(
                model, tokenizer, problem, prompt_fn, extract_fn
            )
            
            print(f"\nRaw model output (first 1000 chars):")
            # Show just the generated part
            if raw_output.startswith(prompt):
                generated_part = raw_output[len(prompt):]
            else:
                generated_part = raw_output
            print(generated_part[:1000])
            
            print(f"\nExtracted completion:")
            print(completion)
            
            print(f"\nFull code to execute:")
            full_code = problem["prompt"] + completion
            print(full_code)
            
            passed, result = test_completion(problem, completion)
            print(f"\nTest result: {result}")
            print("-"*40)


if __name__ == "__main__":
    main()
