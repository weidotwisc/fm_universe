#!/usr/bin/env python3
"""
Evaluate meta-llama/Llama-3.1-8B-Instruct on HumanEval benchmark.

Requirements:
    pip install torch transformers accelerate human-eval datasets

Usage:
    python evaluate_humaneval.py --num_samples 1 --temperature 0.1
    python evaluate_humaneval.py --num_samples 10 --temperature 0.8  # For pass@10
"""

import argparse
import json
import os
import re
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness


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
    return parser.parse_args()


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")

    # Check if it's a local path
    is_local = os.path.isdir(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization if requested
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

    # Load model
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


def create_prompt(problem: dict) -> str:
    """
    Create a prompt for the Llama-3.1 Instruct model.
    
    The prompt instructs the model to complete the given function.
    """
    function_signature = problem["prompt"]
    
    # Build the instruction prompt using Llama 3.1 chat format
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

    # Llama 3.1 Instruct chat template
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def extract_code_completion(generated_text: str, prompt: str) -> str:
    """
    Extract the code completion from the generated text.
    
    Handles various output formats and cleans up the completion.
    """
    # Remove the prompt from the beginning if present
    if generated_text.startswith(prompt):
        completion = generated_text[len(prompt):]
    else:
        completion = generated_text
    
    # Remove any trailing special tokens
    completion = completion.split("<|eot_id|>")[0]
    completion = completion.split("<|end_of_text|>")[0]
    
    # Extract code from markdown code blocks if present
    code_block_match = re.search(r"```python\n?(.*?)```", completion, re.DOTALL)
    if code_block_match:
        completion = code_block_match.group(1)
    else:
        # Try without language specifier
        code_block_match = re.search(r"```\n?(.*?)```", completion, re.DOTALL)
        if code_block_match:
            completion = code_block_match.group(1)
    
    # Stop at common end markers (next function, class, or if __name__)
    stop_patterns = [
        "\ndef ",
        "\nclass ",
        "\nif __name__",
        "\n# Test",
        "\n# Example",
        "\nprint(",
        "\nassert ",
    ]
    
    for pattern in stop_patterns:
        if pattern in completion:
            completion = completion[:completion.index(pattern)]
    
    return completion.rstrip()


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
    
    # Generation parameters
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
    
    completions = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=False)
        completion = extract_code_completion(generated_text, prompt)
        completions.append(completion)
    
    return completions


def evaluate_on_humaneval(
    model,
    tokenizer,
    num_samples: int = 1,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
    output_file: str = "humaneval_samples.jsonl",
):
    """
    Evaluate the model on HumanEval benchmark.
    
    Returns pass@k metrics.
    """
    # Load HumanEval dataset
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    
    samples = []
    total = len(dataset)
    
    print(f"Generating completions for {total} problems...")
    print(f"Settings: num_samples={num_samples}, temperature={temperature}")
    
    for idx, problem in enumerate(dataset):
        task_id = problem["task_id"]
        prompt_text = problem["prompt"]
        
        # Create the instruction prompt
        full_prompt = create_prompt(problem)
        
        # Generate completions
        completions = generate_completion(
            model,
            tokenizer,
            full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )
        
        # Create samples for evaluation
        for completion in completions:
            # The full solution is the original prompt + completion
            full_completion = prompt_text + completion
            samples.append({
                "task_id": task_id,
                "completion": full_completion,
            })
        
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{total} problems completed")
    
    # Write samples to file
    print(f"Writing samples to {output_file}...")
    write_jsonl(output_file, samples)
    
    # Run evaluation
    print("Running functional correctness evaluation...")
    results = evaluate_functional_correctness(
        output_file,
        k=[1, 10, 100] if num_samples >= 100 else [1, 10] if num_samples >= 10 else [1],
        n_workers=4,
    )
    
    return results


def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Run evaluation
    results = evaluate_on_humaneval(
        model,
        tokenizer,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_file=args.output_file,
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
