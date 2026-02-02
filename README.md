# HumanEval Evaluation for Llama-3.1-8B-Instruct

This script evaluates the Llama-3.1-8B-Instruct model on the [HumanEval](https://github.com/openai/human-eval) benchmark.

## Setup

### 1. Install Dependencies

```bash
pip install torch transformers accelerate datasets bitsandbytes
pip install git+https://github.com/openai/human-eval.git
```

### 2. Model Checkpoint

The script is configured to use the local checkpoint at:
```
/gpfs/users/weiz/ckpts/llama/llama-3.1-8b-instruct
```

You can override this with `--model_name` to use a different local path or a HuggingFace model ID.

## Usage

### Basic Evaluation (pass@1)

```bash
python evaluate_humaneval.py --num_samples 1 --temperature 0.1
```

### Pass@10 Evaluation

```bash
python evaluate_humaneval.py --num_samples 10 --temperature 0.8
```

### With a Different Model Path

```bash
python evaluate_humaneval.py --model_name /path/to/your/model
```

### With 4-bit Quantization (less VRAM)

```bash
python evaluate_humaneval.py --num_samples 1 --temperature 0.1 --load_in_4bit
```

### With 8-bit Quantization

```bash
python evaluate_humaneval.py --num_samples 1 --temperature 0.1 --load_in_8bit
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `/gpfs/users/weiz/ckpts/llama/llama-3.1-8b-instruct` | Local path or HuggingFace model name |
| `--num_samples` | 1 | Number of samples per problem |
| `--temperature` | 0.1 | Sampling temperature |
| `--max_new_tokens` | 512 | Maximum tokens to generate |
| `--output_file` | `humaneval_samples.jsonl` | Output file path |
| `--device` | `auto` | Device (auto, cuda, cpu) |
| `--load_in_8bit` | False | Use 8-bit quantization |
| `--load_in_4bit` | False | Use 4-bit quantization |

## Hardware Requirements

- **Full precision (FP16)**: ~16GB VRAM
- **8-bit quantization**: ~10GB VRAM
- **4-bit quantization**: ~6GB VRAM

## Output

The script produces:

1. `humaneval_samples.jsonl` - Generated completions for all problems
2. `humaneval_samples_results.json` - Final pass@k metrics

## Expected Results

For Llama-3.1-8B-Instruct, you can expect approximately:

- **pass@1**: ~40-50% (with temperature=0.1)
- **pass@10**: ~60-70% (with temperature=0.8)

*Note: Results may vary based on prompting strategy and generation parameters.*

## Troubleshooting

### CUDA Out of Memory

Use quantization:

```bash
python evaluate_humaneval.py --load_in_4bit
```

### human-eval Installation Issues

The `human-eval` package requires code execution. On some systems, you may need to:

```bash
export HF_ALLOW_CODE_EVAL=1
```

### Model Not Found

Make sure the checkpoint path exists and contains the model files (config.json, model weights, tokenizer files):

```bash
ls /gpfs/users/weiz/ckpts/llama/llama-3.1-8b-instruct/
```
