"""
GSM8K Evaluation Script: Base vs GRPO Model

Evaluates both the base model (unsloth/gemma-3-1b-it) and the GRPO fine-tuned model
on the GSM8K test split using lenient answer extraction.

Usage:
    python eval.py

Configuration: Edit the constants at the top of this file
"""

import sys
sys.path.insert(0, '/workspace/rl_gym/grpo-training-result')

import json
import re
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from unsloth import FastModel
from peft import PeftModel
from utils import extract_hash_answer, system_prompt

# ============================================================
# CONFIGURATION - Edit these parameters as needed
# ============================================================
NUM_SAMPLES = 1319          # Number of test examples to evaluate (max: 1319)
BATCH_SIZE = 256            # Number of questions to process in parallel
OUTPUT_DIR = "/workspace/rl_gym/grpo-training-result/eval_results"
SEED = 42                 # Random seed for reproducible sampling

# Generation parameters (same as training)
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
MAX_NEW_TOKENS = 1792     # (2048 - 256)
MAX_PROMPT_LENGTH = 256

# Model paths
BASE_MODEL_NAME = "unsloth/gemma-3-1b-it"
GRPO_ADAPTER_PATH = "/workspace/rl_gym/grpo-training-result/gemma-3/"
# ============================================================

# Regex patterns for answer extraction
match_boxed = re.compile(r'\$\\boxed\{([^}]+)\}\$', flags=re.MULTILINE | re.DOTALL)
match_any_number = re.compile(r'[\d\.]+')


def load_base_model():
    """Load base model from HuggingFace"""
    print("Loading base model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    model.eval()
    return model, tokenizer


def load_grpo_model():
    """Load base model + GRPO LoRA adapters"""
    print("Loading GRPO model (base + LoRA adapters)...")

    # First load base model
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # Then load LoRA adapters
    model = PeftModel.from_pretrained(
        model,
        GRPO_ADAPTER_PATH,
        is_trainable=False,
    )
    model.eval()
    return model, tokenizer


def load_and_prepare_dataset(num_samples, seed=42):
    """Load and prepare GSM8K test split"""
    print(f"Loading GSM8K test split...")

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Total test examples: {len(dataset)}")

    # Sample if needed
    if num_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
        print(f"Sampled {num_samples} examples (seed={seed})")

    # Preprocess: extract numerical answers and add system prompt
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
        "question": x["question"],
    })

    return dataset


def extract_answer(response):
    """
    Extract answer from model response with priority logic.

    Priority (elif chain):
    1. $\\boxed{...}$ pattern (take LAST occurrence)
    2. Number(s) after <SOLUTION> tag:
       - Single number: use it
       - Multiple numbers: use LAST one
    3. LAST number in entire output

    Returns:
        tuple: (extracted_answer, extraction_method)
    """

    # Priority 1: Check for boxed pattern (take LAST)
    boxed_matches = match_boxed.findall(response)
    if boxed_matches:
        answer = boxed_matches[-1].strip()  # Take LAST occurrence
        # Extract number from boxed content (might contain LaTeX)
        numbers = match_any_number.findall(answer)
        if numbers:
            return numbers[-1], "boxed"
        return answer, "boxed"

    # Priority 2: Check for <SOLUTION> tag
    elif "<SOLUTION>" in response:
        # Extract all numbers after <SOLUTION>
        solution_part = response.split("<SOLUTION>", 1)[1]
        if "</SOLUTION>" in solution_part:
            solution_part = solution_part.split("</SOLUTION>", 1)[0]

        numbers = match_any_number.findall(solution_part)
        if len(numbers) == 1:
            return numbers[0], "solution_single"
        elif len(numbers) > 1:
            return numbers[-1], "solution_multiple"  # Take LAST
        else:
            return None, "solution_no_number"

    # Priority 3: Extract LAST number from entire output
    else:
        numbers = match_any_number.findall(response)
        if numbers:
            return numbers[-1], "last_number"
        return None, "no_number"


def evaluate_answer(extracted, ground_truth):
    """
    Compare extracted answer with ground truth using float conversion.

    Returns:
        bool: True if correct, False otherwise
    """
    if extracted is None:
        return False

    try:
        extracted_num = float(extracted.strip())
        truth_num = float(ground_truth.strip())
        return abs(extracted_num - truth_num) < 1e-6
    except (ValueError, AttributeError):
        return str(extracted).strip() == str(ground_truth).strip()


def generate_batch(model, tokenizer, prompts, batch_size=8):
    """
    Generate completions in batches for efficiency.

    Args:
        model: The model to generate with
        tokenizer: The tokenizer
        prompts: List of distinct chat message lists (one per question)
        batch_size: Number of DISTINCT questions to process in parallel

    Note: Each question is only generated once. Batching is for parallelization.

    Returns:
        List of generated texts
    """
    generations = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size  # Ceiling division

    for batch_idx, i in enumerate(range(0, len(prompts), batch_size), start=1):
        batch = prompts[i:i+batch_size]  # Get next batch_size distinct questions

        # Progress message
        end_idx = min(i + batch_size - 1, len(prompts) - 1)
        print(f"  Processing batch {batch_idx}/{total_batches} (questions {i}-{end_idx})...")

        # Apply chat template to each prompt
        texts = [
            tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in batch
        ]

        # Tokenize batch
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
        ).to("cuda")

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode only the generated part (skip input)
        for j, output in enumerate(outputs):
            input_length = inputs['input_ids'][j].shape[0]
            generated = tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True,
            )
            generations.append(generated)

    return generations


def evaluate_model(model, tokenizer, dataset, batch_size=8, model_name="base"):
    """
    Evaluate a model on the dataset.

    Returns:
        dict: Results containing per-example details and aggregate metrics
    """
    results = {
        "model": model_name,
        "num_examples": len(dataset),
        "examples": [],
        "metrics": {},
    }

    # Extract data
    prompts = [ex["prompt"] for ex in dataset]
    questions = [ex["question"] for ex in dataset]
    ground_truths = [ex["answer"] for ex in dataset]

    # Generate completions
    print(f"Generating {len(prompts)} completions for {model_name} model...")
    generations = generate_batch(model, tokenizer, prompts, batch_size=batch_size)

    # Evaluate each example
    correct = 0
    extraction_methods = {
        "boxed": 0,
        "solution_single": 0,
        "solution_multiple": 0,
        "solution_no_number": 0,
        "last_number": 0,
        "no_number": 0,
    }

    for i, (question, generation, truth) in enumerate(zip(questions, generations, ground_truths)):
        extracted, method = extract_answer(generation)
        is_correct = evaluate_answer(extracted, truth)

        if is_correct:
            correct += 1

        extraction_methods[method] += 1

        results["examples"].append({
            "index": i,
            "question": question,
            "ground_truth": truth,
            "generation": generation,
            "extracted_answer": extracted,
            "extraction_method": method,
            "correct": is_correct,
        })

    # Calculate metrics
    accuracy = correct / len(dataset) * 100 if len(dataset) > 0 else 0.0
    results["metrics"] = {
        "accuracy": accuracy,
        "num_correct": correct,
        "extraction_method_counts": extraction_methods,
        "extraction_method_percentages": {
            k: v / len(dataset) * 100 if len(dataset) > 0 else 0.0
            for k, v in extraction_methods.items()
        },
    }

    return results


def print_summary(results):
    """Print evaluation summary"""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    base = results["base_model"]["metrics"]
    grpo = results["grpo_model"]["metrics"]

    print(f"\nBase Model ({BASE_MODEL_NAME}):")
    print(f"  Accuracy: {base['accuracy']:.2f}%")
    print(f"  Correct: {base['num_correct']}/{results['base_model']['num_examples']}")

    print(f"\nGRPO Model (Base + LoRA):")
    print(f"  Accuracy: {grpo['accuracy']:.2f}%")
    print(f"  Correct: {grpo['num_correct']}/{results['grpo_model']['num_examples']}")

    print(f"\nImprovement: {results['comparison']['improvement']:.2f}%")

    print(f"\nExtraction Method Distribution (GRPO):")
    for method, pct in grpo['extraction_method_percentages'].items():
        count = grpo['extraction_method_counts'][method]
        print(f"  {method:20s}: {count:3d} ({pct:5.1f}%)")

    print("\n" + "="*60)


def main():
    """Main evaluation script"""
    print("\n" + "="*60)
    print("GSM8K EVALUATION: Base vs GRPO Model")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Num samples: {NUM_SAMPLES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Seed: {SEED}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Top-p: {TOP_P}")
    print(f"  Top-k: {TOP_K}")
    print()

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_and_prepare_dataset(NUM_SAMPLES, SEED)

    # Evaluate base model
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL")
    print("="*60)
    base_model, tokenizer = load_base_model()
    base_results = evaluate_model(base_model, tokenizer, dataset, BATCH_SIZE, "base")

    # Clear memory
    print("\nClearing GPU memory...")
    del base_model
    torch.cuda.empty_cache()

    # Evaluate GRPO model
    print("\n" + "="*60)
    print("EVALUATING GRPO MODEL")
    print("="*60)
    grpo_model, tokenizer = load_grpo_model()
    grpo_results = evaluate_model(grpo_model, tokenizer, dataset, BATCH_SIZE, "grpo")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"eval_results_{timestamp}.json"

    combined_results = {
        "config": {
            "num_samples": NUM_SAMPLES,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "base_model": BASE_MODEL_NAME,
            "grpo_adapter_path": GRPO_ADAPTER_PATH,
        },
        "base_model": base_results,
        "grpo_model": grpo_results,
        "comparison": {
            "base_accuracy": base_results["metrics"]["accuracy"],
            "grpo_accuracy": grpo_results["metrics"]["accuracy"],
            "improvement": grpo_results["metrics"]["accuracy"] - base_results["metrics"]["accuracy"],
        }
    }

    with open(results_file, "w") as f:
        json.dump(combined_results, f, indent=2)

    # Print summary
    print_summary(combined_results)
    print(f"\nResults saved to: {results_file}")
    print("\n✨ Evaluation complete!\n")


if __name__ == "__main__":
    main()
