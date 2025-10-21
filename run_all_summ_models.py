import subprocess
import sys
import os
import time
import csv
import argparse
from typing import List, Dict


SEQ2SEQ_MODELS = [
    'facebook/bart-large-cnn',
    'google/pegasus-cnn_dailymail',
    'google/pegasus-xsum',
    'google/flan-t5-large',
    'Talina06/arxiv-summarization',
    'philschmid/bart-large-cnn-samsum',
]

GPT_LIKE_MODELS = [
    'NousResearch/Nous-Hermes-13b',
    'tiiuae/falcon-7b-instruct',
    'openlm-research/open_llama_13b',
    'google/gemma-7b-it',
    'Qwen/Qwen1.5-7B-Chat',
]

MODELS = SEQ2SEQ_MODELS + GPT_LIKE_MODELS


MODEL_OVERRIDES: Dict[str, Dict[str, str]] = {
    'NousResearch/Nous-Hermes-13b': {
        'gen_batch_size': '2',
        'batch_size_group': '10'
    },
    'mosaicml/mpt-7b-instruct': {
        'gen_batch_size': '2',
        'batch_size_group': '10'
    },
    'tiiuae/falcon-7b-instruct': {
        'gen_batch_size': '2',
        'batch_size_group': '10'
    },
    'openlm-research/open_llama_13b': {
        'gen_batch_size': '2',
        'batch_size_group': '10'
    },
    'google/gemma-7b-it': {
        'gen_batch_size': '2',
        'batch_size_group': '10',
    },
    'Qwen/Qwen1.5-7B-Chat': {
        'gen_batch_size': '2',
        'batch_size_group': '8',
    },

}

DEFAULT_OUTPUT_CSV = 'model_comparison_all.csv'
RESULTS_DIR = 'results'


def get_completed_models(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        return []
    completed = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row.get('model_name')
            if model_name:
                completed.append(model_name.strip())
    return completed


def get_models_subset(subset: str) -> List[str]:
    subset = (subset or 'all').lower()
    if subset == 'gpt':
        return GPT_LIKE_MODELS
    if subset == 'seq2seq':
        return SEQ2SEQ_MODELS
    return MODELS


def run_evaluation(model_name: str, output_csv: str, raw_path: str, ref_path: str, skip_bertscore: bool) -> bool:
    print(f"\n{'='*60}")
    print(f"Running evaluation for model: {model_name}")
    print(f"{'='*60}")

    # Override parameters for large GPT models
    gen_bs = MODEL_OVERRIDES.get(model_name, {}).get('gen_batch_size', '16')
    group_bs = MODEL_OVERRIDES.get(model_name, {}).get('batch_size_group', '10')

    cmd = [
        sys.executable, 'evaluate_summarization.py',
        '--model', model_name,
        '--out', output_csv,
        '--raw', raw_path,
        '--ref', ref_path,
        '--batch_size_group', group_bs,
        '--gen_batch_size', gen_bs,
        '--bertscore_model', 'allenai/scibert_scivocab_uncased',
        '--json_out_dir', RESULTS_DIR,
        '--min_words', '100',
        '--max_words', '140',
    ]

    if skip_bertscore:
        cmd.append('--skip_bertscore')

    try:
        start_time = time.time()
        result = subprocess.run(cmd, text=True, encoding='utf-8')
        end_time = time.time()

        print(f"\nExecution time: {end_time - start_time:.2f} seconds")

        if result.returncode == 0:
            print(f"Model {model_name} processed successfully!")
            return True
        else:
            print(f"ERROR processing model {model_name}")
            return False

    except Exception as e:
        print(f"EXCEPTION running model {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch runner for summarization evaluation')
    parser.add_argument('--subset', default='all', choices=['all', 'gpt', 'seq2seq'],
                        help='Which subset of models to run')
    parser.add_argument('--out', default=DEFAULT_OUTPUT_CSV, help='Path to CSV with results')
    parser.add_argument('--raw', default='raw_snippets.json', help='Path to file with raw snippets')
    parser.add_argument('--ref', default='summ_snippets.json', help='Path to reference summaries')
    parser.add_argument('--skip_bertscore', action='store_true', help='Skip BERTScore for faster execution')
    args = parser.parse_args()

    models_to_run = get_models_subset(args.subset)

    # Check for required files
    for file in ['evaluate_summarization.py', args.raw, args.ref]:
        if not os.path.exists(file):
            print(f"Missing required file: {file}")
            return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load list of already processed models
    completed_models = get_completed_models(args.out)
    print(f"Already processed models: {len(completed_models)}")
    if completed_models:
        print("Skipping:", ", ".join(completed_models))

    print(f"\nStarting run of {len(models_to_run)} models (subset={args.subset})...")
    print(f"Results will be written to: {args.out}")
    print(f"JSON reports for each model will be in folder: {RESULTS_DIR}")

    successful = []
    failed = []
    total_start = time.time()

    for i, model in enumerate(models_to_run, 1):
        print(f"\nProgress: {i}/{len(models_to_run)} — {model}")

        if model in completed_models:
            print(f"Model {model} already in CSV — skipping.")
            continue

        success = run_evaluation(model, args.out, args.raw, args.ref, args.skip_bertscore)

        if success:
            successful.append(model)
        else:
            failed.append(model)

        # Small pause between models
        if i < len(models_to_run):
            print("Pausing 5 seconds to free up memory...")
            time.sleep(5)

    total_time = time.time() - total_start

    # Final report
    print(f"\n{'='*80}")
    print("FINAL REPORT")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Successfully processed: {len(successful)}/{len(models_to_run)} new models")

    if successful:
        print("\nNew successful models:")
        for m in successful:
            print(f"  - {m}")

    if failed:
        print("\nErrors with models:")
        for m in failed:
            print(f"  - {m}")

    print(f"\nFinal results in: {args.out}")
    print(f"Summaries in: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
