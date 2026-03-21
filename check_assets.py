import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("DATASET CHECK")
print("=" * 60)

# Core datasets needed for v3 experiments
datasets_to_check = [
    'pretext_jobs',
    'pretext_forums',
    'pretext_microblog',
    'pretext_initialization_c4_en',
    'pretext_code'
]

from thesis_platform.data.loaders import load_samples

for ds in datasets_to_check:
    path = Path(f'thesis_platform/datasets/{ds}/formatted')
    if path.exists():
        files = list(path.glob('*.json'))
        if files:
            try:
                # Try loading via the platform's loader
                samples = load_samples(
                    path,
                    dataset_name=ds,
                    source='check',
                    task_type='check',
                    round_id=0,
                    client_id='check',
                    prefix='check'
                )
                print(f'{ds}: OK ({len(files)} files, {len(samples)} samples loaded)')
            except Exception as e:
                print(f'{ds}: ERROR - {e}')
        else:
            print(f'{ds}: EMPTY (no files)')
    else:
        print(f'{ds}: MISSING')

print()
print("=" * 60)
print("MODEL CHECK")
print("=" * 60)

models_to_check = [
    'llama_2_7b_hf',
    'llama_3_1_8b_instruct',
    'llama_3_2_3b_instruct',
    'roberta_large',
    'all_minilm_l6_v2',
    'distilgpt2'
]

for model in models_to_check:
    path = Path(f'thesis_platform/open_model/{model}')
    if path.exists():
        # Check for key files
        has_config = (path / 'config.json').exists()
        has_model = any(f.suffix in ['.safetensors', '.bin', '.pt'] for f in path.glob('*'))
        has_tokenizer = (path / 'tokenizer.json').exists() or (path / 'tokenizer_config.json').exists()
        if has_config and has_model:
            print(f'{model}: OK (config={has_config}, model={has_model}, tokenizer={has_tokenizer})')
        else:
            print(f'{model}: INCOMPLETE (config={has_config}, model={has_model}, tokenizer={has_tokenizer})')
    else:
        print(f'{model}: MISSING')

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
