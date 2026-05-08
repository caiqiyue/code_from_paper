from __future__ import annotations
import subprocess
import time
from datetime import datetime
from pathlib import Path
from paper_new_selector.repeat10_baseline_runner import (
    append_repeat10_summary_row,
    build_repeat10_child_env,
    build_repeat10_command,
    build_repeat10_run_specs,
    classify_retryable_failure,
    reset_repeat10_output_dir,
    resolve_repeat10_effective_status,
    resolve_repeat10_runtime_output_dir,
)

ROOT = Path('/mnt/public/caiqiyue_file/code_from_paper/paper-new-round19')
START_EXPERIMENT = 'ep_microblog_repeat10_seed04'
SUMMARY_PATH = ROOT / 'logs' / 'repeat10_baseline_screening_summary.tsv'
MASTER_LOG = ROOT / 'logs' / 'repeat10_baseline_screening_resume_from_ep_microblog_seed04_20260508.log'


def log(message: str) -> None:
    line = f"{datetime.now().strftime('%F %T')} {message}"
    print(line, flush=True)
    with MASTER_LOG.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def main() -> int:
    child_env = build_repeat10_child_env()
    specs = build_repeat10_run_specs()
    started = False
    had_failure = 0
    MASTER_LOG.write_text('', encoding='utf-8')
    for spec in specs:
        if not started:
            if spec.experiment_id != START_EXPERIMENT:
                continue
            started = True
        config_path = ROOT / spec.relative_config_path
        output_dir = resolve_repeat10_runtime_output_dir(spec)
        log_path = ROOT / 'logs' / f'{spec.experiment_id}.log'
        status = 1
        log(f'START {spec.experiment_id} dataset={spec.dataset} seed={spec.seed} cfg={config_path}')
        for attempt in (1, 2):
            reset_repeat10_output_dir(output_dir)
            mode = 'w' if attempt == 1 else 'a'
            with log_path.open(mode, encoding='utf-8') as handle:
                if attempt > 1:
                    handle.write(f'\n===== retry attempt {attempt} =====\n')
                completed = subprocess.run(
                    build_repeat10_command(spec, config_path),
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=child_env,
                )
            status = resolve_repeat10_effective_status(completed.returncode, output_dir)
            if status == 0:
                break
            failure_class = classify_retryable_failure(log_path.read_text(encoding='utf-8'))
            if failure_class != 'retryable_vllm_cache' or attempt == 2:
                break
            time.sleep(5)
        append_repeat10_summary_row(SUMMARY_PATH, spec, status)
        had_failure = had_failure or int(status != 0)
        log(f'END {spec.experiment_id} dataset={spec.dataset} seed={spec.seed} status={status}')
        time.sleep(2)
    return had_failure


if __name__ == '__main__':
    raise SystemExit(main())
