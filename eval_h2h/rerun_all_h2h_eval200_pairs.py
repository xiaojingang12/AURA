#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BATCH_SCRIPT = SCRIPT_DIR / "batch_h2h_eval_200.sh"
DEFAULT_EVAL_ROOT = REPO_ROOT / "eval_results_h2h_eval200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun batch_h2h_eval_200.sh for every comparison pair found under "
            "eval_results_h2h_eval200 without modifying the original batch script."
        )
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help="Directory containing existing *_vs_* comparison folders.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help="Output root passed to the temporary batch script.",
    )
    parser.add_argument(
        "--batch-script",
        type=Path,
        default=DEFAULT_BATCH_SCRIPT,
        help="Path to the original batch_h2h_eval_200.sh template.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Run only one specific pair directory name such as GGraphRAG_vs_HippoRAG. Can be used multiple times.",
    )
    parser.add_argument(
        "--list-json",
        type=Path,
        help="Optional override for LIST_JSON in the temporary batch script.",
    )
    parser.add_argument(
        "--collected-root",
        type=Path,
        help="Optional override for COLLECTED_ROOT in the temporary batch script.",
    )
    parser.add_argument(
        "--engine",
        help="Optional override for ENGINE in the temporary batch script.",
    )
    parser.add_argument(
        "--api-base",
        help="Optional override for API_BASE in the temporary batch script.",
    )
    parser.add_argument(
        "--api-key",
        help="Optional override for API_KEY in the temporary batch script.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Optional override for NUM_WORKERS in the temporary batch script.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Optional override for MAX_RETRIES in the temporary batch script.",
    )
    parser.add_argument(
        "--force-existing",
        action="store_true",
        help="Set FORCE=true so existing per-sample outputs are recomputed as well.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pairs and substituted settings without executing anything.",
    )
    return parser.parse_args()


def quote_bash(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"{escaped}"'


def replace_assignment(script_text: str, name: str, new_line: str) -> str:
    pattern = re.compile(rf"^{re.escape(name)}=.*$", re.MULTILINE)
    updated_text, count = pattern.subn(new_line, script_text, count=1)
    if count != 1:
        raise ValueError(f"Failed to replace assignment for {name}")
    return updated_text


def discover_pairs(eval_root: Path, requested_pairs: list[str]) -> list[str]:
    if requested_pairs:
        missing_pairs = [pair for pair in requested_pairs if not (eval_root / pair).is_dir()]
        if missing_pairs:
            missing = ", ".join(missing_pairs)
            raise FileNotFoundError(
                f"Requested pair directories do not exist under {eval_root}: {missing}"
            )
        return requested_pairs

    return sorted(
        path.name
        for path in eval_root.iterdir()
        if path.is_dir() and "_vs_" in path.name
    )


def build_temp_script(
    template_text: str,
    batch_script: Path,
    pair_name: str,
    out_root: Path,
    args: argparse.Namespace,
) -> Path:
    method1, method2 = pair_name.split("_vs_", 1)

    patched_text = template_text
    patched_text = replace_assignment(
        patched_text,
        "METHOD1",
        f"METHOD1={quote_bash(method1)}",
    )
    patched_text = replace_assignment(
        patched_text,
        "METHOD2",
        f"METHOD2={quote_bash(method2)}",
    )
    patched_text = replace_assignment(
        patched_text,
        "FORCE",
        f'FORCE={quote_bash("true" if args.force_existing else "false")}',
    )
    patched_text = replace_assignment(
        patched_text,
        "OUT_ROOT",
        f"OUT_ROOT={quote_bash(str(out_root))}",
    )

    optional_replacements: list[tuple[str, str | int | Path | None]] = [
        ("LIST_JSON", args.list_json),
        ("COLLECTED_ROOT", args.collected_root),
        ("ENGINE", args.engine),
        ("API_BASE", args.api_base),
        ("API_KEY", args.api_key),
        ("NUM_WORKERS", args.num_workers),
        ("MAX_RETRIES", args.max_retries),
    ]

    for name, value in optional_replacements:
        if value is None:
            continue
        if isinstance(value, Path):
            replacement = f"{name}={quote_bash(str(value.resolve()))}"
        elif isinstance(value, int):
            replacement = f"{name}={value}"
        else:
            replacement = f"{name}={quote_bash(str(value))}"
        patched_text = replace_assignment(patched_text, name, replacement)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=batch_script.parent,
        prefix=f"tmp_batch_h2h_eval_200_{pair_name}_",
        suffix=".sh",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        handle.write(patched_text)
    finally:
        handle.close()
    return temp_path


def run_pair(
    template_text: str,
    batch_script: Path,
    pair_name: str,
    out_root: Path,
    args: argparse.Namespace,
    total_pairs: int,
    index: int,
) -> int:
    method1, method2 = pair_name.split("_vs_", 1)
    print(f"[INFO] ({index}/{total_pairs}) rerun pair: {pair_name}")
    print(f"[INFO] ({index}/{total_pairs}) METHOD1={method1} METHOD2={method2}")

    if args.dry_run:
        print(
            f"[DRY-RUN] FORCE={'true' if args.force_existing else 'false'} OUT_ROOT={out_root}"
        )
        return 0

    temp_script = build_temp_script(template_text, batch_script, pair_name, out_root, args)
    try:
        completed = subprocess.run(
            ["bash", str(temp_script)],
            cwd=str(REPO_ROOT),
            check=False,
        )
        return completed.returncode
    finally:
        temp_script.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    batch_script = args.batch_script.resolve()
    eval_root = args.eval_root.resolve()
    out_root = args.out_root.resolve()

    if not batch_script.is_file():
        raise FileNotFoundError(f"Batch script not found: {batch_script}")
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Eval root does not exist: {eval_root}")

    pair_names = discover_pairs(eval_root, args.pair)
    if not pair_names:
        raise SystemExit(f"No *_vs_* comparison directories found under: {eval_root}")

    template_text = batch_script.read_text(encoding="utf-8")

    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] Batch template: {batch_script}")
    print(f"[INFO] Pair source root: {eval_root}")
    print(f"[INFO] Output root: {out_root}")
    print(f"[INFO] Total pairs to run: {len(pair_names)}")
    print(f"[INFO] FORCE={'true' if args.force_existing else 'false'}")

    failed_pairs: list[str] = []
    for idx, pair_name in enumerate(pair_names, start=1):
        try:
            exit_code = run_pair(
                template_text=template_text,
                batch_script=batch_script,
                pair_name=pair_name,
                out_root=out_root,
                args=args,
                total_pairs=len(pair_names),
                index=idx,
            )
        except Exception as exc:
            print(f"[ERROR] ({idx}/{len(pair_names)}) {pair_name}: {exc}", file=sys.stderr)
            failed_pairs.append(pair_name)
            continue

        if exit_code != 0:
            print(
                f"[ERROR] ({idx}/{len(pair_names)}) {pair_name} exited with code {exit_code}",
                file=sys.stderr,
            )
            failed_pairs.append(pair_name)

    succeeded = len(pair_names) - len(failed_pairs)
    print(f"[DONE] Completed pairs: {succeeded}/{len(pair_names)}")
    if failed_pairs:
        print("[DONE] Failed pairs:")
        for pair_name in failed_pairs:
            print(f"  - {pair_name}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
