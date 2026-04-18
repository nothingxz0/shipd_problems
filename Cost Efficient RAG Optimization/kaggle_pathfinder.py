#!/usr/bin/env python3
"""
Kaggle dataset pathfinder.

Purpose:
- Detect where train/test files are mounted in a Kaggle notebook.
- Print a clear, copy-paste-ready --data-dir value.

Usage:
  python kaggle_pathfinder.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

REQUIRED_FILES = ("train.csv", "test.csv")
OPTIONAL_FILES = ("sample_submission.csv",)


def list_immediate_dirs(path: Path) -> List[str]:
    if not path.exists() or not path.is_dir():
        return []
    out: List[str] = []
    for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir():
            out.append(child.name)
    return out


def scan_for_dataset_dirs(root: Path, max_depth: int = 5) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []

    candidates: List[Path] = []
    root_depth = len(root.parts)

    for cur, dirs, files in os.walk(root):
        cur_path = Path(cur)
        depth = len(cur_path.parts) - root_depth

        if depth > max_depth:
            dirs[:] = []
            continue

        file_set = set(files)
        if all(name in file_set for name in REQUIRED_FILES):
            candidates.append(cur_path)

    return candidates


def score_candidate(path: Path) -> Tuple[int, int, str]:
    names = {p.name for p in path.iterdir() if p.is_file()}
    optional_hits = sum(1 for name in OPTIONAL_FILES if name in names)
    # Higher optional file count is better, shorter paths are better.
    return (optional_hits, -len(path.parts), str(path))


def describe_candidate(path: Path) -> Dict[str, object]:
    files = sorted([p.name for p in path.iterdir() if p.is_file()])
    return {
        "path": str(path),
        "has_required": all(name in files for name in REQUIRED_FILES),
        "has_optional": [name for name in OPTIONAL_FILES if name in files],
        "files": files,
    }


def main() -> None:
    cwd = Path.cwd()
    kaggle_input = Path("/kaggle/input")

    roots_to_scan = [
        kaggle_input,
        cwd,
        cwd / "dataset",
        cwd / "dataset" / "public",
        cwd / "data",
    ]

    all_candidates: List[Path] = []
    seen = set()

    for root in roots_to_scan:
        for candidate in scan_for_dataset_dirs(root):
            if candidate not in seen:
                seen.add(candidate)
                all_candidates.append(candidate)

    all_candidates.sort(key=score_candidate, reverse=True)

    print("=== Kaggle Pathfinder ===")
    print(f"cwd: {cwd}")
    print(f"kaggle_input_exists: {kaggle_input.exists()}")
    print(f"kaggle_input_dirs: {list_immediate_dirs(kaggle_input)}")
    print()

    if not all_candidates:
        print("No dataset directory found with required files:", ", ".join(REQUIRED_FILES))
        print("Hint: in Kaggle, check that your dataset is attached and mounted under /kaggle/input")
        return

    best = all_candidates[0]
    print("Best match:")
    print(f"DATA_DIR={best}")
    print(f"Use with solution.py: --data-dir {best}")
    print()

    print("All matches:")
    for idx, candidate in enumerate(all_candidates, start=1):
        info = describe_candidate(candidate)
        print(f"{idx}. {json.dumps(info, ensure_ascii=True)}")


if __name__ == "__main__":
    main()
