from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Set

from pydantic import BaseModel


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_processed_sources(checkpoint_file: Path) -> Set[str]:
    if not checkpoint_file.exists():
        return set()
    processed: Set[str] = set()
    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            processed.add(line)
    return processed


def append_processed_source(checkpoint_file: Path, source_path: str) -> None:
    with checkpoint_file.open("a", encoding="utf-8") as f:
        f.write(source_path + "\n")


def append_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

