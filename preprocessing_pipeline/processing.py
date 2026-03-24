from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, List

from tqdm import tqdm

from preprocessing_pipeline.extraction import SourceFile, discover_source_files, extract_text
from preprocessing_pipeline.gigachat_client import GigaChatClient
from preprocessing_pipeline.models import Pass1Result, Pass2Result, ReviewRecord
from preprocessing_pipeline.prompts import (
    PASS1_SYSTEM_PROMPT,
    PASS2_SYSTEM_PROMPT,
    build_pass1_function,
    build_pass1_user_prompt,
    build_pass2_function,
    build_pass2_user_prompt,
)
from preprocessing_pipeline.utils import (
    append_jsonl,
    append_processed_source,
    ensure_directory,
    read_processed_sources,
)


@dataclass
class PipelineStats:
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    output_records: int = 0


@dataclass
class SourceProcessResult:
    source: SourceFile
    source_path: str
    status: str
    records: List[ReviewRecord]
    error: str = ""


def run_pipeline(
    *,
    input_dir: Path,
    output_path: Path,
    checkpoint_dir: Path,
    client_factory: Callable[[], GigaChatClient],
    logger: logging.Logger,
    limit: int | None = None,
    max_workers: int = 1,
) -> PipelineStats:
    ensure_directory(checkpoint_dir)
    ensure_directory(output_path.parent)
    checkpoint_file = checkpoint_dir / "processed_sources.txt"
    done_sources = read_processed_sources(checkpoint_file)

    all_sources = discover_source_files(input_dir)
    if limit is not None:
        all_sources = all_sources[:limit]

    stats = PipelineStats(total_files=len(all_sources))
    pass1_function = build_pass1_function()
    pass2_function = build_pass2_function()

    pending_sources: List[SourceFile] = []
    progress = tqdm(total=len(all_sources), desc="Processing reviews", unit="file")
    for source in all_sources:
        source_path = str(source.path.resolve())
        if source_path in done_sources:
            stats.skipped_files += 1
            progress.update(1)
            continue
        pending_sources.append(source)

    if max_workers <= 1:
        for source in pending_sources:
            result = _process_source(
                source=source,
                pass1_function=pass1_function,
                pass2_function=pass2_function,
                client_factory=client_factory,
            )
            _consume_result(
                result=result,
                checkpoint_file=checkpoint_file,
                done_sources=done_sources,
                output_path=output_path,
                stats=stats,
                logger=logger,
            )
            progress.set_postfix_str(source.path.name[:30], refresh=False)
            progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_source,
                    source=source,
                    pass1_function=pass1_function,
                    pass2_function=pass2_function,
                    client_factory=client_factory,
                )
                for source in pending_sources
            ]
            for future in as_completed(futures):
                result = future.result()
                _consume_result(
                    result=result,
                    checkpoint_file=checkpoint_file,
                    done_sources=done_sources,
                    output_path=output_path,
                    stats=stats,
                    logger=logger,
                )
                progress.set_postfix_str(result.source.path.name[:30], refresh=False)
                progress.update(1)

    progress.close()
    return stats


def _process_source(
    *,
    source: SourceFile,
    pass1_function,
    pass2_function,
    client_factory: Callable[[], GigaChatClient],
) -> SourceProcessResult:
    source_path = str(source.path.resolve())
    client = client_factory()
    try:
        raw_text = extract_text(source.path)
        if not raw_text.strip():
            raise RuntimeError("Extracted text is empty")

        pass1 = client.extract_structured(
            system_prompt=PASS1_SYSTEM_PROMPT,
            user_prompt=build_pass1_user_prompt(raw_text=raw_text, source_path=source_path),
            function=pass1_function,
            response_model=Pass1Result,
        )
        if not pass1.reviews:
            return SourceProcessResult(
                source=source,
                source_path=source_path,
                status="processed",
                records=[],
            )

        records = _build_records_for_source(
            source=source,
            source_path=source_path,
            raw_text=raw_text,
            pass1=pass1,
            client=client,
            pass2_function=pass2_function,
        )
        return SourceProcessResult(
            source=source,
            source_path=source_path,
            status="processed",
            records=records,
        )
    except Exception as exc:
        message = str(exc)
        if "looks like binary garbage" in message:
            return SourceProcessResult(
                source=source,
                source_path=source_path,
                status="skipped",
                records=[],
                error=message,
            )
        return SourceProcessResult(
            source=source,
            source_path=source_path,
            status="failed",
            records=[],
            error=message,
        )
    finally:
        client.close()


def _consume_result(
    *,
    result: SourceProcessResult,
    checkpoint_file: Path,
    done_sources: set[str],
    output_path: Path,
    stats: PipelineStats,
    logger: logging.Logger,
) -> None:
    if result.status == "processed":
        if result.records:
            append_jsonl(output_path, result.records)
        append_processed_source(checkpoint_file, result.source_path)
        done_sources.add(result.source_path)
        stats.processed_files += 1
        stats.output_records += len(result.records)
        logger.info("Processed %s -> %d record(s)", result.source.path.name, len(result.records))
        if not result.records:
            logger.warning("No valid reviews extracted for %s", result.source.path.name)
        return

    if result.status == "skipped":
        logger.warning("Skipping garbage source %s: %s", result.source.path.name, result.error)
        append_processed_source(checkpoint_file, result.source_path)
        done_sources.add(result.source_path)
        stats.skipped_files += 1
        return

    stats.failed_files += 1
    logger.error("Failed processing %s: %s", result.source.path, result.error)


def _build_records_for_source(
    *,
    source: SourceFile,
    source_path: str,
    raw_text: str,
    pass1: Pass1Result,
    client: GigaChatClient,
    pass2_function,
) -> List[ReviewRecord]:
    output: List[ReviewRecord] = []
    for review in pass1.reviews:
        clean_text_for_pass2 = raw_text if review.review_text_clean == "__RAW_TEXT__" else review.review_text_clean
        pass2 = client.extract_structured(
            system_prompt=PASS2_SYSTEM_PROMPT,
            user_prompt=build_pass2_user_prompt(
                clean_review_text=clean_text_for_pass2,
                book_title=review.book_title,
            ),
            function=pass2_function,
            response_model=Pass2Result,
        )

        output.append(
            ReviewRecord(
                year=source.year,
                reviewer_id=source.reviewer_id,
                book_authors=review.book_authors,
                book_title=review.book_title,
                book_reference=(review.book_reference or None),
                nomination=(review.nomination or None),
                review_text_raw=raw_text,
                review_text_clean=clean_text_for_pass2,
                sections=pass2.sections,
                rating_overall=(review.rating_overall or None),
                rating_details=(
                    {item.criterion: item.value for item in review.rating_details}
                    if review.rating_details
                    else None
                ),
                source_path=source_path,
            )
        )
    return output

