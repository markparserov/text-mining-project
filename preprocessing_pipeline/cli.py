from __future__ import annotations

import logging
from pathlib import Path

import click
from tqdm import tqdm

from preprocessing_pipeline.config import load_config
from preprocessing_pipeline.gigachat_client import GigaChatClient
from preprocessing_pipeline.processing import run_pipeline


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logger(log_level: str) -> logging.Logger:
    logger = logging.getLogger("textmining-pipeline")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


@click.group()
def cli() -> None:
    """LLM preprocessing pipeline for review documents."""


@cli.command()
@click.option("--input-dir", default="тексты", show_default=True, type=click.Path(path_type=Path))
@click.option("--output", "output_path", default="output.jsonl", show_default=True, type=click.Path(path_type=Path))
@click.option("--env-file", default=".env", show_default=True, type=click.Path(path_type=Path))
@click.option("--checkpoint-dir", default=".checkpoints", show_default=True, type=click.Path(path_type=Path))
@click.option("--model", default="GigaChat-2-Max", show_default=True, type=str)
@click.option("--delay", "request_delay_seconds", default=1.0, show_default=True, type=float)
@click.option("--timeout", "request_timeout_seconds", default=120.0, show_default=True, type=float)
@click.option("--max-retries", default=3, show_default=True, type=int)
@click.option("--verify-ssl-certs/--no-verify-ssl-certs", default=False, show_default=True)
@click.option("--use-vllm-only/--disable-use-vllm-only", default=False, show_default=True)
@click.option("--enable-vllm-fallback/--disable-vllm-fallback", default=True, show_default=True)
@click.option("--vllm-base-url", default="http://127.0.0.1:8004/v1", show_default=True, type=str)
@click.option("--vllm-model", default="qwen3.5", show_default=True, type=str)
@click.option("--vllm-api-key", default="EMPTY", show_default=True, type=str)
@click.option("--max-workers", default=1, show_default=True, type=click.IntRange(1, 64))
@click.option("--log-level", default="INFO", show_default=True, type=str)
@click.option("--limit", default=None, type=int, help="Process only first N files")
def process(
    input_dir: Path,
    output_path: Path,
    env_file: Path,
    checkpoint_dir: Path,
    model: str,
    request_delay_seconds: float,
    request_timeout_seconds: float,
    max_retries: int,
    verify_ssl_certs: bool,
    use_vllm_only: bool,
    enable_vllm_fallback: bool,
    vllm_base_url: str,
    vllm_model: str,
    vllm_api_key: str,
    max_workers: int,
    log_level: str,
    limit: int | None,
) -> None:
    """Run preprocessing and export structured JSONL."""
    logger = setup_logger(log_level=log_level)
    config = load_config(
        env_file=env_file,
        input_dir=input_dir,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
        model=model,
        request_delay_seconds=request_delay_seconds,
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
        verify_ssl_certs=verify_ssl_certs,
        enable_vllm_fallback=enable_vllm_fallback,
        use_vllm_only=use_vllm_only,
        vllm_base_url=vllm_base_url,
        vllm_model=vllm_model,
        vllm_api_key=vllm_api_key,
    )
    stats = run_pipeline(
        input_dir=config.input_dir,
        output_path=config.output_path,
        checkpoint_dir=config.checkpoint_dir,
        client_factory=lambda: GigaChatClient(
            credentials=config.gigachat_auth_token,
            scope=config.gigachat_scope,
            model=config.model,
            verify_ssl_certs=config.verify_ssl_certs,
            max_retries=config.max_retries,
            request_delay_seconds=config.request_delay_seconds,
            request_timeout_seconds=config.request_timeout_seconds,
            use_vllm_only=config.use_vllm_only,
            enable_vllm_fallback=config.enable_vllm_fallback,
            vllm_base_url=config.vllm_base_url,
            vllm_model=config.vllm_model,
            vllm_api_key=config.vllm_api_key,
            logger=logger,
        ),
        logger=logger,
        limit=limit,
        max_workers=max_workers,
    )

    logger.info(
        "Done. total=%d processed=%d skipped=%d failed=%d records=%d output=%s",
        stats.total_files,
        stats.processed_files,
        stats.skipped_files,
        stats.failed_files,
        stats.output_records,
        str(config.output_path),
    )


if __name__ == "__main__":
    cli()

