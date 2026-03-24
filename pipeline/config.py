from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class PipelineConfig:
    gigachat_auth_token: str
    gigachat_scope: str
    input_dir: Path
    output_path: Path
    checkpoint_dir: Path
    model: str
    request_delay_seconds: float
    max_retries: int
    verify_ssl_certs: bool
    request_timeout_seconds: float
    use_vllm_only: bool
    enable_vllm_fallback: bool
    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str


def load_config(
    env_file: Path,
    input_dir: Path,
    output_path: Path,
    checkpoint_dir: Path,
    model: str,
    request_delay_seconds: float,
    max_retries: int,
    verify_ssl_certs: bool,
    request_timeout_seconds: float,
    use_vllm_only: bool,
    enable_vllm_fallback: bool,
    vllm_base_url: str,
    vllm_model: str,
    vllm_api_key: str,
) -> PipelineConfig:
    load_dotenv(env_file)
    token = os.getenv("GIGACHAT_AUTH_TOKEN", "").strip().strip('"')
    scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip().strip('"')
    if not use_vllm_only and not token:
        raise ValueError("GIGACHAT_AUTH_TOKEN is not set in .env")

    return PipelineConfig(
        gigachat_auth_token=token,
        gigachat_scope=scope or "GIGACHAT_API_PERS",
        input_dir=input_dir,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
        model=model,
        request_delay_seconds=request_delay_seconds,
        max_retries=max_retries,
        verify_ssl_certs=verify_ssl_certs,
        request_timeout_seconds=request_timeout_seconds,
        use_vllm_only=use_vllm_only,
        enable_vllm_fallback=enable_vllm_fallback,
        vllm_base_url=(os.getenv("VLLM_BASE_URL", vllm_base_url).strip() or vllm_base_url),
        vllm_model=(os.getenv("VLLM_MODEL", vllm_model).strip() or vllm_model),
        vllm_api_key=(os.getenv("VLLM_API_KEY", vllm_api_key).strip() or vllm_api_key),
    )

