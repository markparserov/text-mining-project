from __future__ import annotations

import json
import logging
import time
import re
import ast
from typing import Any, Dict, Iterable, Type, TypeVar
import urllib.error
import urllib.request

from gigachat import GigaChat
from gigachat.exceptions import (
    AuthenticationError,
    BadRequestError,
    ForbiddenError,
    NotFoundError,
    RateLimitError,
    RequestEntityTooLargeError,
    ResponseError,
    ServerError,
    UnprocessableEntityError,
)
from gigachat.models import Chat, Function, Messages
from pydantic import BaseModel, ValidationError
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


T = TypeVar("T", bound=BaseModel)


class LLMValidationError(RuntimeError):
    pass


class GigaChatClient:
    def __init__(
        self,
        *,
        credentials: str,
        scope: str,
        model: str,
        verify_ssl_certs: bool,
        max_retries: int,
        request_delay_seconds: float,
        request_timeout_seconds: float,
        use_vllm_only: bool,
        enable_vllm_fallback: bool,
        vllm_base_url: str,
        vllm_model: str,
        vllm_api_key: str,
        logger: logging.Logger,
    ) -> None:
        self._use_vllm_only = use_vllm_only
        self._client: GigaChat | None = None
        if not self._use_vllm_only:
            self._client = GigaChat(
                credentials=credentials,
                scope=scope,
                model=model,
                verify_ssl_certs=verify_ssl_certs,
                timeout=request_timeout_seconds,
            )
        self._model = model
        self._max_retries = max_retries
        self._request_delay_seconds = request_delay_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self._logger = logger
        self._enable_vllm_fallback = enable_vllm_fallback
        self._vllm_base_url = vllm_base_url.rstrip("/")
        self._vllm_model = vllm_model
        self._vllm_api_key = vllm_api_key

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def extract_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        function: Function,
        response_model: Type[T],
    ) -> T:
        if self._use_vllm_only:
            payload = self._vllm_structured_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                function=function,
            )
            try:
                return response_model.model_validate(payload)
            except ValidationError as exc:
                raise RuntimeError(f"vLLM structured call schema validation failed: {exc}") from exc

        retryer = Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    ServerError,
                    ResponseError,
                    LLMValidationError,
                    UnprocessableEntityError,
                )
            ),
            reraise=True,
        )

        try:
            for attempt in retryer:
                with attempt:
                    payload = self._single_function_call(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        function=function,
                    )
                    try:
                        return response_model.model_validate(payload)
                    except ValidationError as exc:
                        err = f"Schema validation failed: {exc}"
                        self._logger.warning(err)
                        raise LLMValidationError(err) from exc
        except (
            RetryError,
            AuthenticationError,
            BadRequestError,
            ForbiddenError,
            NotFoundError,
            RequestEntityTooLargeError,
        ) as exc:
            if self._enable_vllm_fallback:
                return self._extract_with_vllm_fallback(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    function=function,
                    response_model=response_model,
                    previous_error=exc,
                )
            raise RuntimeError(f"GigaChat request failed: {exc}") from exc

        except (RateLimitError, ServerError, ResponseError, LLMValidationError, UnprocessableEntityError) as exc:
            if self._enable_vllm_fallback:
                return self._extract_with_vllm_fallback(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    function=function,
                    response_model=response_model,
                    previous_error=exc,
                )
            raise RuntimeError(f"GigaChat request failed: {exc}") from exc

        raise RuntimeError("Unreachable state in extract_structured")

    def _extract_with_vllm_fallback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        function: Function,
        response_model: Type[T],
        previous_error: Exception,
    ) -> T:
        self._logger.warning(
            "GigaChat failed (%s). Trying vLLM fallback %s (%s).",
            previous_error,
            self._vllm_base_url,
            self._vllm_model,
        )
        payload = self._vllm_structured_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            function=function,
        )
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(f"vLLM fallback schema validation failed: {exc}") from exc

    def _vllm_structured_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        function: Function,
    ) -> Dict[str, Any] | list[Any]:
        schema_obj = function.return_parameters or function.parameters
        schema: Dict[str, Any] = {}
        if hasattr(schema_obj, "model_dump"):
            schema = schema_obj.model_dump(by_alias=True, exclude_none=True)
        elif isinstance(schema_obj, dict):
            schema = dict(schema_obj)

        fallback_system = (
            f"{system_prompt}\n\n"
            "Ты работаешь в строгом JSON-режиме. Верни ТОЛЬКО валидный JSON без markdown и пояснений.\n"
            "Формат JSON должен точно соответствовать этой схеме:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        payload = {
            "model": self._vllm_model,
            "messages": [
                {"role": "system", "content": fallback_system},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self._vllm_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_api_key}"
        request = urllib.request.Request(
            url=f"{self._vllm_base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM fallback request failed: {exc}") from exc

        try:
            parsed_body = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"vLLM fallback returned non-JSON response: {exc}") from exc

        result = _extract_json_payload_from_vllm_response(parsed_body)
        if isinstance(result, (dict, list)):
            return result

        # Repair pass: ask model to convert the malformed output to strict JSON.
        malformed = _extract_first_candidate_text(parsed_body)
        repaired = self._vllm_repair_content_to_json(
            malformed_content=malformed,
            schema=schema,
        )
        if isinstance(repaired, (dict, list)):
            return repaired

        preview = (malformed or "")[:600].replace("\n", "\\n")
        self._logger.warning("Unparseable vLLM payload preview: %s", preview)
        raise RuntimeError("vLLM fallback did not return valid JSON payload")

    def _vllm_repair_content_to_json(
        self,
        *,
        malformed_content: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any] | list[Any] | None:
        if not malformed_content:
            return None

        repair_prompt = (
            "Преобразуй текст ниже в ВАЛИДНЫЙ JSON строго по схеме. "
            "Верни только JSON, без markdown и пояснений.\n\n"
            f"Схема:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Текст:\n{malformed_content}"
        )
        payload = {
            "model": self._vllm_model,
            "messages": [
                {"role": "system", "content": "Ты JSON-ремонтник. Возвращай только валидный JSON."},
                {"role": "user", "content": repair_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self._vllm_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_api_key}"
        request = urllib.request.Request(
            url=f"{self._vllm_base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout_seconds) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
        except Exception:
            return None
        return _extract_json_payload_from_vllm_response(parsed)

    def _single_function_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        function: Function,
    ) -> Dict[str, Any] | list[Any]:
        if self._client is None:
            raise RuntimeError("GigaChat client is unavailable in vLLM-only mode")
        if self._request_delay_seconds > 0:
            time.sleep(self._request_delay_seconds)

        chat = Chat(
            model=self._model,
            messages=[
                Messages(role="system", content=system_prompt),
                Messages(role="user", content=user_prompt),
            ],
            functions=[function],
            function_call={"name": function.name},
            temperature=0.001,
            stream=False,
        )

        response = self._client.chat(chat)
        if not response.choices:
            raise LLMValidationError("Empty choices in LLM response")

        message = response.choices[0].message
        function_call = message.function_call
        if function_call is None:
            content = message.content or ""
            parsed_from_content = _try_parse_json_from_content(content)
            if parsed_from_content is not None:
                return parsed_from_content
            fallback = self._fallback_plain_json_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            if fallback is not None:
                return fallback
            raise LLMValidationError("Model did not return function_call")
        if function_call.name != function.name:
            raise LLMValidationError(
                f"Unexpected function name: {function_call.name}, expected {function.name}"
            )

        arguments = function_call.arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise LLMValidationError(f"Function arguments are not valid JSON: {exc}") from exc
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            raise LLMValidationError(f"Unsupported function arguments type: {type(arguments)!r}")

        if not isinstance(parsed, dict):
            raise LLMValidationError("Function arguments must be a JSON object")
        # Preferred path: function arguments already contain the target object.
        if "payload_json" not in parsed:
            return parsed

        # Legacy compatibility: some model replies still wrap data in payload_json string.
        if "payload_json" in parsed:
            payload_json = parsed.get("payload_json")
            if isinstance(payload_json, (dict, list)):
                return payload_json
            if not isinstance(payload_json, str):
                raise LLMValidationError("payload_json must be a string or JSON object")
            try:
                parsed_payload = json.loads(payload_json)
            except json.JSONDecodeError as exc:
                extracted = _extract_first_json_value(payload_json)
                if extracted is not None:
                    parsed_payload = extracted
                else:
                    literal_obj = _try_parse_python_literal(payload_json)
                    if literal_obj is not None:
                        parsed_payload = literal_obj
                    else:
                        relaxed = _try_parse_relaxed_json(payload_json)
                        if relaxed is not None:
                            parsed_payload = relaxed
                        else:
                            repaired = self._repair_json_payload(payload_json)
                            if repaired is None:
                                self._logger.warning(
                                    "Unparseable payload_json sample: %s",
                                    payload_json[:400].replace("\n", "\\n"),
                                )
                                raise LLMValidationError(f"payload_json is not valid JSON: {exc}") from exc
                            parsed_payload = repaired
            if not isinstance(parsed_payload, (dict, list)):
                raise LLMValidationError("Decoded payload_json must be a JSON object or array")
            return parsed_payload
        return parsed

    def _fallback_plain_json_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any] | list[Any] | None:
        if self._client is None:
            raise RuntimeError("GigaChat client is unavailable in vLLM-only mode")
        fallback_system = (
            f"{system_prompt}\n\n"
            "ВНИМАНИЕ: инструмент недоступен. Верни только валидный JSON-объект без пояснений и markdown."
        )
        chat = Chat(
            model=self._model,
            messages=[
                Messages(role="system", content=fallback_system),
                Messages(role="user", content=user_prompt),
            ],
            temperature=0.001,
            stream=False,
        )
        response = self._client.chat(chat)
        if not response.choices:
            return None
        content = response.choices[0].message.content or ""
        parsed = _try_parse_json_from_content(content)
        if parsed is not None:
            return parsed

        repair_chat = Chat(
            model=self._model,
            messages=[
                Messages(
                    role="system",
                    content="Преобразуй текст в валидный JSON-объект. Верни только JSON без пояснений.",
                ),
                Messages(role="user", content=content),
            ],
            temperature=0.001,
            stream=False,
        )
        repair_response = self._client.chat(repair_chat)
        if not repair_response.choices:
            return None
        repaired_content = repair_response.choices[0].message.content or ""
        return _try_parse_json_from_content(repaired_content)

    def _repair_json_payload(self, malformed_payload: str) -> Dict[str, Any] | list[Any] | None:
        if self._client is None:
            raise RuntimeError("GigaChat client is unavailable in vLLM-only mode")
        repair_chat = Chat(
            model=self._model,
            messages=[
                Messages(
                    role="system",
                    content="Исправь испорченный JSON и верни только валидный JSON-объект.",
                ),
                Messages(role="user", content=malformed_payload),
            ],
            temperature=0.001,
            stream=False,
        )
        response = self._client.chat(repair_chat)
        if not response.choices:
            return None
        content = response.choices[0].message.content or ""
        parsed = _try_parse_json_from_content(content)
        return parsed if isinstance(parsed, (dict, list)) else None


def _try_parse_json_from_content(content: str) -> Dict[str, Any] | list[Any] | None:
    if content is None:
        return None
    content = content.strip()
    if not content:
        return None
    candidates = [content]
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    candidates.extend(fenced)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, (dict, list)):
                if isinstance(obj, dict) and "payload_json" in obj and isinstance(obj["payload_json"], str):
                    inner = _extract_first_json_value(obj["payload_json"])
                    if isinstance(inner, (dict, list)):
                        return inner
                return obj
        except Exception:
            extracted = _extract_first_json_value(candidate)
            if isinstance(extracted, (dict, list)):
                return extracted
            literal_obj = _try_parse_python_literal(candidate)
            if isinstance(literal_obj, (dict, list)):
                return literal_obj
            relaxed = _try_parse_relaxed_json(candidate)
            if isinstance(relaxed, (dict, list)):
                return relaxed
            continue
    return None


def _extract_first_json_value(text: str) -> Dict[str, Any] | list[Any] | None:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not starts:
        return None
    start = min(starts)
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                fragment = text[start : idx + 1]
                try:
                    obj = json.loads(fragment)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, (dict, list)) else None
    return None


def _try_parse_python_literal(text: str) -> Dict[str, Any] | list[Any] | None:
    try:
        obj = ast.literal_eval(text)
    except Exception:
        return None
    return obj if isinstance(obj, (dict, list)) else None


def _try_parse_relaxed_json(text: str) -> Dict[str, Any] | list[Any] | None:
    base = text.strip()
    if not base:
        return None
    candidates = [base]
    if '\\"' in base:
        candidates.append(base.replace('\\"', '"'))

    for candidate in candidates:
        normalized = candidate
        # Quote bare keys: { foo: "bar" } -> { "foo": "bar" }
        normalized = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', normalized)
        # Replace single quotes with double quotes.
        normalized = normalized.replace("'", '"')
        # Fix duplicated quote before colon in keys: "key"": -> "key":
        normalized = re.sub(r'"([^"]+)""\s*:', r'"\1":', normalized)
        # Remove trailing commas before closing braces/brackets.
        normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
        try:
            obj = json.loads(normalized)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, (dict, list)):
            return obj
    return None


def _extract_json_payload_from_vllm_response(parsed_body: object) -> Dict[str, Any] | list[Any] | None:
    if not isinstance(parsed_body, dict):
        return None
    choices = parsed_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None
    message = choice0.get("message")
    if not isinstance(message, dict):
        return None

    candidates: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        candidates.append(content)
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        candidates.append(reasoning)

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        arguments = function_call.get("arguments")
        if isinstance(arguments, str):
            candidates.append(arguments)

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                candidates.append(arguments)

    for candidate in _dedupe_nonempty(candidates):
        parsed = _try_parse_json_from_content(candidate)
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _extract_first_candidate_text(parsed_body: object) -> str:
    if not isinstance(parsed_body, dict):
        return ""
    choices = parsed_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return ""
    message = choice0.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        arguments = function_call.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            return arguments
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments
    return ""


def _dedupe_nonempty(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        trimmed = value.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        out.append(trimmed)
    return out

