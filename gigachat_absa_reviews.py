#!/usr/bin/env python3
"""Извлечение аспектного сентимента из книжных рецензий через GigaChat"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Function, FunctionParameters, Messages, MessagesRole
from tqdm import tqdm

PROMPT_VERSION = "gigachat_absa_v3"
FUNCTION_NAME = "extract_book_absa"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_idx} in {path}: {exc}") from exc
            records.append(item)
    return records


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_review_text(record: Dict[str, Any]) -> str:
    for key in ("review_text_clean", "review_text_raw", "review_text"):
        value = record.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def build_absa_prompt(text: str) -> str:
    return (
        "Задача: проанализировать рецензию на книгу (экспертную оценку для книжной премии) "
        "и извлечь из неё мнения рецензента непосредственно об этой книге.\n\n"
        "Каждое мнение — кортеж из трёх составляющих:\n"
        "- Target: объект мнения — аспект книги, именованная сущность или понятие из текста.\n"
        "- Polarity: POS (положительное) или NEG (отрицательное). Нейтральные мнения не включай.\n"
        "- Expression: одна или несколько фраз из текста рецензии, на основании которых "
        "определена тональность.\n\n"
        "Типичные аспекты (Target), которые могут детерминировать оценку книги "
        "рецензентом премии:\n"
        "- Актуальность и значимость темы\n"
        "- Теоретическая глубина и оригинальность\n"
        "- Методология исследования\n"
        "- Ценность и качество эмпирического материала\n"
        "- Качество аргументации и обоснованность выводов\n"
        "- Оригинальность и значимость полученных результатов\n"
        "- Структура и логика изложения\n"
        "- Качество языка и стиля\n"
        "- Качество перевода (для переводных работ)\n"
        "- Качество издательского оформления (сноски, библиография)\n"
        "- Наличие и качество сопроводительной / вступительной статьи\n"
        "- Вписанность в академические дискуссии (российский и международный контекст)\n"
        "- Дисциплинарная принадлежность\n"
        "Этот список не исчерпывающий — извлекай любые аспекты, о которых "
        "рецензент высказывает мнение.\n\n"
        "Верни результат СТРОГО в виде JSON-объекта по схеме:\n"
        "{\n"
        '  "schema_version": "1.0",\n'
        '  "language": "ru",\n'
        '  "review_sentiment_tuples": [\n'
        "    {\n"
        '      "Target": "строка",\n'
        '      "Polarity": "POS или NEG",\n'
        '      "Expression": ["строка1", "строка2"],\n'
        '      "Evidence": "краткое пояснение (опционально)"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Ограничения:\n"
        "1) Выделяй только POS/NEG, нейтральные мнения не включай.\n"
        "2) Target — аспект книги или понятие из текста рецензии.\n"
        "3) Каждый элемент Expression — фраза из текста рецензии, выражающая мнение.\n"
        "4) Удаляй дубликаты одинаковых (Target, Polarity, Expression).\n"
        "5) Если валидных кортежей нет, верни пустой массив review_sentiment_tuples.\n"
        "6) Не добавляй никаких ключей, кроме schema_version, language, "
        "review_sentiment_tuples, Target, Polarity, Expression, Evidence.\n"
        "7) Никакого markdown, комментариев и поясняющего текста вне JSON.\n\n"
        "Примеры:\n\n"
        "1) ***Текст***\n"
        "Работа выполнена на достойном для своей области уровне.\n"
        "Target: работа, Polarity: POS, Expression: выполнена на достойном "
        "для своей области уровне\n\n"
        "2) ***Текст***\n"
        "Книгу отличает политическая ангажированность, полное отсутствие "
        "авторской нейтральности и непредвзятости.\n"
        "Target: книгу, Polarity: NEG, Expression: отличает политическая "
        "ангажированность, полное отсутствие авторской нейтральности и непредвзятости\n\n"
        "3) ***Текст***\n"
        "Тема актуальная, но интерпретация предвзятая, заведомо смещенная "
        "в сторону идеологического официоза. В международный контекст не вписано "
        "никак. Эмпирический материал сомнительный, и превратно интерпретируемый.\n"
        "Target: тема, Polarity: POS, Expression: актуальная\n"
        "Target: интерпретация, Polarity: NEG, Expression: предвзятая, заведомо "
        "смещенная в сторону идеологического официоза\n"
        "Target: книга, Polarity: NEG, Expression: в международный контекст "
        "не вписано никак\n"
        "Target: эмпирический материал, Polarity: NEG, Expression: сомнительный, "
        "и превратно интерпретируемый\n\n"
        "4) ***Текст***\n"
        "Первые главы читать обязательно, в них содержится объяснение фрилансерства "
        "как феномена современного рынка труда, а также очень подробно (можно сказать, "
        "эталонно) объясняется, на каких данных основан эмпирический анализ. Читать "
        "это все в какой-то момент становится не то, чтобы скучно, но долго, тягуче.\n"
        "Target: первые главы, Polarity: POS, Expression: читать обязательно\n"
        "Target: эмпирический анализ, Polarity: POS, Expression: очень подробно "
        "(можно сказать, эталонно) объясняется, на каких данных основан\n"
        "Target: книга, Polarity: NEG, Expression: читать это все в какой-то момент "
        "становится не то, чтобы скучно, но долго, тягуче\n\n"
        "5) ***Текст***\n"
        "Краткая вступительная статья безусловно помогает читателю войти "
        "в интеллектуальное пространство работы Адорно, но всё-таки не в полной "
        "мере раскрывает позицию Minima moralia. Издание безусловно должно войти "
        "в дальнейшие списки.\n"
        "Target: вступительная статья, Polarity: POS, Expression: помогает "
        "читателю войти в интеллектуальное пространство\n"
        "Target: вступительная статья, Polarity: NEG, Expression: не в полной "
        "мере раскрывает позицию\n"
        "Target: издание, Polarity: POS, Expression: безусловно должно войти "
        "в дальнейшие списки\n\n"
        "6) ***Текст***\n"
        "Книга крайне фрагментирована. Она как состояла из различных текстов, "
        "так и осталась. Материал собран значительный, но обоснование "
        "методологических решений часто отсутствует.\n"
        "Target: книга, Polarity: NEG, Expression: крайне фрагментирована\n"
        "Target: материал, Polarity: POS, Expression: собран значительный\n"
        "Target: методологические решения, Polarity: NEG, Expression: обоснование "
        "часто отсутствует\n\n"
        "7) ***Текст***\n"
        "В качестве реферативного обобщения книга прекрасна. Содержательные выводы "
        "менее выражены. Книга предваряется развернутым вступлением одного из "
        "рецензентов. Одно из лучших мест книги.\n"
        "Target: книга, Polarity: POS, Expression: в качестве реферативного "
        "обобщения книга прекрасна\n"
        "Target: содержательные выводы, Polarity: NEG, Expression: менее выражены\n"
        "Target: вступление, Polarity: POS, Expression: одно из лучших мест книги\n\n"
        "8) ***Текст***\n"
        "Это не социологическое исследование, а встревоженное эссе автора, "
        "который решил поспекулировать о зависимости от интернет-платформ. "
        "Автор вбрасывает факты, которые в социологическом исследовании "
        "требовали бы проверки. Перевод, к сожалению, только усиливает это "
        "ощущение. Вёрстка ужасная.\n"
        "Target: книга, Polarity: NEG, Expression: не социологическое "
        "исследование, а встревоженное эссе\n"
        "Target: аргументация, Polarity: NEG, Expression: вбрасывает факты, "
        "которые в социологическом исследовании требовали бы проверки\n"
        "Target: перевод, Polarity: NEG, Expression: только усиливает это ощущение\n"
        "Target: вёрстка, Polarity: NEG, Expression: ужасная\n\n"
        "Проверь перед ответом:\n"
        "- Ответ валидный JSON.\n"
        "- Polarity только POS/NEG.\n"
        "- Target и Expression встречаются в исходном тексте дословно.\n\n"
        f"Текст рецензии:\n{text}"
    )


def build_repair_prompt(original_prompt: str, previous_output: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "Предыдущий ответ был невалиден JSON.\n"
        "Исправь его и верни только валидный JSON строго по указанной схеме.\n"
        "Без markdown и без любого текста вне JSON.\n\n"
        f"Невалидный предыдущий ответ:\n{previous_output}"
    )


def _build_function_schema() -> Function:
    return Function(
        name=FUNCTION_NAME,
        description="Извлекает аспектно-сентиментные кортежи из рецензии на книгу.",
        parameters=FunctionParameters(
            type="object",
            properties={
                "payload_json": {
                    "type": "string",
                    "description": "JSON-строка c root-объектом schema_version/language/review_sentiment_tuples",
                }
            },
            required=["payload_json"],
        ),
        return_parameters={
            "type": "object",
            "properties": {"payload_json": {"type": "string"}},
            "required": ["payload_json"],
        },
    )


def _extract_payload_json(completion: Any) -> Tuple[Optional[str], Optional[str]]:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return None, "No choices returned by model."

    message = getattr(choices[0], "message", None)
    if message is None:
        return None, "No message in first choice."

    function_call = getattr(message, "function_call", None)
    if function_call is not None:
        arguments = getattr(function_call, "arguments", None)
        if arguments is None:
            return None, "Function call has no arguments."
        parsed_args: Any = arguments
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return None, f"Function arguments are not valid JSON: {exc}"
        if isinstance(parsed_args, dict):
            payload = parsed_args.get("payload_json")
            if payload is None:
                return None, "Function arguments do not contain payload_json."
            if isinstance(payload, dict):
                return json.dumps(payload, ensure_ascii=False), None
            return str(payload), None
        return None, "Function arguments are not a JSON object."

    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip(), None

    return None, "No function_call and no textual content."


def _parse_payload_object(payload_json: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        return None, f"payload_json is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "payload_json must be a JSON object."
    return payload, None


def _parse_payload_object_with_repair(payload_json: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload_obj, parse_error = _parse_payload_object(payload_json)
    if payload_obj is not None:
        return payload_obj, None

    text = payload_json.strip()
    if not text:
        return None, parse_error

    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    candidates: List[str] = [text]
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidates.append(text[left : right + 1])

    if left != -1 and right == -1:
        # пробуем восстановить обрезанный объект из аргументов tool-call
        candidates.append(text[left:] + "}")

    for candidate in candidates:
        payload_obj, candidate_error = _parse_payload_object(candidate)
        if payload_obj is not None:
            return payload_obj, None
        parse_error = candidate_error

    return None, parse_error


def _find_absa_root(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        tuples = item.get("review_sentiment_tuples")
        if isinstance(tuples, list):
            return item
        for value in item.values():
            found = _find_absa_root(value)
            if found is not None:
                return found
    elif isinstance(item, list):
        for value in item:
            found = _find_absa_root(value)
            if found is not None:
                return found
    return None


def _normalize_tuples(payload: Dict[str, Any], source_text: str) -> List[Dict[str, Any]]:
    del source_text
    tuples_raw = payload.get("review_sentiment_tuples", [])
    if not isinstance(tuples_raw, list):
        return []

    normalized: List[Dict[str, Any]] = []
    seen = set()

    for item in tuples_raw:
        if not isinstance(item, dict):
            continue
        target = item.get("Target")
        polarity = item.get("Polarity")
        expression = item.get("Expression")

        if not isinstance(target, str) or not target.strip():
            continue
        if not isinstance(polarity, str):
            continue
        polarity_up = polarity.strip().upper()
        if polarity_up not in {"POS", "NEG"}:
            continue

        expressions: List[str]
        if isinstance(expression, str):
            expressions = [expression]
        elif isinstance(expression, list):
            expressions = [e for e in expression if isinstance(e, str) and e.strip()]
        else:
            continue

        target = target.strip()
        expressions = [e.strip() for e in expressions if e.strip()]
        if not expressions:
            continue

        key = (target, polarity_up, tuple(expressions))
        if key in seen:
            continue
        seen.add(key)

        normalized.append(
            {
                "target": target,
                "polarity": polarity_up,
                "expressions": expressions,
            }
        )

    return normalized


def call_gigachat_absa(
    client: GigaChat,
    model: str,
    temperature: float,
    review_text: str,
    max_retries: int,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    function_schema = _build_function_schema()
    base_prompt = build_absa_prompt(review_text)
    prompt = base_prompt
    last_error: Optional[str] = None
    raw_payload_json: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat(
                Chat(
                    model=model,
                    temperature=temperature,
                    messages=[
                        Messages(
                            role=MessagesRole.SYSTEM,
                            content="Ты извлекаешь аспектные мнения и возвращаешь только JSON по заданной схеме.",
                        ),
                        Messages(role=MessagesRole.USER, content=prompt),
                    ],
                    functions=[function_schema],
                    function_call={"name": FUNCTION_NAME},
                )
            )
            payload_json, extract_error = _extract_payload_json(completion)
            if extract_error:
                last_error = extract_error
                continue
            if payload_json is None:
                last_error = "Model returned empty payload_json."
                continue
            raw_payload_json = payload_json

            payload_obj, parse_error = _parse_payload_object_with_repair(payload_json)
            if parse_error:
                last_error = parse_error
                prompt = build_repair_prompt(base_prompt, payload_json)
                continue
            if payload_obj is None:
                last_error = "payload_json parsed to empty object."
                continue
            payload_root = _find_absa_root(payload_obj)
            if payload_root is None:
                last_error = "payload_json does not contain required key review_sentiment_tuples."
                prompt = build_repair_prompt(base_prompt, payload_json)
                continue

            normalized = _normalize_tuples(payload_root, review_text)
            return normalized, None, raw_payload_json
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"

    return [], last_error, raw_payload_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aspect-based sentiment extraction for preprocessed book reviews via GigaChat-Max."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to input reviews JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Path to output JSONL.")
    parser.add_argument("--model", default="GigaChat-Max", help="GigaChat model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for failed calls.")
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="Client timeout in seconds.")
    parser.add_argument(
        "--enable-ssl-verify",
        action="store_true",
        help="Enable SSL certificate verification for API requests (disabled by default).",
    )
    parser.add_argument(
        "--sleep-between-requests-ms",
        type=int,
        default=100,
        help="Sleep between requests in milliseconds.",
    )
    parser.add_argument(
        "--debug-save-raw",
        action="store_true",
        help="Store raw payload_json in output rows.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Autosave output JSONL every N processed reviews (0 disables autosave).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()

    records = load_jsonl(input_path)
    if not records:
        raise ValueError(f"No records loaded from {input_path}")

    rows: List[Dict[str, Any]] = []
    with GigaChat(
        model=args.model,
        timeout=args.timeout_sec,
        verify_ssl_certs=args.enable_ssl_verify,
    ) as client:
        for rec in tqdm(records, desc="ABSA reviews", unit="review"):
            review_text = _extract_review_text(rec)
            if not review_text:
                out_row = {
                    **rec,
                    "absa_items": [],
                    "absa_error": None,
                    "absa_model": args.model,
                    "absa_prompt_version": PROMPT_VERSION,
                }
                if args.debug_save_raw:
                    out_row["raw_payload_json"] = None
                rows.append(out_row)
                if args.save_every > 0 and len(rows) % args.save_every == 0:
                    write_jsonl(output_path, rows)
                continue

            absa_items, absa_error, raw_payload_json = call_gigachat_absa(
                client=client,
                model=args.model,
                temperature=args.temperature,
                review_text=review_text,
                max_retries=max(0, args.max_retries),
            )
            out_row = {
                **rec,
                "absa_items": absa_items,
                "absa_error": absa_error,
                "absa_model": args.model,
                "absa_prompt_version": PROMPT_VERSION,
            }
            if args.debug_save_raw:
                out_row["raw_payload_json"] = raw_payload_json
            rows.append(out_row)
            if args.save_every > 0 and len(rows) % args.save_every == 0:
                write_jsonl(output_path, rows)
            if args.sleep_between_requests_ms > 0:
                time.sleep(args.sleep_between_requests_ms / 1000.0)

    write_jsonl(output_path, rows)


if __name__ == "__main__":
    main()