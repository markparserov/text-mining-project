from __future__ import annotations

from gigachat.models import Function, FunctionParameters


PASS1_SYSTEM_PROMPT = """Ты помогаешь предобрабатывать корпус рецензий на книги.
Твоя задача:
1) удалить нерелевантные и технические части текста;
2) разделить документ на отдельные рецензии по книгам;
3) извлечь метаданные каждой отдельной рецензии.

Считай нерелевантным:
- описания шкал оценивания 10/9/8/...;
- общие шаблоны критериев для номинаций;
- служебные заголовки и артефакты форматирования;
- повторы, не относящиеся к содержательной части рецензии.

Сохраняй:
- саму рецензию и аргументацию рецензента;
- названия книг, авторов, библиографические сведения;
- номинацию (если указана);
- общую оценку и оценки по критериям (если указаны).

Важно:
- если в документе несколько книг, верни параллельные массивы одинаковой длины (элементы с одним индексом относятся к одной книге);
- ничего не придумывай;
- если поле не найдено: верни пустую строку "";
- rating_details_serialized: строка формата "критерий: значение; критерий: значение";
- возвращай данные ТОЛЬКО через вызов функции."""


PASS2_SYSTEM_PROMPT = """Ты сегментируешь уже очищенный текст одной рецензии на тематические разделы.
Для каждого раздела верни:
- title: короткий осмысленный заголовок;
- description: краткое описание содержания раздела;
- text: полный текст раздела.

Правила:
- разделы должны идти в исходном порядке;
- не теряй факты и смысл;
- не добавляй информацию от себя;
- верни три параллельных массива одинаковой длины:
  section_titles, section_descriptions, section_texts;
- возвращай данные ТОЛЬКО через вызов функции."""


def build_pass1_function() -> Function:
    return Function(
        name="extract_reviews",
        description="Извлекает и очищает рецензии на книги из исходного документа.",
        parameters=FunctionParameters(
            type_="object",
            properties={
                "book_authors_csv": {"type": "array", "items": {"type": "string"}},
                "book_titles": {"type": "array", "items": {"type": "string"}},
                "book_references": {"type": "array", "items": {"type": "string"}},
                "nominations": {"type": "array", "items": {"type": "string"}},
                "rating_overalls": {"type": "array", "items": {"type": "string"}},
                "rating_details_serialized": {"type": "array", "items": {"type": "string"}},
                "review_texts_clean": {"type": "array", "items": {"type": "string"}},
            },
            required=[
                "book_authors_csv",
                "book_titles",
                "book_references",
                "nominations",
                "rating_overalls",
                "rating_details_serialized",
                "review_texts_clean",
            ],
        ),
        return_parameters={
            "type": "object",
            "properties": {
                "book_authors_csv": {"type": "array", "items": {"type": "string"}},
                "book_titles": {"type": "array", "items": {"type": "string"}},
                "book_references": {"type": "array", "items": {"type": "string"}},
                "nominations": {"type": "array", "items": {"type": "string"}},
                "rating_overalls": {"type": "array", "items": {"type": "string"}},
                "rating_details_serialized": {"type": "array", "items": {"type": "string"}},
                "review_texts_clean": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "book_authors_csv",
                "book_titles",
                "book_references",
                "nominations",
                "rating_overalls",
                "rating_details_serialized",
                "review_texts_clean",
            ],
        },
    )


def build_pass2_function() -> Function:
    return Function(
        name="segment_review",
        description="Сегментирует очищенный текст рецензии на тематические разделы.",
        parameters=FunctionParameters(
            type_="object",
            properties={
                "section_titles": {"type": "array", "items": {"type": "string"}},
                "section_descriptions": {"type": "array", "items": {"type": "string"}},
                "section_texts": {"type": "array", "items": {"type": "string"}},
            },
            required=["section_titles", "section_descriptions", "section_texts"],
        ),
        return_parameters={
            "type": "object",
            "properties": {
                "section_titles": {"type": "array", "items": {"type": "string"}},
                "section_descriptions": {"type": "array", "items": {"type": "string"}},
                "section_texts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["section_titles", "section_descriptions", "section_texts"],
        },
    )


def build_pass1_user_prompt(raw_text: str, source_path: str) -> str:
    return (
        "Источник: "
        f"{source_path}\n\n"
        "Обработай документ и верни структурированный результат.\n\n"
        "Верни аргументы функции в виде параллельных массивов одинаковой длины.\n"
        "Пример:\n"
        '{"book_authors_csv":["Иванов И.И.; Петров П.П."],"book_titles":["..."],"book_references":[""],"nominations":[""],"rating_overalls":[""],"rating_details_serialized":["критерий: 8; критерий: 7"],"review_texts_clean":["..."]}\n\n'
        "Текст документа:\n"
        f"{raw_text}"
    )


def build_pass2_user_prompt(clean_review_text: str, book_title: str | None = None) -> str:
    title_hint = f"Книга: {book_title}\n\n" if book_title else ""
    return (
        f"{title_hint}"
        "Раздели очищенный текст рецензии на тематические секции.\n\n"
        "Верни аргументы функции в виде параллельных массивов одинаковой длины.\n"
        "Пример:\n"
        '{"section_titles":["..."],"section_descriptions":["..."],"section_texts":["..."]}\n\n'
        "Очищенный текст:\n"
        f"{clean_review_text}"
    )

