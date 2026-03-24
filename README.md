# Текст-майнинг рецензий на книги

Репозиторий объединяет три рабочих контура:

1. `preprocessing_pipeline` — извлечение и структурирование данных из `.rtf/.doc/.docx` рецензий в единый JSONL.
2. `gigachat_absa_reviews.py` — ABSA (Aspect-Based Sentiment Analysis) по JSONL.
3. `fuzzy_bertopic_paper_pipeline.py` + ноутбуки — тематическое моделирование и аналитика.

## Структура репозитория

- `preprocessing_pipeline/` — двухпроходный LLM-пайплайн препроцессинга документов рецензий.
- `gigachat_absa_reviews.py` — ABSA для JSONL набора рецензий.
- `fuzzy_bertopic_paper_pipeline.py` — Fuzzy BERTopic / paper-style pipeline.
- `absa_score_correlation.ipynb` — анализ связи ABSA-метрик и оценки рецензии.
- `bertopic_lda_analysis.ipynb` — расширенная аналитика тем и когерентности.
- `requirements.txt` — зависимости ABSA + topic modelling части.
- `environment.yml` — conda-окружение для запуска всего репозитория, включая `preprocessing_pipeline`.

## Требования

- Python 3.11 (рекомендуется conda).
- Доступ к API GigaChat для сценариев с GigaChat.
- Опционально локальный OpenAI-совместимый vLLM endpoint для `preprocessing_pipeline`.
- LibreOffice (`soffice`) для fallback-извлечения текста из `.doc/.rtf`.

## Установка окружения

Рекомендуемый вариант:

```bash
conda env create -f environment.yml
conda activate textmining
```

Если окружение уже существует:

```bash
conda activate textmining
conda env update -f environment.yml --prune
```

## Быстрый старт по этапам

### 1) Препроцессинг документов (`preprocessing_pipeline`)

Пайплайн запускается как модуль:

```bash
python -m preprocessing_pipeline.cli process \
  --input-dir тексты \
  --output output.jsonl \
  --checkpoint-dir .checkpoints \
  --max-workers 4
```

Поддерживаемые входные форматы: `.docx`, `.doc`, `.rtf`.

### Что делает `preprocessing_pipeline`

1. Рекурсивно ищет документы в `--input-dir`.
2. Отфильтровывает служебные файлы и дедуплицирует дубли по stem в папке с приоритетом `docx > doc > rtf`.
3. Извлекает текст:
   - `.docx` через `python-docx`;
   - `.rtf` через `striprtf` с очисткой шума и fallback через LibreOffice;
   - `.doc` через LibreOffice `soffice --headless --convert-to txt` и fallback через OLE-потоки.
4. Выполняет 2 LLM-прохода:
   - **Pass 1**: извлечение рецензий и метаданных;
   - **Pass 2**: сегментация рецензии на тематические секции.
5. Нормализует ответы в единую схему, валидирует поля и пишет выход в JSONL.
6. Ведет чекпоинт `processed_sources.txt` для продолжения прерванного запуска.

### Ключевые опции `preprocessing_pipeline`

- `--env-file` — путь к `.env`.
- `--model` — модель GigaChat.
- `--use-vllm-only` — запуск только через vLLM без GigaChat токена.
- `--enable-vllm-fallback` — fallback на vLLM при ошибках GigaChat.
- `--vllm-base-url`, `--vllm-model`, `--vllm-api-key` — параметры OpenAI-совместимого endpoint.
- `--max-workers` — параллелизм (1..64).
- `--limit` — обработать только первые N файлов.

### Переменные окружения для `preprocessing_pipeline`

Минимум для GigaChat-режима:

```bash
GIGACHAT_AUTH_TOKEN=...
GIGACHAT_SCOPE=GIGACHAT_API_PERS
```

Для vLLM (опционально):

```bash
VLLM_BASE_URL=http://127.0.0.1:8004/v1
VLLM_MODEL=qwen3.5
VLLM_API_KEY=EMPTY
```

### Формат выходной записи `preprocessing_pipeline`

Каждая строка выходного JSONL содержит структуру рецензии с полями:

- `year`, `reviewer_id`
- `book_authors`, `book_title`, `book_reference`, `nomination`
- `review_text_raw`, `review_text_clean`
- `sections` (список блоков `title/description/text`)
- `rating_overall`, `rating_details`
- `source_path`

### Логи и артефакты `preprocessing_pipeline`

- Основной результат: файл из `--output` (обычно `output.jsonl`).
- Чекпоинты: каталог из `--checkpoint-dir`.
- Логи: стандартный stdout/stderr; для долгих запусков можно сохранять в файл (`... > run.log 2>&1`).

### 2) ABSA через GigaChat

ABSA-скрипт ожидает JSONL, где в записи есть хотя бы одно из полей:

- `review_text_clean`
- `review_text_raw`
- `review_text`

Пример запуска:

```bash
python gigachat_absa_reviews.py \
  --input-jsonl /absolute/path/reviews.jsonl \
  --output-jsonl /absolute/path/absa_output.jsonl \
  --model GigaChat-Pro \
  --temperature 0.0 \
  --max-retries 10 \
  --timeout-sec 60 \
  --sleep-between-requests-ms 100 \
  --save-every 1
```

Скрипт дописывает `absa_items`, `absa_error`, `absa_model`, `absa_prompt_version`.

### 3) Тематическое моделирование (Fuzzy BERTopic + LDA baseline)

Пример запуска:

```bash
python fuzzy_bertopic_paper_pipeline.py \
  --input-jsonl /absolute/path/reviews.jsonl \
  --output-dir /absolute/path/topic_modelling_output \
  --embedding-model deepvk/USER-bge-m3 \
  --num-topics 10 \
  --top-words 10 \
  --rank-coeff inverse \
  --seed 42 \
  --umap-neighbors 15 \
  --umap-components 5 \
  --doc-embedding-batch-size 2 \
  --term-embedding-batch-size 2 \
  --min-df 2 \
  --max-vocab-size 20000 \
  --absent-similarity -2.0
```

Типичные артефакты в `--output-dir`:

- `paper_doc_topics.jsonl`
- `paper_topic_info.jsonl`
- `paper_doc_embeddings.npy`
- `paper_doc_embeddings_reduced.npy`
- `paper_term_embeddings.npz`

## Ноутбуки

- `absa_score_correlation.ipynb` — метрики `pos_share`/`neg_share` и корреляция с `score`.
- `bertopic_lda_analysis.ipynb` — визуализации тем, когерентность, сравнение с LDA.

Перед запуском ноутбуков обновите пути к вашим локальным данным в первых ячейках.