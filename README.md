# Текст-майнинг рецензий на книги

Проект для анализа русскоязычных книжных рецензий в 3 шага:

1. **Препроцессинг документов** (`.rtf/.doc/.docx`) в структурированный JSONL через `preprocessing_pipeline`.
2. **ABSA (Aspect-Based Sentiment Analysis)** через GigaChat: извлечение аспектов и их тональности (`POS`/`NEG`) из текста рецензии.
3. **Тематическое моделирование через LDA-бейзлайн и Fuzzy BERTopic** (по мотивам статьи Nikbakht & Zojaji, 2026): эмбеддинги -> UMAP -> fuzzy C-means -> топ-слова тем через `TI/TIadj`.

Также есть ноутбуки для последующего анализа результатов.

## Состав проекта

- `preprocessing_pipeline/` — CLI-пайплайн извлечения структурированных полей из `.rtf/.doc/.docx` (каталог `тексты/`) в JSONL.
- `gigachat_absa_reviews.py` — ABSA на JSONL с рецензиями.
- `fuzzy_bertopic_paper_pipeline.py` — fuzzy BERTopic с мягким распределением по темам.
- `absa_score_correlation.ipynb` — проверка связи оценок рецензий (`score`) с ABSA-метриками.
- `bertopic_lda_analysis.ipynb` — визуализации и расширенный анализ результатов тематического моделирования (fuzzy BERTopic + LDA).
- `requirements.txt` — зависимости для ABSA и тематического моделирования.
- `environment.yml` — conda-окружение для запуска всего репозитория, включая препроцессинг.

В репозитории хранится код и ноутбуки. **Данные и артефакты обработки** (входные документы, выходные JSONL/эмбеддинги/чекпоинты и т.п.) создаются **локально** при запуске скриптов.

## Требования

- Python 3.10+ (рекомендуется conda-окружение).
- Доступ к API GigaChat для ABSA.
- Доступ к API GigaChat для препроцессинга ИЛИ локально развёрнутая OpenAI-совместимая LLM (здесь – развёрнутая при помощи vLLM Qwen3.5)
- LibreOffice (`soffice`) для fallback-извлечения текста из `.doc/.rtf`.

## Установка

Рекомендуемый вариант:

```bash
conda env create -f environment.yml
conda activate textmining
```

Альтернатива:

```bash
conda create -n text-mining-project python=3.11 -y
conda activate text-mining-project
pip install -r requirements.txt
```

## 0) Запуск препроцессинга документов

Базовый запуск (с параметрами-плейсхолдерами):

```bash
python -m preprocessing_pipeline.cli process \
  --input-dir <INPUT_DIR> \
  --output <OUTPUT_JSONL> \
  --checkpoint-dir <CHECKPOINT_DIR> \
  --max-workers <N_WORKERS>
```

Схема работы пайплайна:

`SOURCE_DOCUMENTS -> text extraction -> Pass 1 (review extraction) -> Pass 2 (section segmentation) -> normalization/validation -> JSONL records`

Детализация этапов:

1. **Обнаружение источников**: рекурсивный поиск документов поддерживаемых форматов и устранение технических дублей.
2. **Извлечение текста**: конвертация в текстовый слой, нормализация кодировок и очистка служебного шума.
3. **Pass 1**: структурирование документа в список рецензий с метаданными (книга, номинация, оценочные поля, очищенный текст).
4. **Pass 2**: сегментация каждой рецензии на содержательные секции (`title`, `description`, `text`).
5. **Валидация**: приведение результата к единой схеме и контроль обязательных полей.
6. **Сохранение и возобновление**: запись результата в JSONL и ведение checkpoint-файла для устойчивых перезапусков.

### Модель и serving для препроцессинга

Для vLLM-режима в проекте используется модель [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).  
Для serving рекомендуется [vllm-project/vllm](https://github.com/vllm-project/vllm) (OpenAI-compatible API).

Схема запуска через vLLM-only:

```bash
python -m preprocessing_pipeline.cli process \
  --input-dir <INPUT_DIR> \
  --output <OUTPUT_JSONL> \
  --checkpoint-dir <CHECKPOINT_DIR> \
  --use-vllm-only \
  --vllm-base-url <VLLM_BASE_URL> \
  --vllm-model Qwen/Qwen3.5-35B-A3B \
  --vllm-api-key <VLLM_API_KEY>
```

Минимальные переменные окружения (если используете `.env`):

```bash
GIGACHAT_AUTH_TOKEN=...
GIGACHAT_SCOPE=GIGACHAT_API_PERS
VLLM_BASE_URL=<VLLM_BASE_URL>
VLLM_MODEL=Qwen/Qwen3.5-35B-A3B
VLLM_API_KEY=<VLLM_API_KEY>
```

## Формат входных данных

ABSA и тематическое моделирование работают с JSONL (один JSON-объект на строку, UTF-8).

Минимально в записи должен быть хотя бы один из полей текста:

- `review_text_clean`
- `review_text_raw`
- `review_text`

Абстрактная схема записи:

```json
{
  "year": "<YEAR>",
  "reviewer_id": "<REVIEWER_ID>",
  "book_title": "<BOOK_TITLE>",
  "score": "<OPTIONAL_SCORE>",
  "review_text_clean": "<TEXT>"
}
```

Подготовьте свой JSONL с рецензиями (или используйте уже предобработанный файл у себя на диске) и укажите к нему путь в командах ниже.

## 1) Запуск ABSA через GigaChat

Перед запуском настройте переменные окружения для SDK `gigachat` (например, через `.env`).

Схема запуска:

```bash
python gigachat_absa_reviews.py \
  --input-jsonl <INPUT_JSONL> \
  --output-jsonl <OUTPUT_JSONL> \
  --model <GIGACHAT_MODEL> \
  --temperature <TEMPERATURE> \
  --max-retries <MAX_RETRIES> \
  --timeout-sec <TIMEOUT_SEC> \
  --sleep-between-requests-ms <DELAY_MS> \
  --save-every <SAVE_EVERY_N>
```

Что добавляется в каждую запись:

- `absa_items` — список аспектно-сентиментных кортежей:
  - `target`
  - `polarity` (`POS`/`NEG`)
  - `expressions` (список фраз-оснований)
- `absa_error` — ошибка обработки (если была)
- `absa_model` — имя модели
- `absa_prompt_version` — версия промпта

Опция `--debug-save-raw` сохраняет сырой ответ модели в `raw_payload_json`.

Итоговый JSONL с `absa_items` сохраняется в путь из `--output-jsonl`. Те же результаты можно анализировать в `absa_score_correlation.ipynb` (в ноутбуке укажите путь к своему файлу ABSA).

## 2) Запуск fuzzy BERTopic + анализ результатов тематического моделирования

Схема запуска:

```bash
python fuzzy_bertopic_paper_pipeline.py \
  --input-jsonl <INPUT_JSONL> \
  --output-dir <OUTPUT_DIR> \
  --embedding-model <EMBEDDING_MODEL> \
  --num-topics <NUM_TOPICS> \
  --top-words <TOP_WORDS> \
  --rank-coeff <RANK_COEFF> \
  --seed <SEED> \
  --umap-neighbors <N_NEIGHBORS> \
  --umap-components <N_COMPONENTS> \
  --doc-embedding-batch-size <DOC_BATCH> \
  --term-embedding-batch-size <TERM_BATCH> \
  --min-df <MIN_DF> \
  --max-vocab-size <MAX_VOCAB> \
  --absent-similarity <ABSENT_SIMILARITY>
```

Основные выходные файлы в `--output-dir`:

- `paper_doc_topics.jsonl` — документы с:
  - `dominant_topic`
  - `topic_membership` (мягкое распределение вероятностей по темам)
- `paper_topic_info.jsonl` — топ-слова и веса по каждой теме.
- `paper_doc_embeddings.npy` — эмбеддинги документов.
- `paper_doc_embeddings_reduced.npy` — UMAP-проекции документов.
- `paper_term_embeddings.npz` — эмбеддинги термов.
- `paper_vocab_scores.jsonl` — опционально при `--write-vocab-scores`.

Кэширование:

- По умолчанию скрипт переиспользует `paper_doc_embeddings.npy` и `paper_doc_embeddings_reduced.npy`, если размерности совпадают.
- Для отключения используйте `--no-reuse-embeddings`.

В каталоге `--output-dir` появятся `paper_doc_topics.jsonl`, `paper_topic_info.jsonl`, эмбеддинги и при необходимости другие файлы; метрики когерентности и таблицы из ноутбука — по путям, которые вы зададите при анализе. Визуализации и расчёты — в `bertopic_lda_analysis.ipynb` (пути к `paper_*` артефактам задайте в первых ячейках).

## Ноутбуки

- `absa_score_correlation.ipynb`
  - считает метрики `pos_share`, `neg_share`.
  - анализирует корреляцию с итоговым `score`.
- `bertopic_lda_analysis.ipynb`
  - анализирует результаты `paper_*` артефактов,
  - строит визуализации и считает метрики когерентности,
  - содержит блоки для LDA-сравнения.

Перед запуском ноутбуков проверьте пути к входным файлам в первых ячейках.
