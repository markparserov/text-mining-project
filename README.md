# Текст-майнинг рецензий на книги

Проект для анализа русскоязычных книжных рецензий в 2 шага:

1. **ABSA (Aspect-Based Sentiment Analysis)** через GigaChat: извлечение аспектов и их тональности (`POS`/`NEG`) из текста рецензии.
2. **Тематическое моделирование через LDA-бейзлайн и Fuzzy BERTopic** (по мотивам статьи Nikbakht & Zojaji, 2026): эмбеддинги -> UMAP -> fuzzy C-means -> топ-слова тем через `TI/TIadj`.

Также есть ноутбуки для последующего анализа результатов.

## Состав проекта

- `gigachat_absa_reviews.py` — ABSA на JSONL с рецензиями.
- `fuzzy_bertopic_paper_pipeline.py` — fuzzy BERTopic с мягким распределением по темам.
- `absa_score_correlation.ipynb` — проверка связи оценок рецензий (`score`) с ABSA-метриками.
- `bertopic_lda_analysis.ipynb` — визуализации и расширенный анализ результатов тематического моделирования (fuzzy BERTopic + LDA).
- `requirements.txt` — зависимости.

В репозитории хранится код и ноутбуки. **Данные и тяжёлые артефакты** (входной JSONL с рецензиями, выход ABSA, `topic_modelling_results/` с эмбеддингами и т.п.) создаются **локально** при запуске скриптов.

## Требования

- Python 3.10+ (рекомендуется conda-окружение).
- Доступ к API GigaChat для ABSA-скрипта.

## Установка

```bash
conda create -n text-mining-project python=3.11 -y
conda activate text-mining-project
pip install -r requirements.txt
```

Примечание: для `hdbscan` в `requirements.txt` есть подсказка для установки через conda-forge при возникновении проблем.

## Формат входных данных

Оба скрипта работают с JSONL (один JSON-объект на строку, UTF-8).

Минимально в записи должен быть хотя бы один из полей текста:

- `review_text_clean`
- `review_text_raw`
- `review_text`

Пример строки:

```json
{"year": 2024, "reviewer_id": "R13", "book_title": "Русская ловушка", "score": 9.3, "review_text_clean": "Работа выполнена на достойном уровне..."}
```

Подготовьте свой JSONL с рецензиями (или используйте уже предобработанный файл у себя на диске) и укажите к нему путь в командах ниже.

## 1) Запуск ABSA через GigaChat

Перед запуском настройте переменные окружения для SDK `gigachat` (например, через `.env`).

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