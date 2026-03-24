#!/usr/bin/env python3
"""Базовая реализация fuzzy BERTopic-пайплайна из статьи (Nikbakht & Zojaji, 2026)

Реализует:
1) эмбеддинги документов через sentence-transformers
2) уменьшение размерности через UMAP
3) кластеризацию через fuzzy C-means
4) извлечение топ-слов через ti/tiadj скоринг
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from fcmeans import FCM
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
from umap import UMAP
import nltk


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


def rank_coefficient(rank: np.ndarray, mode: str) -> np.ndarray:
    rank = np.asarray(rank, dtype=np.float64)
    if mode == "inverse":
        return 1.0 / rank
    if mode == "inverse_square":
        return 1.0 / np.square(rank)
    if mode == "inverse_log10":
        return 1.0 / (np.log10(rank) + 1.0)
    if mode == "inverse_exp_square":
        return np.exp(-np.square(rank - 1.0))
    raise ValueError(f"Unknown rank coefficient mode: {mode}")


@dataclass
class PaperPipelineOutputs:
    doc_rows: List[Dict[str, Any]]
    topic_rows: List[Dict[str, Any]]
    vocab_rows: List[Dict[str, Any]]
    doc_embeddings: np.ndarray
    reduced_embeddings: np.ndarray
    term_embeddings: np.ndarray
    terms: List[str]


def _safe_num_topics(requested_topics: int, n_docs: int) -> int:
    # оставляем минимум 2 кластера и избегаем неустойчивости n_clusters == n_docs
    if n_docs <= 3:
        return 2
    return max(2, min(requested_topics, n_docs - 1))


def _safe_umap_neighbors(requested_neighbors: int, n_docs: int) -> int:
    if n_docs <= 2:
        return 1
    return max(2, min(requested_neighbors, n_docs - 1))


def _extract_review_text(record: Dict[str, Any]) -> str:
    for key in ("review_text_clean", "review_text_raw", "review_text"):
        value = record.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def _build_russian_stop_words() -> List[str]:
    stop_words: set[str] = set()

    # 1) русские стоп-слова из nltk
    try:
        from nltk.corpus import stopwords as nltk_stopwords

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        stop_words.update(w.lower() for w in nltk_stopwords.words("russian"))
    except Exception:
        pass

    # 2) стоп-слова из пакета stop_words
    try:
        stop_words.update(w.lower() for w in get_stop_words("ru"))
    except Exception:
        pass

    # 3) доменные стоп-слова для проекта
    manual_stop_words = {
        # односимвольные токены русского алфавита
        "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м",
        "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ",
        "ы", "ь", "э", "ю", "я",
        # книга
        "книга", "книги", "книге", "книгу", "книгой", "книгою", "книг", "книгам", "книгами", "книгах",
        # работа
        "работа", "работы", "работе", "работу", "работой", "работою", "работ", "работам", "работами", "работах",
        # автор
        "автор", "автора", "автору", "автором", "авторе", "авторы", "авторов", "авторам", "авторами", "авторах",
        # перевод
        "перевод", "перевода", "переводу", "переводом", "переводе", "переводы", "переводов", "переводам", "переводами", "переводах",
        # исследование
        "исследование", "исследования", "исследованию", "исследованием", "исследовании",
        "исследований", "исследованиям", "исследованиями", "исследованиях",
        # целый
        "целый", "целая", "целое", "целые", "целого", "целой", "целому", "целым", "целых", "целую", "целыми",
        # представлять
        "представлять", "представляет", "представляют", "представлял", "представляла", "представляли",
        "представлен", "представлена", "представлено", "представлены", "представить", "представляя",
        # социолог / социология
        "социолог", "социолога", "социологу", "социологом", "социологе", "социологи", "социологов",
        "социология", "социологии", "социологию", "социологией", "социологию", "социологический", "социологическая", "социологические",
        # скорее
        "скорее",
        # лист
        "лист", "листа", "листу", "листом", "листе", "листы", "листов", "листам", "листами", "листах",
        # поэтому / поскольку / такой
        "поэтому", "поскольку", "такой", "такая", "такое", "такие", "такого", "такой", "такому", "таким", "таких", "такую", "такими",
        # материал
        "материал", "материала", "материалу", "материалом", "материале", "материалы", "материалов", "материалам", "материалами", "материалах",
        # являться
        "являться", "является", "являются", "являлся", "являлась", "являлись", "являясь", "явиться",
        # например / пример
        "например", "пример", "примера", "примеру", "примером", "примере", "примеры", "примеров", "примерам", "примерами", "примерах",
        # достаточно / принципиально / разный
        "достаточно", "принципиально",
        "разный", "разная", "разное", "разные", "разного", "разной", "разному", "разным", "разных", "разную", "разными",
        # дополнительные формы
        "целом", "др", "пр", "безусловно", "вполне", "читателя",
        "ред", "лонг", "пер", "англ", "вообще", "стр", "изд",
        # контекст
        "контекст", "контекста", "контексту", "контекстом", "контексте", "контексты", "контекстов", "контекстам", "контекстами", "контекстах",
        # должен
        "должен", "должна", "должно", "должны", "должного", "должной", "должному", "должным", "должных", "должную",
        "долженствующий", "должна быть", "должно быть",
        # область
        "область", "области", "областью", "областей", "областям", "областями", "областях",
        # премия
        "премия", "премии", "премию", "премией", "премиею", "премий", "премиям", "премиями", "премиях", "премией",
        # некоторый
        "некоторый", "некоторая", "некоторое", "некоторые", "некоторого", "некоторой", "некоторому", "некоторым", "некоторых", "некоторую", "некоторыми",
        # шорт в том числе шорт-лист
        "шорт", "шорта", "шорту", "шортом", "шорте", "шорты", "шортов", "шортам", "шортами", "шортах",
        "шортлист", "шортлиста", "шортлисту", "шортлистом", "шортлисте", "шортлисты", "шортлистов", "шортлистам", "шортлистами", "шортлистах",
        "шорт-лист", "шорт-листа", "шорт-листу", "шорт-листом", "шорт-листе", "шорт-листы", "шорт-листов", "шорт-листам", "шорт-листами", "шорт-листах",
        # сборник
        "сборник", "сборника", "сборнику", "сборником", "сборнике", "сборники", "сборников", "сборникам", "сборниками", "сборниках",
        # текст
        "текст", "текста", "тексту", "текстом", "тексте", "тексты", "текстов", "текстам", "текстами", "текстах",
        # статья
        "статья", "статьи", "статью", "статье", "статьей", "статьею", "статей", "статьям", "статьями", "статьях",
        # образ
        "образ", "образа", "образу", "образом", "образе", "образы", "образов", "образам", "образами", "образах",
    }
    stop_words.update(manual_stop_words)

    return sorted(w.strip() for w in stop_words if w and w.strip())


def run_paper_pipeline(
    records: List[Dict[str, Any]],
    embedding_model: str,
    num_topics: int,
    top_words: int,
    rank_coeff_mode: str,
    random_state: int,
    umap_neighbors: int,
    umap_components: int,
    doc_embedding_batch_size: int,
    term_embedding_batch_size: int,
    min_df: int,
    max_vocab_size: int,
    absent_similarity: float,
    precomputed_doc_embeddings: np.ndarray | None = None,
    precomputed_reduced_embeddings: np.ndarray | None = None,
) -> PaperPipelineOutputs:
    docs = [_extract_review_text(r) for r in records]
    if len(docs) < 2:
        raise ValueError("At least 2 documents are required.")
    if not any(docs):
        raise ValueError("All review_text_clean/review_text_raw/review_text fields are empty.")

    russian_stop_words = _build_russian_stop_words()

    embedder = SentenceTransformer(embedding_model)
    if precomputed_doc_embeddings is not None:
        doc_embeddings = np.asarray(precomputed_doc_embeddings, dtype=np.float32)
        if doc_embeddings.ndim != 2 or doc_embeddings.shape[0] != len(docs):
            raise ValueError(
                "Invalid precomputed_doc_embeddings shape: "
                f"expected (n_docs, dim) with n_docs={len(docs)}, got {doc_embeddings.shape}"
            )
        print(f"Using precomputed document embeddings: {doc_embeddings.shape}")
    else:
        doc_embeddings = embedder.encode(
            docs,
            show_progress_bar=True,
            batch_size=doc_embedding_batch_size,
            normalize_embeddings=True,
        )

    n_docs = len(docs)
    n_topics = _safe_num_topics(num_topics, n_docs)
    neighbors = _safe_umap_neighbors(umap_neighbors, n_docs)
    if n_docs <= 2:
        components = 1
    else:
        components = max(2, min(umap_components, doc_embeddings.shape[1], n_docs - 1))

    if precomputed_reduced_embeddings is not None:
        reduced = np.asarray(precomputed_reduced_embeddings, dtype=np.float32)
        if reduced.ndim != 2 or reduced.shape[0] != n_docs or reduced.shape[1] != components:
            raise ValueError(
                "Invalid precomputed_reduced_embeddings shape: "
                f"expected ({n_docs}, {components}), got {reduced.shape}"
            )
        print(f"Using precomputed reduced embeddings: {reduced.shape}")
    else:
        reduced = UMAP(
            n_neighbors=neighbors,
            n_components=components,
            min_dist=0.0,
            metric="cosine",
            init="random",
            random_state=random_state,
        ).fit_transform(doc_embeddings)

    fcm = FCM(
        n_clusters=n_topics,
        m=2.0,
        max_iter=300,
        random_state=random_state,
    )
    fcm.fit(reduced)
    membership = np.asarray(fcm.soft_predict(reduced), dtype=np.float64)
    membership = np.nan_to_num(membership, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = membership.sum(axis=1, keepdims=True)
    zero_rows = np.isclose(row_sums, 0.0).reshape(-1)
    if np.any(zero_rows):
        membership[zero_rows] = 1.0 / membership.shape[1]
        row_sums = membership.sum(axis=1, keepdims=True)
    membership = membership / row_sums
    labels = np.argmax(membership, axis=1)

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=russian_stop_words if russian_stop_words else None,
        ngram_range=(1, 2),
        # исключаем чистые числа и числоподобные токены, оставляем только токены, где есть хотя бы одна буква
        token_pattern=r"(?u)\b(?=\w*[A-Za-zА-Яа-яЁё])\w+\b",
        min_df=max(1, min_df),
        max_features=max_vocab_size if max_vocab_size > 0 else None,
    )
    term_doc = vectorizer.fit_transform(docs)  # матрица документов и термов [n_docs, n_terms]
    terms: np.ndarray = vectorizer.get_feature_names_out()
    if terms.size == 0:
        raise ValueError("Vocabulary is empty after vectorization.")

    n_terms = len(terms)
    ti = np.zeros((n_terms, n_topics), dtype=np.float64)
    term_embeddings = np.zeros((n_terms, doc_embeddings.shape[1]), dtype=np.float32)

    # считаем ti батчами, чтобы не держать в памяти полную плотную матрицу terms x docs
    for start in range(0, n_terms, term_embedding_batch_size):
        end = min(start + term_embedding_batch_size, n_terms)
        batch_terms = terms[start:end].tolist()
        term_emb = embedder.encode(
            batch_terms,
            show_progress_bar=False,
            batch_size=term_embedding_batch_size,
            normalize_embeddings=True,
        )
        term_embeddings[start:end] = np.asarray(term_emb, dtype=np.float32)
        sim = cosine_similarity(term_emb, doc_embeddings)  # матрица близостей [batch_terms, n_docs]

        presence = (term_doc[:, start:end].toarray().T > 0)
        sim = np.where(presence, sim, absent_similarity)
        ti[start:end, :] = sim @ membership

    # ранжируем каждый терм по темам через ti и применяем tiadj = ti * coeff(rank)
    order = np.argsort(-ti, axis=1)  # сортировка по убыванию ti для каждого терма
    rank = np.empty_like(order, dtype=np.int64)
    rank[np.arange(n_terms)[:, None], order] = np.arange(1, n_topics + 1)
    coeff = rank_coefficient(rank, rank_coeff_mode)
    ti_adj = ti * coeff

    topic_rows: List[Dict[str, Any]] = []
    for topic_id in range(n_topics):
        idx = np.argsort(-ti_adj[:, topic_id])[:top_words]
        words = terms[idx].tolist()
        weights = ti_adj[idx, topic_id].tolist()
        topic_rows.append(
            {
                "topic_id": int(topic_id),
                "top_words": words,
                "top_word_weights": weights,
                "topic_size_soft": float(np.sum(membership[:, topic_id])),
            }
        )

    doc_rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        doc_rows.append(
            {
                **rec,
                "dominant_topic": int(labels[i]),
                "topic_membership": membership[i].tolist(),
            }
        )

    vocab_rows: List[Dict[str, Any]] = []
    for term_idx, term in enumerate(terms.tolist()):
        vocab_rows.append(
            {
                "term": term,
                "ti": ti[term_idx].tolist(),
                "ti_adj": ti_adj[term_idx].tolist(),
                "best_topic_by_ti": int(np.argmax(ti[term_idx])),
                "best_topic_by_ti_adj": int(np.argmax(ti_adj[term_idx])),
            }
        )

    return PaperPipelineOutputs(
        doc_rows=doc_rows,
        topic_rows=topic_rows,
        vocab_rows=vocab_rows,
        doc_embeddings=np.asarray(doc_embeddings, dtype=np.float32),
        reduced_embeddings=np.asarray(reduced, dtype=np.float32),
        term_embeddings=term_embeddings,
        terms=terms.tolist(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-style fuzzy topic pipeline (embedding + UMAP + FCM + TI/TIadj)."
    )
    parser.add_argument("--input-jsonl", required=True, help="Absolute path to input JSONL.")
    parser.add_argument("--output-dir", required=True, help="Absolute path to output directory.")
    parser.add_argument(
        "--embedding-model",
        default="deepvk/USER-bge-m3",
        help="Sentence-Transformers model name.",
    )
    parser.add_argument("--num-topics", type=int, default=8, help="Requested number of topics.")
    parser.add_argument("--top-words", type=int, default=12, help="Top words per topic.")
    parser.add_argument(
        "--rank-coeff",
        default="inverse",
        choices=["inverse", "inverse_square", "inverse_log10", "inverse_exp_square"],
        help="Rank coefficient mode for TIadj.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap-components", type=int, default=5, help="UMAP n_components.")
    parser.add_argument(
        "--doc-embedding-batch-size",
        type=int,
        default=2,
        help="Batch size for document embedding.",
    )
    parser.add_argument(
        "--term-embedding-batch-size",
        type=int,
        default=2,
        help="Batch size for term embedding during TI computation.",
    )
    parser.add_argument("--min-df", type=int, default=1, help="Min DF for vocabulary.")
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=20000,
        help="Max vocabulary size (0 means unlimited).",
    )
    parser.add_argument(
        "--absent-similarity",
        type=float,
        default=-2.0,
        help="Similarity value for term-document pairs where term is absent.",
    )
    parser.add_argument(
        "--write-vocab-scores",
        action="store_true",
        help="Write per-term TI/TIadj debug file (may be large).",
    )
    parser.add_argument(
        "--no-reuse-embeddings",
        action="store_true",
        help=(
            "Disable auto-reuse of cached embeddings from output-dir "
            "(paper_doc_embeddings.npy and paper_doc_embeddings_reduced.npy). "
            "By default, both caches are reused when available and shape matches."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)

    precomputed_doc_embeddings: np.ndarray | None = None
    precomputed_reduced_embeddings: np.ndarray | None = None
    cached_doc_emb_path = output_dir / "paper_doc_embeddings.npy"
    cached_reduced_emb_path = output_dir / "paper_doc_embeddings_reduced.npy"
    if not args.no_reuse_embeddings and cached_doc_emb_path.exists():
        try:
            cached = np.load(cached_doc_emb_path)
            cached = np.asarray(cached, dtype=np.float32)
            if cached.ndim == 2 and cached.shape[0] == len(records):
                precomputed_doc_embeddings = cached
                print(f"Loaded cached doc embeddings from {cached_doc_emb_path}: {cached.shape}")
            else:
                print(
                    "Cached doc embeddings shape mismatch, recomputing. "
                    f"Got {cached.shape}, expected ({len(records)}, dim)."
                )
        except Exception as exc:
            print(f"Failed to load cached doc embeddings ({cached_doc_emb_path}): {exc}")

    if not args.no_reuse_embeddings and cached_reduced_emb_path.exists():
        try:
            cached_red = np.load(cached_reduced_emb_path)
            cached_red = np.asarray(cached_red, dtype=np.float32)
            if cached_red.ndim == 2 and cached_red.shape[0] == len(records):
                precomputed_reduced_embeddings = cached_red
                print(f"Loaded cached reduced embeddings from {cached_reduced_emb_path}: {cached_red.shape}")
            else:
                print(
                    "Cached reduced embeddings shape mismatch, recomputing. "
                    f"Got {cached_red.shape}, expected ({len(records)}, components)."
                )
        except Exception as exc:
            print(f"Failed to load cached reduced embeddings ({cached_reduced_emb_path}): {exc}")

    outputs = run_paper_pipeline(
        records=records,
        embedding_model=args.embedding_model,
        num_topics=args.num_topics,
        top_words=args.top_words,
        rank_coeff_mode=args.rank_coeff,
        random_state=args.seed,
        umap_neighbors=args.umap_neighbors,
        umap_components=args.umap_components,
        doc_embedding_batch_size=max(1, args.doc_embedding_batch_size),
        term_embedding_batch_size=max(1, args.term_embedding_batch_size),
        min_df=max(1, args.min_df),
        max_vocab_size=max(0, args.max_vocab_size),
        absent_similarity=args.absent_similarity,
        precomputed_doc_embeddings=precomputed_doc_embeddings,
        precomputed_reduced_embeddings=precomputed_reduced_embeddings,
    )

    write_jsonl(output_dir / "paper_doc_topics.jsonl", outputs.doc_rows)
    write_jsonl(output_dir / "paper_topic_info.jsonl", outputs.topic_rows)
    np.save(output_dir / "paper_doc_embeddings.npy", outputs.doc_embeddings)
    np.save(output_dir / "paper_doc_embeddings_reduced.npy", outputs.reduced_embeddings)
    np.savez_compressed(
        output_dir / "paper_term_embeddings.npz",
        terms=np.asarray(outputs.terms, dtype=object),
        embeddings=outputs.term_embeddings,
    )
    if args.write_vocab_scores:
        write_jsonl(output_dir / "paper_vocab_scores.jsonl", outputs.vocab_rows)


if __name__ == "__main__":
    main()