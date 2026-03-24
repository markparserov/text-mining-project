from __future__ import annotations

from typing import Dict, List, Optional
import re

from pydantic import BaseModel, Field, model_validator


class Section(BaseModel):
    title: str = Field(default="")
    description: str = Field(default="")
    text: str = Field(default="")


class RatingDetails(BaseModel):
    scores: Dict[str, str] = Field(default_factory=dict)


class RatingDetailItem(BaseModel):
    criterion: str = Field(default="")
    value: str = Field(default="")


class Pass1Review(BaseModel):
    book_authors: List[str] = Field(min_length=1)
    book_title: str = Field(min_length=1)
    book_reference: Optional[str] = None
    nomination: Optional[str] = None
    rating_overall: Optional[str] = None
    rating_details: Optional[List[RatingDetailItem]] = None
    review_text_clean: str = Field(min_length=1)


class Pass1Result(BaseModel):
    reviews: List[Pass1Review] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_single_review_payload(cls, data):
        def _is_json_fragment_noise(value: str) -> bool:
            if value is None:
                return True
            text = value.strip()
            if not text:
                return True
            if text in {'"', "}", "]", "],", '"],', '"\n    ],'}:
                return True
            if "\\n    ]," in text or "\n    ]," in text:
                return True
            if text.count("{") + text.count("}") >= 2 and len(text) < 40:
                return True
            if text.count("[") + text.count("]") >= 2 and len(text) < 40:
                return True
            return False

        def _clean_optional_text(value: object) -> Optional[str]:
            if value is None:
                return None
            if not isinstance(value, str):
                value = str(value)
            cleaned = value.strip()
            if _is_json_fragment_noise(cleaned):
                return None
            return cleaned

        def _clean_rating(value: object) -> Optional[str]:
            cleaned = _clean_optional_text(value)
            if not cleaned:
                return None
            if re.fullmatch(r"[0-9]+([.,][0-9]+)?", cleaned):
                return cleaned.replace(",", ".")
            return None

        if isinstance(data, dict):
            flat_keys = {
                "book_authors_csv",
                "book_titles",
                "book_references",
                "nominations",
                "rating_overalls",
                "rating_details_serialized",
                "review_texts_clean",
            }
            if flat_keys.issubset(set(data.keys())):
                authors_arr = data.get("book_authors_csv") or []
                titles_arr = data.get("book_titles") or []
                refs_arr = data.get("book_references") or []
                noms_arr = data.get("nominations") or []
                overalls_arr = data.get("rating_overalls") or []
                details_arr = data.get("rating_details_serialized") or []
                texts_arr = data.get("review_texts_clean") or []
                lengths = [
                    len(arr) if isinstance(arr, list) else 0
                    for arr in [
                        authors_arr,
                        titles_arr,
                        refs_arr,
                        noms_arr,
                        overalls_arr,
                        details_arr,
                        texts_arr,
                    ]
                ]
                n = max(lengths) if lengths else 0
                reviews = []
                for i in range(n):
                    authors_csv = (authors_arr[i] if i < len(authors_arr) and authors_arr[i] is not None else "") or ""
                    if not isinstance(authors_csv, str):
                        authors_csv = str(authors_csv)
                    book_authors = [a.strip() for a in authors_csv.replace("|", ";").split(";") if isinstance(a, str) and a.strip()]
                    details_serialized = (
                        details_arr[i] if i < len(details_arr) and details_arr[i] is not None and isinstance(details_arr[i], str) else ""
                    ) or ""
                    rating_details = []
                    for chunk in details_serialized.split(";"):
                        part = (chunk or "").strip()
                        if not part or ":" not in part:
                            continue
                        criterion, value = part.split(":", 1)
                        criterion = (criterion or "").strip()
                        value = (value or "").strip()
                        if criterion and value:
                            rating_details.append({"criterion": criterion, "value": value})
                    reviews.append(
                        {
                            "book_authors": book_authors,
                            "book_title": (titles_arr[i] if i < len(titles_arr) else None) or "",
                            "book_reference": (refs_arr[i] if i < len(refs_arr) else None) or "",
                            "nomination": (noms_arr[i] if i < len(noms_arr) else None) or "",
                            "rating_overall": (overalls_arr[i] if i < len(overalls_arr) else None) or "",
                            "rating_details": rating_details,
                            "review_text_clean": (texts_arr[i] if i < len(texts_arr) else None) or "",
                        }
                    )
                data = {"reviews": reviews}

        if isinstance(data, list):
            data = {"reviews": data}
        if isinstance(data, dict) and "reviews" not in data:
            if "payload_json" in data and isinstance(data["payload_json"], list):
                data = {"reviews": data["payload_json"]}
            has_review_fields = "book_title" in data or "review_text_clean" in data
            if has_review_fields:
                data = {"reviews": [data]}

        if isinstance(data, dict) and isinstance(data.get("reviews"), list):
            normalized_reviews = []
            for item in data["reviews"]:
                if not isinstance(item, dict):
                    continue
                candidate = dict(item)
                title_raw = candidate.get("book_title", "")
                title_clean = _clean_optional_text(title_raw) or ""
                clean_text = (
                    candidate.get("review_text_clean")
                    or candidate.get("review_text")
                    or candidate.get("text")
                    or ""
                )
                if isinstance(clean_text, str):
                    clean_text = clean_text.strip()
                else:
                    clean_text = ""
                if not clean_text or _is_json_fragment_noise(clean_text):
                    # vLLM sometimes returns metadata but empty review body.
                    # Keep such rows and recover the text from source document later.
                    has_any_metadata = bool(
                        title_clean
                        or _clean_optional_text(candidate.get("book_reference"))
                        or _clean_optional_text(candidate.get("nomination"))
                        or _clean_optional_text(candidate.get("rating_overall"))
                        or candidate.get("rating_details")
                    )
                    if not has_any_metadata:
                        continue
                    clean_text = "__RAW_TEXT__"
                candidate["review_text_clean"] = clean_text

                candidate["book_title"] = title_clean
                authors = candidate.get("book_authors")
                if isinstance(authors, str):
                    parts = [p.strip() for p in authors.replace("|", ";").split(";")]
                    authors = [a for a in parts if a and not _is_json_fragment_noise(a)]
                elif isinstance(authors, list):
                    authors = [
                        a.strip()
                        for a in authors
                        if isinstance(a, str) and a.strip() and not _is_json_fragment_noise(a.strip())
                    ]
                else:
                    authors = []
                candidate["book_authors"] = authors or ["Не указан автор книги"]
                candidate["book_reference"] = _clean_optional_text(candidate.get("book_reference"))
                candidate["nomination"] = _clean_optional_text(candidate.get("nomination"))
                candidate["rating_overall"] = _clean_rating(candidate.get("rating_overall"))
                if not candidate["book_title"]:
                    candidate["book_title"] = "Не указано название книги"
                normalized_reviews.append(candidate)
            data["reviews"] = normalized_reviews
        return data


class Pass2Result(BaseModel):
    sections: List[Section] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_sections_payload(cls, data):
        if isinstance(data, dict):
            if {"section_titles", "section_descriptions", "section_texts"}.issubset(set(data.keys())):
                titles = data.get("section_titles") or []
                descriptions = data.get("section_descriptions") or []
                texts = data.get("section_texts") or []
                n = max(
                    len(titles) if isinstance(titles, list) else 0,
                    len(descriptions) if isinstance(descriptions, list) else 0,
                    len(texts) if isinstance(texts, list) else 0,
                )
                sections = []
                for i in range(n):
                    sections.append(
                        {
                            "title": (titles[i] if i < len(titles) else None) or "",
                            "description": (descriptions[i] if i < len(descriptions) else None) or "",
                            "text": (texts[i] if i < len(texts) else None) or "",
                        }
                    )
                data = {"sections": sections}
        if isinstance(data, dict) and "sections" not in data:
            text_candidate = (
                data.get("message")
                or data.get("disclaimer")
                or data.get("statement")
                or data.get("description")
                or data.get("text")
                or ""
            )
            if isinstance(text_candidate, str) and text_candidate.strip():
                return {
                    "sections": [
                        {
                            "title": "Основной текст",
                            "description": "Автоматически выделенный блок текста",
                            "text": text_candidate.strip(),
                        }
                    ]
                }
        if isinstance(data, dict) and isinstance(data.get("sections"), list) and not data["sections"]:
            return {
                "sections": [
                    {
                        "title": "Основной текст",
                        "description": "Пустой ответ модели, сохранен fallback-блок",
                        "text": "",
                    }
                ]
            }
        return data


class ReviewRecord(BaseModel):
    year: int
    reviewer_id: str
    book_authors: List[str] = Field(min_length=1)
    book_title: str = Field(min_length=1)
    book_reference: Optional[str] = None
    nomination: Optional[str] = None
    review_text_raw: str = Field(min_length=1)
    review_text_clean: str = Field(min_length=1)
    sections: List[Section] = Field(min_length=1)
    rating_overall: Optional[str] = None
    rating_details: Optional[Dict[str, str]] = None
    source_path: str

    @model_validator(mode="after")
    def validate_year(self) -> "ReviewRecord":
        if self.year < 1900 or self.year > 2100:
            raise ValueError("year is out of expected range")
        return self

