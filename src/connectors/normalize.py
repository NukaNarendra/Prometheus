import re
import hashlib
import random
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class PaperRecord(BaseModel):
    paper_id: str
    source: str
    title: str
    abstract: str
    authors: List[Author]
    publication_date: str
    url: str
    doi: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    is_open_access: bool = False

    @field_validator("title")
    @classmethod
    def clean_title(cls, v: str) -> str:
        v = re.sub(r"<[^>]+>", "", v)
        v = re.sub(r"\s+", " ", v)
        return v.strip()

    @field_validator("abstract")
    @classmethod
    def clean_abstract(cls, v: str) -> str:
        v = re.sub(r"<[^>]+>", "", v)
        v = re.sub(r"\s+", " ", v)
        v = re.sub(r"\$[^$]+\$", "MATH_EXPR", v)
        return v.strip()

    @field_validator("publication_date")
    @classmethod
    def standardize_date(cls, v: str) -> str:
        try:
            parsed_date = datetime.strptime(v, "%Y-%m-%d")
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            try:
                parsed_date = datetime.strptime(v, "%Y-%m")
                return parsed_date.strftime("%Y-%m-01")
            except ValueError:
                try:
                    parsed_date = datetime.strptime(v, "%Y")
                    return parsed_date.strftime("%Y-01-01")
                except ValueError:
                    return "1970-01-01"

    @field_validator("doi")
    @classmethod
    def clean_doi(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v = v.lower().strip()
        if v.startswith("http://dx.doi.org/"):
            return v.replace("http://dx.doi.org/", "")
        if v.startswith("https://doi.org/"):
            return v.replace("https://doi.org/", "")
        if v.startswith("doi:"):
            return v.replace("doi:", "").strip()
        return v


class DatasetSplitter:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        random.seed(self.random_state)

    def generate_hash_id(self, record: PaperRecord) -> str:
        unique_string = f"{record.title}{record.publication_date}{record.source}"
        return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

    def deterministic_split(self, records: List[PaperRecord]) -> Dict[str, List[PaperRecord]]:
        train_set = []
        test_set = []
        for record in records:
            record.paper_id = self.generate_hash_id(record)
            if random.random() > self.test_size:
                train_set.append(record)
            else:
                test_set.append(record)
        return {"train": train_set, "test": test_set}


class TextSanitizer:
    @staticmethod
    def remove_html_entities(text: str) -> str:
        entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'"}
        for entity, replacement in entities.items():
            text = text.replace(entity, replacement)
        return text

    @staticmethod
    def extract_keywords_from_text(text: str) -> List[str]:
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        return list(set([word for word in words if word not in stop_words and len(word) > 4]))[:10]


def normalize_pubmed_entry(xml_dict: Dict[str, Any]) -> PaperRecord:
    title = xml_dict.get("ArticleTitle", "Unknown Title")
    abstract = xml_dict.get("AbstractText", "No abstract available")
    date_str = xml_dict.get("PubDate", {}).get("Year", "1970")
    authors_raw = xml_dict.get("AuthorList", [])

    parsed_authors = []
    for a in authors_raw:
        last = a.get("LastName", "")
        initials = a.get("Initials", "")
        name = f"{last} {initials}".strip()
        parsed_authors.append(Author(name=name))

    return PaperRecord(
        paper_id=xml_dict.get("PMID", ""),
        source="PubMed",
        title=TextSanitizer.remove_html_entities(title),
        abstract=TextSanitizer.remove_html_entities(abstract),
        authors=parsed_authors,
        publication_date=date_str,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{xml_dict.get('PMID', '')}/"
    )


def normalize_arxiv_entry(entry_dict: Dict[str, Any]) -> PaperRecord:
    return PaperRecord(
        paper_id=entry_dict.get("id", "").split("/")[-1],
        source="ArXiv",
        title=entry_dict.get("title", "").replace("\n", " "),
        abstract=entry_dict.get("summary", "").replace("\n", " "),
        authors=[Author(name=a.get("name", "")) for a in entry_dict.get("authors", [])],
        publication_date=entry_dict.get("published", "1970-01-01").split("T")[0],
        url=entry_dict.get("id", "")
    )


def normalize_semantic_scholar_entry(json_dict: Dict[str, Any]) -> PaperRecord:
    return PaperRecord(
        paper_id=json_dict.get("paperId", ""),
        source="SemanticScholar",
        title=json_dict.get("title", ""),
        abstract=json_dict.get("abstract", "") or "No abstract available",
        authors=[Author(name=a.get("name", "")) for a in json_dict.get("authors", [])],
        publication_date=str(json_dict.get("year", "1970")) + "-01-01",
        url=json_dict.get("url", ""),
        doi=json_dict.get("externalIds", {}).get("DOI", "")
    )


def validate_corpus(records: List[PaperRecord]) -> List[PaperRecord]:
    valid_records = []
    for record in records:
        if len(record.abstract) > 50 and len(record.title) > 5:
            valid_records.append(record)
    return valid_records


def deduplicate_corpus(records: List[PaperRecord]) -> List[PaperRecord]:
    seen_titles = set()
    unique_records = []
    for record in records:
        normalized_title = record.title.lower().replace(" ", "")
        if normalized_title not in seen_titles:
            seen_titles.add(normalized_title)
            unique_records.append(record)
    return unique_records