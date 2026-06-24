import asyncio
import aiohttp
import json
import xml.etree.ElementTree as ET
import urllib.parse
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import apis, directories
from src.connectors.normalize import (
    PaperRecord,
    normalize_pubmed_entry,
    normalize_arxiv_entry,
    normalize_semantic_scholar_entry,
    DatasetSplitter,
    validate_corpus,
    deduplicate_corpus
)


class RateLimiter:
    def __init__(self, calls_per_second: float):
        self.delay = 1.0 / calls_per_second
        self.last_call = 0.0

    async def wait(self) -> None:
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_call
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self.last_call = asyncio.get_event_loop().time()


class PubMedClient:
    def __init__(self):
        self.base_url = apis.pubmed_base_url
        self.rate_limiter = RateLimiter(apis.pubmed_rate_limit)

    async def search_ids(self, query: str, session: aiohttp.ClientSession, max_results: int = 50) -> List[str]:
        await self.rate_limiter.wait()
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "tool": apis.pubmed_tool,
            "email": apis.pubmed_email
        }
        url = f"{self.base_url}/esearch.fcgi"
        async with session.get(url, params=params) as response:
            if response.status != 200:
                return []
            data = await response.json()
            return data.get("esearchresult", {}).get("idlist", [])

    async def fetch_details(self, pmids: List[str], session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        await self.rate_limiter.wait()
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": apis.pubmed_tool,
            "email": apis.pubmed_email
        }
        url = f"{self.base_url}/efetch.fcgi"
        async with session.get(url, params=params) as response:
            if response.status != 200:
                return []
            xml_text = await response.text()
            return self.parse_xml(xml_text)

    def parse_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        root = ET.fromstring(xml_text)
        results = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.find(".//PMID")
            title = article.find(".//ArticleTitle")
            abstract = article.find(".//AbstractText")
            year = article.find(".//PubDate/Year")

            author_list = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                initials = author.find("Initials")
                author_list.append({
                    "LastName": last_name.text if last_name is not None else "",
                    "Initials": initials.text if initials is not None else ""
                })

            results.append({
                "PMID": pmid.text if pmid is not None else "",
                "ArticleTitle": title.text if title is not None else "",
                "AbstractText": abstract.text if abstract is not None else "",
                "PubDate": {"Year": year.text if year is not None else "1970"},
                "AuthorList": author_list
            })
        return results


class ArXivClient:
    def __init__(self):
        self.base_url = apis.arxiv_base_url
        self.rate_limiter = RateLimiter(apis.arxiv_rate_limit)

    async def fetch_papers(self, query: str, session: aiohttp.ClientSession, max_results: int = 50) -> List[
        Dict[str, Any]]:
        await self.rate_limiter.wait()
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results
        }
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        async with session.get(url) as response:
            if response.status != 200:
                return []
            xml_text = await response.text()
            return self.parse_atom(xml_text)

    def parse_atom(self, xml_text: str) -> List[Dict[str, Any]]:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("atom:entry", ns):
            id_val = entry.find("atom:id", ns)
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            published = entry.find("atom:published", ns)

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                authors.append({"name": name.text if name is not None else ""})

            results.append({
                "id": id_val.text if id_val is not None else "",
                "title": title.text if title is not None else "",
                "summary": summary.text if summary is not None else "",
                "published": published.text if published is not None else "1970-01-01T00:00:00Z",
                "authors": authors
            })
        return results


class SemanticScholarClient:
    def __init__(self):
        self.base_url = apis.semantic_scholar_base_url
        self.api_key = apis.semantic_scholar_api_key
        self.rate_limiter = RateLimiter(apis.semantic_scholar_rate_limit)

    async def fetch_papers(self, query: str, session: aiohttp.ClientSession, max_results: int = 50) -> List[
        Dict[str, Any]]:
        await self.rate_limiter.wait()
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        params = {
            "query": query,
            "limit": max_results,
            "fields": "paperId,title,abstract,year,authors,url,externalIds"
        }
        url = f"{self.base_url}/paper/search"
        async with session.get(url, params=params, headers=headers) as response:
            if response.status != 200:
                return []
            data = await response.json()
            return data.get("data", [])


class CorpusBuilder:
    def __init__(self, query: str, num_records_per_source: int = 50):
        self.query = query
        self.num_records = num_records_per_source
        self.pubmed = PubMedClient()
        self.arxiv = ArXivClient()
        self.semantic = SemanticScholarClient()
        self.all_records: List[PaperRecord] = []

    async def gather_data(self) -> None:
        async with aiohttp.ClientSession() as session:
            pubmed_task = self.fetch_from_pubmed(session)
            arxiv_task = self.fetch_from_arxiv(session)
            semantic_task = self.fetch_from_semantic(session)
            await asyncio.gather(pubmed_task, arxiv_task, semantic_task)

    async def fetch_from_pubmed(self, session: aiohttp.ClientSession) -> None:
        logger.info(f"Fetching from PubMed for query: {self.query}")
        pmids = await self.pubmed.search_ids(self.query, session, self.num_records)
        logger.info(f"PubMed found {len(pmids)} PMIDs.")
        raw_data = await self.pubmed.fetch_details(pmids, session)
        for item in raw_data:
            try:
                record = normalize_pubmed_entry(item)
                self.all_records.append(record)
            except Exception as e:
                logger.warning(f"Failed to normalize PubMed record: {e}")

    async def fetch_from_arxiv(self, session: aiohttp.ClientSession) -> None:
        logger.info(f"Fetching from ArXiv for query: {self.query}")
        raw_data = await self.arxiv.fetch_papers(self.query, session, self.num_records)
        logger.info(f"ArXiv returned {len(raw_data)} records.")
        for item in raw_data:
            try:
                record = normalize_arxiv_entry(item)
                self.all_records.append(record)
            except Exception as e:
                logger.warning(f"Failed to normalize ArXiv record: {e}")

    async def fetch_from_semantic(self, session: aiohttp.ClientSession) -> None:
        logger.info(f"Fetching from Semantic Scholar for query: {self.query}")
        raw_data = await self.semantic.fetch_papers(self.query, session, self.num_records)
        logger.info(f"Semantic Scholar returned {len(raw_data)} records.")
        for item in raw_data:
            try:
                record = normalize_semantic_scholar_entry(item)
                self.all_records.append(record)
            except Exception as e:
                logger.warning(f"Failed to normalize Semantic Scholar record: {e}")

    def process_and_save(self) -> None:
        logger.info("Starting processing and saving of records...")
        directories.create_all()
        valid_records = validate_corpus(self.all_records)
        logger.info(f"Validated {len(valid_records)} out of {len(self.all_records)} total fetched records.")
        unique_records = deduplicate_corpus(valid_records)
        logger.info(f"Deduplicated down to {len(unique_records)} unique records.")

        splitter = DatasetSplitter(test_size=0.2)
        splits = splitter.deterministic_split(unique_records)

        train_path = directories.corpus_dir / "train.jsonl"
        test_path = directories.corpus_dir / "test.jsonl"

        self.write_jsonl(splits["train"], train_path)
        self.write_jsonl(splits["test"], test_path)

        logger.info(f"Successfully saved {len(splits['train'])} records to {train_path}")
        logger.info(f"Successfully saved {len(splits['test'])} records to {test_path}")

    def write_jsonl(self, records: List[PaperRecord], filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")


async def main() -> None:
    logger.info("Initializing Corpus Builder...")
    query = "KRAS G12C inhibition resistance mechanism"
    builder = CorpusBuilder(query=query, num_records_per_source=100)
    await builder.gather_data()
    builder.process_and_save()
    logger.info("Data seeding complete.")


if __name__ == "__main__":
    asyncio.run(main())