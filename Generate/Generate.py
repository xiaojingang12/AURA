import argparse
import json
import logging
import os
import random
import time
from contextlib import suppress
from pathlib import Path
from typing import Iterable

import requests
from openai import OpenAI

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze


OPENREVIEW_API_URL = "https://api.openreview.net/notes"
LOGGER = logging.getLogger(__name__)


def fetch_papers(year: int = 2023) -> list[dict]:
    params = {
        "invitation": f"ICLR.cc/{year}/Conference/-/Blind_Submission",
        "details": "replyCount",
    }
    response = requests.get(OPENREVIEW_API_URL, params=params, timeout=60)
    if response.status_code != 200:
        LOGGER.warning("OpenReview request failed with status code %s", response.status_code)
        return []

    papers = response.json()
    results = []
    for paper in papers.get("notes", []):
        paper_id = paper.get("id")
        content = paper.get("content", {})
        results.append(
            {
                "title": content.get("title", "Untitled"),
                "authors": content.get("authors", []),
                "pdf_link": f"https://openreview.net/pdf?id={paper_id}",
                "id": paper_id,
            }
        )
    return results


def fetch_reviews(forum_id: str, max_retries: int = 5, base_delay: float = 1.0) -> list[dict]:
    params = {
        "forum": forum_id,
        "trash": "true",
        "details": "replyCount,writable,revisions,original,overwriting,invitation,tags",
        "limit": 1000,
        "offset": 0,
    }

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(OPENREVIEW_API_URL, params=params, timeout=60)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                LOGGER.info("Rate limited. Retrying in %.2f seconds (%s/%s).", wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                LOGGER.warning("OpenReview review request failed with status code %s", response.status_code)
                return []

            reviews = []
            for note in response.json().get("notes", []):
                content = note.get("content", {})
                review = {
                    "summary": str(content.get("summary_of_the_paper", "")).strip(),
                    "strength_and_weaknesses": str(content.get("strength_and_weaknesses", "")).strip(),
                }
                if any(review.values()):
                    reviews.append(review)
            return reviews
        except Exception as exc:
            LOGGER.warning("OpenReview request failed: %s", exc)
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(wait_time)

    return []


def parse_pdf_to_markdown(pdf_link: str, work_dir: Path) -> str:
    work_dir.mkdir(parents=True, exist_ok=True)
    image_dir = work_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = work_dir / "paper.pdf"

    response = requests.get(pdf_link, timeout=120)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)

    image_writer = FileBasedDataWriter(str(image_dir))
    md_writer = FileBasedDataWriter(str(work_dir))
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(str(pdf_path))

    dataset = PymuDocDataset(pdf_bytes)
    if dataset.classify() == SupportedPdfParseMethod.OCR:
        infer_result = dataset.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = dataset.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    md_filename = "paper.md"
    pipe_result.dump_md(md_writer, md_filename, str(image_dir))
    md_path = work_dir / md_filename
    paper_content = md_path.read_text(encoding="utf-8")

    with suppress(Exception):
        pdf_path.unlink()
    with suppress(Exception):
        md_path.unlink()
    return paper_content


class TopicExtractor:
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o"):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model

    def call_llm(self, prompt: str, max_retries: int = 5) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                LOGGER.warning("LLM call failed (%s/%s): %s", attempt + 1, max_retries, exc)
                time.sleep(2)
        raise RuntimeError(f"Exceeded maximum retries ({max_retries})")

    def extract_topic(self, reviews_data: Iterable[dict]) -> list[str]:
        combined_text = "\n\n".join(
            f"{review.get('summary', '')} {review.get('strength_and_weaknesses', '')}".strip()
            for review in reviews_data
        )
        prompt = f"""
Analyze the following academic paper reviews and extract 10-15 technical terms or key phrases that best represent the paper's core contributions and methodological focus.

Output requirements:
- Return only a comma-separated list.
- Use technical terminology.
- Prefer compound terms over single words.
- Do not include explanations, markdown, or numbering.

Reviews:
{combined_text}
"""
        LOGGER.info("Prompt length: %s characters", len(prompt))
        raw_response = self.call_llm(prompt)
        cleaned_response = raw_response.translate(str.maketrans("", "", "\"'"))
        topics = [topic.strip() for topic in cleaned_response.split(",") if topic.strip()]
        LOGGER.info("Extracted %s topics", len(topics))
        return topics


def append_jsonl(record: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper topic records from OpenReview metadata and reviews.")
    parser.add_argument("--year", type=int, default=2023, help="ICLR conference year.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index within fetched paper list.")
    parser.add_argument("--max-papers", type=int, default=None, help="Maximum number of papers to process.")
    parser.add_argument("--output-path", type=Path, default=Path("openreview_paper_topics.jsonl"), help="Output JSONL file.")
    parser.add_argument("--work-dir", type=Path, default=Path("output/openreview_pdf"), help="Temporary directory for PDF parsing.")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", ""), help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default=os.getenv("AURA_GENERATE_MODEL", "gpt-4o"), help="Topic extraction model.")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF download and content extraction.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Use --api-key or set OPENAI_API_KEY.")

    extractor = TopicExtractor(base_url=args.api_base, api_key=args.api_key, model=args.model)
    papers = fetch_papers(args.year)
    selected_papers = papers[args.start_index :]
    if args.max_papers is not None:
        selected_papers = selected_papers[: args.max_papers]

    for paper in selected_papers:
        LOGGER.info("Processing paper: %s by %s", paper["title"], ", ".join(paper["authors"]))
        reviews = fetch_reviews(paper["id"])
        if not reviews:
            LOGGER.info("No reviews found for paper %s. Skipping.", paper["title"])
            continue

        topics = extractor.extract_topic(reviews)
        paper_content = "" if args.skip_pdf else parse_pdf_to_markdown(paper["pdf_link"], args.work_dir / str(paper["id"]))
        append_jsonl(
            {
                "title": paper["title"],
                "authors": paper["authors"],
                "link": paper["pdf_link"],
                "reviews": reviews,
                "content": paper_content,
                "question": f"Please summarize the paper {paper['title']}",
                "question_type": "summarization",
                "topic": topics,
                "answer": "The paper discusses the following topics: %s" % ", ".join(topics),
            },
            args.output_path,
        )


if __name__ == "__main__":
    main()
