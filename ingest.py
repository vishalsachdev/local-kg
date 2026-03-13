"""
Document ingestion pipeline.

Sensory memory layer: reads raw files and converts them into clean text chunks
ready for entity extraction. Handles markdown, HTML, and plain text.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import markdown


@dataclass
class Chunk:
    """A piece of text with provenance tracking."""
    text: str
    source_file: str
    section: str = ""
    chunk_index: int = 0

    @property
    def id(self) -> str:
        return f"{self.source_file}::{self.section}::{self.chunk_index}"


def read_markdown(path: Path) -> str:
    """Convert markdown to plain text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    html = markdown.markdown(raw)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def read_html(path: Path) -> str:
    """Extract text from HTML, stripping boilerplate."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    # Remove nav, header, footer, script, style
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def read_text(path: Path) -> str:
    """Read plain text files."""
    return path.read_text(encoding="utf-8", errors="replace")


READERS = {
    ".md": read_markdown,
    ".html": read_html,
    ".htm": read_html,
    ".txt": read_text,
}


def extract_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_title, section_body) pairs."""
    lines = text.split("\n")
    sections = []
    current_title = "introduction"
    current_body = []

    for line in lines:
        # Detect section headers (markdown-style or all-caps lines)
        stripped = line.strip()
        if stripped and (
            stripped.startswith("#")
            or (stripped.isupper() and len(stripped) > 3 and len(stripped) < 100)
        ):
            if current_body:
                sections.append((current_title, "\n".join(current_body)))
            current_title = re.sub(r"^#+\s*", "", stripped).strip().lower()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_title, "\n".join(current_body)))

    return sections


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks sized for local LLM context."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind(".", start, end)
            if last_period > start + max_chars // 2:
                end = last_period + 1
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def ingest_directory(directory: str) -> list[Chunk]:
    """
    Ingest all supported files from a directory into chunks.
    This is the sensory → working memory transition.
    """
    chunks = []
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    supported = set(READERS.keys())
    files = sorted(
        f for f in dir_path.rglob("*")
        if f.suffix.lower() in supported and not f.name.startswith(".")
    )

    for file_path in files:
        reader = READERS[file_path.suffix.lower()]
        text = reader(file_path)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if not text or len(text) < 50:
            continue

        rel_path = str(file_path.relative_to(dir_path))
        sections = extract_sections(text)

        for section_title, section_body in sections:
            section_body = section_body.strip()
            if len(section_body) < 30:
                continue

            text_chunks = chunk_text(section_body)
            for i, chunk_text_content in enumerate(text_chunks):
                chunks.append(Chunk(
                    text=chunk_text_content.strip(),
                    source_file=rel_path,
                    section=section_title,
                    chunk_index=i,
                ))

    return chunks


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    chunks = ingest_directory(target)
    print(f"Ingested {len(chunks)} chunks from {target}")
    for c in chunks[:3]:
        print(f"\n--- {c.source_file} [{c.section}] ---")
        print(c.text[:200])
