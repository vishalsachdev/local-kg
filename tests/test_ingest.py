"""Tests for the ingestion / chunking layer (no LLM required)."""

from ingest import (
    Chunk,
    chunk_text,
    extract_sections,
    ingest_directory,
    read_html,
    read_markdown,
)


def test_chunk_id_is_stable():
    c = Chunk(text="hi", source_file="a.md", section="intro", chunk_index=2)
    assert c.id == "a.md::intro::2"


def test_chunk_text_short_returns_single():
    assert chunk_text("short text", max_chars=2000) == ["short text"]


def test_chunk_text_splits_with_overlap():
    text = "x" * 5000
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    # Reassembled length accounts for overlap, so total chars exceed original.
    assert sum(len(c) for c in chunks) >= len(text)


def test_chunk_text_prefers_sentence_boundary():
    text = "A" * 900 + ". " + "B" * 900
    chunks = chunk_text(text, max_chars=1000, overlap=0)
    # First chunk should end at the period rather than mid-"B".
    assert chunks[0].rstrip().endswith(".")


def test_extract_sections_splits_on_headers():
    text = "# Intro\nbody one\n# Methods\nbody two"
    sections = extract_sections(text)
    titles = [t for t, _ in sections]
    assert "intro" in titles
    assert "methods" in titles


def test_read_markdown_strips_formatting(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nSome **bold** text.")
    out = read_markdown(f)
    assert "Title" in out
    assert "**" not in out


def test_read_html_strips_boilerplate(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><nav>menu</nav><body><p>real content</p>"
        "<footer>copyright</footer></body></html>"
    )
    out = read_html(f)
    assert "real content" in out
    assert "menu" not in out
    assert "copyright" not in out


def test_ingest_directory_end_to_end(tmp_path):
    (tmp_path / "a.md").write_text(
        "# Section One\n" + "This is a meaningful paragraph about analytics. " * 5
    )
    (tmp_path / ".hidden.md").write_text("# Hidden\nshould be skipped " * 5)
    chunks = ingest_directory(str(tmp_path), max_chars=2000, overlap=200)
    assert chunks, "expected at least one chunk"
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source_file == "a.md" for c in chunks)  # hidden file skipped


def test_ingest_directory_missing_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        ingest_directory(str(tmp_path / "nope"))
