"""Tests for the Open Knowledge Format (OKF) export and round-trip import."""

import yaml

from okf_export import export_okf, load_okf_bundle, slugify


def test_slugify():
    assert slugify("BADM 557") == "badm-557"
    assert slugify("Power BI!") == "power-bi"
    assert slugify("  weird__name  ") == "weird-name"
    assert slugify("***") == "concept"  # never empty


def test_export_creates_bundle_structure(sample_graph, tmp_path):
    out = tmp_path / "bundle"
    summary = export_okf(sample_graph, str(out))

    assert (out / "index.md").exists()
    assert summary["entities"] == sample_graph.number_of_nodes()
    assert summary["files_written"] == sample_graph.number_of_nodes()

    # Directory structure encodes the type taxonomy.
    assert (out / "entities" / "course").is_dir()
    assert (out / "entities" / "course" / "index.md").exists()
    assert (out / "entities" / "course" / "badm-557.md").exists()


def test_entity_file_has_valid_frontmatter(sample_graph, tmp_path):
    out = tmp_path / "bundle"
    export_okf(sample_graph, str(out))

    text = (out / "entities" / "course" / "badm-557.md").read_text()
    assert text.startswith("---")
    fm = yaml.safe_load(text.split("---", 2)[1])
    assert fm["type"] == "course"          # only required OKF field
    assert fm["title"] == "BADM 557"
    assert "timestamp" in fm
    # Relationships rendered as markdown links to other concept files.
    assert "## Relationships" in text
    assert "](" in text


def test_relationships_use_relative_links(sample_graph, tmp_path):
    out = tmp_path / "bundle"
    export_okf(sample_graph, str(out))
    text = (out / "entities" / "course" / "badm-557.md").read_text()
    # Cross-type link from a course to a person.
    assert "../person/xing.md" in text


def test_round_trip_preserves_graph(sample_graph, tmp_path):
    out = tmp_path / "bundle"
    export_okf(sample_graph, str(out))
    reloaded = load_okf_bundle(str(out))

    assert reloaded.number_of_nodes() == sample_graph.number_of_nodes()
    assert reloaded.number_of_edges() == sample_graph.number_of_edges()
    # Edges and their relation labels survive the markdown round trip.
    assert reloaded.has_edge("BADM 557", "BADM 576")
    assert reloaded.edges["BADM 557", "BADM 576"]["relation"] == "prerequisite_of"
    assert reloaded.has_edge("BADM 557", "Power BI")


def test_round_trip_does_not_duplicate_edges_from_backlinks(sample_graph, tmp_path):
    """The 'Referenced by' section must not be re-parsed into extra edges."""
    out = tmp_path / "bundle"
    export_okf(sample_graph, str(out))
    reloaded = load_okf_bundle(str(out))
    # Xing is referenced by BADM 557 (taught_by) but has no outgoing edges.
    assert reloaded.out_degree("Xing") == 0
