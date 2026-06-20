"""Shared pytest fixtures for the local-kg test suite."""

import sys
from pathlib import Path

import pytest

# Make the project root importable when pytest is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract import Entity, Relationship  # noqa: E402
from graph import build_graph  # noqa: E402


@pytest.fixture
def sample_entities():
    return [
        Entity("BADM 557", "course", "Business Intelligence with AI", confidence=0.95,
               source_file="syllabus.md"),
        Entity("badm 557", "course", "BI course", confidence=0.6,
               source_file="notes.md"),  # duplicate (normalizes to same key)
        Entity("BADM 576", "course", "Machine Learning II", confidence=0.9,
               source_file="syllabus.md"),
        Entity("Xing", "person", "Faculty member", confidence=0.8,
               source_file="roster.md"),
        Entity("Power BI", "tool", "BI dashboarding tool", confidence=0.85,
               source_file="syllabus.md"),
    ]


@pytest.fixture
def sample_relationships():
    return [
        Relationship("BADM 557", "BADM 576", "prerequisite_of",
                     "BI before ML II", confidence=0.9, source_file="syllabus.md"),
        Relationship("BADM 557", "Xing", "taught_by",
                     "Taught by Xing", confidence=0.8, source_file="roster.md"),
        Relationship("BADM 557", "Power BI", "uses",
                     "Uses Power BI", confidence=0.85, source_file="syllabus.md"),
    ]


@pytest.fixture
def sample_graph(sample_entities, sample_relationships):
    return build_graph(sample_entities, sample_relationships)
