"""청커: 표 헤더 재주입, 캡션 전파, 페이지 추적, 메타데이터 태그, Parent 크기 보정."""

from __future__ import annotations

import re

import pytest

from rag.chunking import ParentChildChunker, TableAwareSplitter, parse_markdown_table
from rag.config import ChunkerConfig
from rag.exceptions import ChunkingError
from rag.schemas import ParsedDocument

HEADER = "| 코드 | 가격 | 비고 |"
SEPARATOR = "|:---|---:|---|"


def _doc(markdown: str, name: str = "t.pdf") -> ParsedDocument:
    return ParsedDocument("d1", name, f"/x/{name}", "pdf", markdown, "test")


@pytest.fixture(scope="module")
def small_chunker():
    return ParentChildChunker(
        ChunkerConfig(parent_max_chars=600, parent_min_chars=200, child_chunk_chars=200, child_overlap_chars=20)
    )


@pytest.fixture(scope="module")
def big_doc_result(small_chunker):
    rows = "\n".join(f"| R{i:02d} | {i * 10} | 비고{i} |" for i in range(60))
    long_text = "가나다라마바사 아자차카타파하. " * 60
    md = f"""<!-- page: 1 -->

# 1장 서론

짧은 도입.

## 1.1 목적

목적 설명 문장입니다.

<!-- page: 2 -->

# 2장 가격

(단위: 원)

{HEADER}
{SEPARATOR}
{rows}

<!-- page: 3 -->

## 2.1 긴 본문

{long_text}

```python
# 이건 헤더가 아님
| 이것도 | 표 아님 |
```

# 3장 부록

끝.
"""
    return small_chunker.split(_doc(md))


def _tables(result):
    return [c for c in result.children if c.metadata["chunk_type"] == "table"]


def test_every_table_fragment_repeats_header(big_doc_result):
    tables = _tables(big_doc_result)
    assert len(tables) > 5
    assert all(f"{HEADER}\n{SEPARATOR}" in c.page_content for c in tables)
    # 부모 단계에서 쪼개진 표 조각도 헤더를 가져야 함
    parents_with_rows = [p for p in big_doc_result.parents if "| R" in p.page_content]
    assert len(parents_with_rows) > 1
    assert all(HEADER in p.page_content for p in parents_with_rows)


def test_no_table_row_lost_or_duplicated(big_doc_result):
    rows = [line for c in _tables(big_doc_result) for line in c.page_content.split("\n") if line.startswith("| R")]
    assert len(rows) == 60 and len(set(rows)) == 60


def test_caption_propagates_with_continued_marker(big_doc_result):
    tables = _tables(big_doc_result)
    captions = [c.page_content.split("\n")[1] for c in tables]
    assert captions[0] == "(단위: 원)"
    assert all(cap == "(단위: 원) (계속)" for cap in captions[1:])


def test_metadata_tag_prefix_on_all_chunks(big_doc_result):
    for chunk in big_doc_result.children + big_doc_result.parents:
        assert chunk.page_content.startswith("[문서명: t.pdf, 섹션: ")


def test_no_heading_only_parents(big_doc_result):
    for parent in big_doc_result.parents:
        body = [line for line in parent.page_content.split("\n")[1:] if line.strip()]
        assert not all(re.match(r"^#+ ", line) for line in body), parent.metadata["parent_id"]


def test_pages_are_tracked(big_doc_result):
    by_section = {c.metadata["section"]: c.metadata.get("pages") for c in big_doc_result.children}
    assert by_section["1장 서론"] == "1"
    assert by_section["2장 가격"] == "2"
    assert by_section["2장 가격 > 2.1 긴 본문"] == "3"  # 페이지 중간 시작 섹션도 페이지를 알아야 함


def test_code_fence_preserved(big_doc_result):
    assert any("# 이건 헤더가 아님" in c.page_content for c in big_doc_result.children)
    assert "이건 헤더가 아님" not in {c.metadata["section"] for c in big_doc_result.children}


def test_small_sections_merge_within_chapter_only(big_doc_result):
    labels = [p.metadata["section"] for p in big_doc_result.parents]
    assert "1장 서론 (1.1 목적 포함)" in labels  # 도입부 + 소절 병합 라벨
    assert all("3장" not in label or label == "3장 부록" for label in labels)  # 다른 장과 섞이지 않음


def test_children_point_to_existing_parents(big_doc_result):
    parent_ids = {p.metadata["parent_id"] for p in big_doc_result.parents}
    assert all(c.metadata["parent_id"] in parent_ids for c in big_doc_result.children)
    child_ids = [c.metadata["child_id"] for c in big_doc_result.children]
    assert len(child_ids) == len(set(child_ids))


def test_metadata_is_chroma_safe(big_doc_result):
    # Chroma는 None / list 값이 있으면 upsert가 실패함
    for chunk in big_doc_result.children + big_doc_result.parents:
        for value in chunk.metadata.values():
            assert isinstance(value, (str, int, float, bool)), (chunk.metadata, value)


def test_document_without_headers(small_chunker):
    result = small_chunker.split(_doc("그냥 문단 하나.\n\n두 번째 문단.", "n.docx"))
    assert [c.page_content for c in result.children] == ["[문서명: n.docx, 섹션: 본문]\n그냥 문단 하나.\n\n두 번째 문단."]
    assert "pages" not in result.children[0].metadata  # 페이지 없는 문서는 키 자체가 없어야 함


def test_heading_only_document_raises(small_chunker):
    with pytest.raises(ChunkingError):
        small_chunker.split(_doc("# 제목만\n\n## 소제목"))


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ParentChildChunker(ChunkerConfig(child_chunk_chars=5000, parent_max_chars=3000))


# ------------------------------------------------------- TableAwareSplitter
def test_table_splitter_keeps_small_table_whole():
    table = parse_markdown_table("| a | b |\n|---|---|\n| 1 | 2 |")
    assert TableAwareSplitter(500).split(table) == ["| a | b |\n|---|---|\n| 1 | 2 |"]


def test_parse_markdown_table_with_caption():
    table = parse_markdown_table("표 1. 단가\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert table.caption == "표 1. 단가" and table.rows == ["| 1 | 2 |"]
    assert parse_markdown_table("그냥 문장") is None
