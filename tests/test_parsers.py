"""파서: 포맷별 변환 품질과 실패 분류."""

from __future__ import annotations

import pytest

from rag.exceptions import EmptyDocumentError, EncryptedDocumentError, UnsupportedFileError
from rag.parsers import ParserRouter, PDFParser


@pytest.fixture(scope="module")
def router():
    return ParserRouter()


# ------------------------------------------------------------------- Excel
def test_excel_simple_sheet_keeps_leading_zeros(router, samples):
    md = router.parse(samples["xlsx"]).markdown
    # tabulate가 숫자로 바꿔 "1"이 되면 제품코드 검색이 깨집니다.
    assert "| 001 |" in md and "| 005 |" in md


def test_excel_complex_sheet_becomes_narrative_rows(router, samples):
    doc = router.parse(samples["xlsx"])
    assert doc.metadata["sheet_modes"] == {"단가표": "markdown_table", "견적": "narrative"}
    md = doc.markdown
    # 제목 행은 헤더가 아니라 표 앞 문단으로 분리
    assert "2025년 견적서" in md
    # 2단 헤더 평탄화 + 퍼센트 서식 복원
    assert "[행4] 구분: 사무용품, 품목: A4용지, 가격_단가: 100, 가격_합계: 300, 할인율: 15%" in md
    # 세로 병합(A4:A5) 값이 다음 행에도 채워져야 함
    assert "[행5] 구분: 사무용품, 품목: 볼펜" in md
    # 날짜는 00:00:00 없이
    assert "2025-07-15" in md and "00:00:00" not in md


def test_excel_skips_hidden_and_empty_sheets(router, samples):
    doc = router.parse(samples["xlsx"])
    assert doc.metadata["sheets"] == ["단가표", "견적"]
    assert any("숨김" in w for w in doc.warnings)
    assert any("빈시트" in w for w in doc.warnings)


# -------------------------------------------------------------------- DOCX
def test_docx_headings_and_table(router, samples):
    md = router.parse(samples["docx"]).markdown
    assert "# 1장 개요" in md and "## 1.1 적용 범위" in md
    assert "| 항목 | 가격 | 비고 |" in md
    # 셀 안의 파이프는 이스케이프되어야 표가 깨지지 않음
    assert "재고 있음 \\| 긴급" in md


# --------------------------------------------------------------------- PDF
def test_pdf_has_page_markers(router, samples):
    doc = router.parse(samples["pdf"])
    for page in (1, 2, 3):
        assert f"<!-- page: {page} -->" in doc.markdown
    assert doc.metadata["page_count"] == 3


class _FakeDoclingDoc:
    pages = {1: None, 2: None, 3: None}

    def __init__(self, text: str) -> None:
        self._text = text

    def export_to_markdown(self, page_no=None):
        return self._text


def _fake_converter(document=None, exc=None):
    class Converter:
        def convert(self, path):
            if exc:
                raise exc
            return type("R", (), {"status": "ConversionStatus.SUCCESS", "document": document})()

    return Converter()


def _parser_with_docling(converter) -> PDFParser:
    parser = PDFParser()
    parser._docling_available = True
    parser._get_docling_converter = lambda ocr: converter
    return parser


def test_pdf_docling_success(samples):
    body = "## Title\n\n" + "unit price is 100 won for item. " * 20
    doc = _parser_with_docling(_fake_converter(_FakeDoclingDoc(body))).parse(samples["pdf"])
    assert doc.parser_used == "docling"
    assert doc.markdown.count("<!-- page:") == 3


def test_pdf_docling_crash_falls_back(samples):
    doc = _parser_with_docling(_fake_converter(exc=RuntimeError("layout model crashed"))).parse(samples["pdf"])
    assert doc.parser_used == "pymupdf4llm"
    assert doc.metadata["route_reason"] == "docling_fallback"


def test_pdf_docling_silent_loss_falls_back(samples):
    # 예외 없이 본문 대부분을 누락한 "조용한 실패"도 원문 글자 수 비교로 잡아야 함
    doc = _parser_with_docling(_fake_converter(_FakeDoclingDoc("x"))).parse(samples["pdf"])
    assert doc.parser_used == "pymupdf4llm"
    assert any("본문 누락" in w for w in doc.warnings)


# ---------------------------------------------------------------- failures
@pytest.mark.parametrize(
    "key, exc",
    [
        ("locked", EncryptedDocumentError),
        ("empty", EmptyDocumentError),
        ("blank", EmptyDocumentError),  # 텍스트 레이어 없음 + OCR 불가
        ("lockfile", UnsupportedFileError),
        ("txt", UnsupportedFileError),
    ],
)
def test_router_classifies_failures(samples, key, exc):
    router = ParserRouter()
    router._parsers[".pdf"]._docling_available = False
    with pytest.raises(exc):
        router.parse(samples[key])


def test_parse_many_isolates_failures(router, samples):
    docs, failures = router.parse_many([samples["xlsx"], samples["locked"], samples["txt"], samples["docx"]])
    assert [d.source for d in docs] == ["견적_테스트.xlsx", "규정_테스트.docx"]
    assert {f.error_type for f in failures} == {"EncryptedDocumentError", "UnsupportedFileError"}


def test_source_name_is_nfc(router, samples):
    import unicodedata

    doc = router.parse(samples["xlsx"])
    assert doc.source == unicodedata.normalize("NFC", doc.source)
