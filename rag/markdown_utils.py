"""Parser와 Chunker가 공유하는 Markdown 규약과 유틸리티.

[설계 의도]
- 페이지 마커 형식, 표 구분선 판별 같은 "규약"이 파서와 청커에 각각 하드코딩되면
  한쪽만 바뀌었을 때 페이지 추적이 조용히 깨집니다(에러 없이 인용 페이지만 틀려짐).
  이런 버그는 발견이 매우 늦기 때문에 규약을 한 파일에 모아 단일 원천으로 둡니다.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# 페이지 마커
# ---------------------------------------------------------------------------
# HTML 주석을 쓰는 이유: Markdown 렌더링 시 보이지 않고, 헤더(#)나 표(|)로
# 오인될 일이 없으며, 청커가 정규식으로 확실하게 찾아 제거할 수 있습니다.
PAGE_MARKER_TEMPLATE = "<!-- page: {page} -->"
PAGE_MARKER_RE = re.compile(r"^\s*<!--\s*page:\s*(\d+)\s*-->\s*$")


def page_marker(page: int) -> str:
    return PAGE_MARKER_TEMPLATE.format(page=page)


def match_page_marker(line: str) -> Optional[int]:
    """페이지 마커 줄이면 페이지 번호를, 아니면 None을 반환합니다."""
    m = PAGE_MARKER_RE.match(line)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 텍스트 정규화
# ---------------------------------------------------------------------------
# 탭(\t), 줄바꿈(\n)을 제외한 제어문자. PDF 추출 결과에 NUL(\x00)이나 폼피드(\x0c)가
# 섞여 있으면 임베딩 모델 토크나이저나 Chroma 저장 단계에서 예측하기 어려운 오류가 납니다.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
# Docling이 이미지 자리에 남기는 플레이스홀더. 검색에 아무 정보도 주지 않습니다.
_IMAGE_PLACEHOLDER_RE = re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.MULTILINE)


def normalize_text(text: str) -> str:
    """유니코드/공백 정규화.

    NFC 정규화가 특히 중요합니다. macOS 파일시스템은 한글 파일명을 NFD(자모 분리)로
    저장하므로, 정규화 없이 파일명을 태그에 넣으면 "견적서"가 눈에는 같아 보여도
    다른 코드포인트 열이 되어 BM25 키워드 매칭과 임베딩이 모두 어긋납니다.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace(" ", " ")  # non-breaking space → 일반 공백
    text = _IMAGE_PLACEHOLDER_RE.sub("", text)
    # 줄 끝 공백 제거 (Markdown의 "두 칸 공백 줄바꿈" 규칙이 청킹을 교란하지 않도록)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def strip_page_markers(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if match_page_marker(line) is None)


def has_meaningful_content(markdown: str) -> bool:
    """페이지 마커, 헤더 기호, 표 구분선 등을 제외하고 실제 글자가 남는지 확인합니다."""
    body = strip_page_markers(markdown)
    body = re.sub(r"[#|\-:*_>\s`=]", "", body)
    return len(body) > 0


# ---------------------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------------------
_HEADER_DECORATION_RE = re.compile(r"(\*\*|__|\*|_|`)")


def clean_header_text(text: str) -> str:
    """헤더 텍스트의 강조 기호, 닫는 #, 중복 공백을 제거합니다.

    PyMuPDF4LLM은 굵은 글씨 제목을 `## **제목**`처럼 출력하는데, 이 `**`가 섹션명에
    그대로 남으면 인용 표기가 지저분해지고 BM25 토큰에도 쓰레기가 섞입니다.
    """
    text = _HEADER_DECORATION_RE.sub("", text)
    text = re.sub(r"\s+#+\s*$", "", text)  # "## 제목 ##" 형태의 닫는 해시
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 표
# ---------------------------------------------------------------------------
def is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def is_table_separator(line: str) -> bool:
    """`|---|:---:|` 형태의 구분선인지 판별합니다.

    파서마다 구분선 모양이 다릅니다(Docling: `|---|`, tabulate: `|:-----|------:|`).
    특정 모양을 정규식으로 맞추는 대신 "파이프/대시/콜론/공백 외 문자가 없고
    대시가 하나 이상"이라는 성질로 판별해야 모든 파서 출력에 견고합니다.
    """
    s = line.strip()
    if "|" not in s:
        return False
    if "-" not in s:
        return False
    return re.sub(r"[|\-:\s]", "", s) == ""


def escape_table_cell(value: str) -> str:
    """셀 값 안의 파이프와 줄바꿈을 이스케이프합니다.

    셀 안에 `|`가 그대로 있으면 열 개수가 달라져 표 전체가 깨지고,
    줄바꿈이 있으면 한 행이 여러 줄로 쪼개져 행 단위 분할이 불가능해집니다.
    """
    value = value.replace("|", "\\|")
    return re.sub(r"\s*\n\s*", " ", value).strip()


def build_markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """헤더와 행 목록으로 Markdown 표 문자열을 만듭니다 (DOCX Fallback 등에서 사용)."""
    width = len(header)
    lines: List[str] = [
        "| " + " | ".join(escape_table_cell(h) for h in header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in rows:
        cells = list(row) + [""] * (width - len(row))  # 열 수가 모자란 행은 빈 셀로 채움
        lines.append("| " + " | ".join(escape_table_cell(c) for c in cells[:width]) + " |")
    return "\n".join(lines)
