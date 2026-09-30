"""Parser와 Chunker가 공유하는 Markdown 규약과 유틸리티.

[설계 의도]
- 페이지 마커 형식, 표 구분선 판별 같은 "규약"이 파서와 청커에 각각 하드코딩되면
  한쪽만 바뀌었을 때 페이지 추적이 조용히 깨집니다(에러 없이 인용 페이지만 틀려짐).
  이런 버그는 발견이 매우 늦기 때문에 규약을 한 파일에 모아 단일 원천으로 둡니다.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# PDF 헤더 수준 복원
# ---------------------------------------------------------------------------
# PDF 파서(PyMuPDF4LLM, Docling)는 글자 크기로 제목을 찾기 때문에 "2부", "13장", "핵심 교훈"을
# 모두 같은 수준(##)으로 뽑는 경우가 많습니다. 계층이 평평하면
#   (1) "실전 체크리스트"가 몇 장 소속인지 사라져 인용이 모호해지고,
#   (2) 청커의 "같은 장 안에서만 병합" 규칙이 동작하지 않아 Parent가 잘게 쪼개집니다.
# 제목의 번호 패턴(N장, 부록, 1., 1.1)으로 수준을 다시 매깁니다.
_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BOLD_SEGMENT_RE = re.compile(r"\*\*(.+?)\*\*")
_PART_RE = re.compile(r"^\d+\s*부[.\s:]")
_CHAPTER_RE = re.compile(r"^(제\s*)?\d+\s*장[.\s:]|^chapter\s+\d+\b|^부록(\s|$|[A-Za-z~])", re.IGNORECASE)
_SUBNUMBER_RE = re.compile(r"^\d+(\.\d+)*[.)]\s")
# 제목이 아니라 강조된 문장/도입구인 경우: "왜냐하면:", "그러나 현실은 냉혹했다."
_SENTENCE_LIKE_RE = re.compile(r"(:|다\.|요\.|니다\.)$")
# 이 문자로 끝나는 조각은 줄바꿈으로 잘린 제목의 앞부분 → 다음 조각과 합침 ("…AGENTS.md," + "CLAUDE.md…")
_CONTINUATION_ENDINGS = (",", "·", "&", "—", "-", "및")


def _split_heading_segments(text: str) -> List[str]:
    """`**1장. 제목** **시작하며: 부제**` 처럼 한 줄에 합쳐진 여러 제목을 조각으로 나눕니다."""
    segments = _BOLD_SEGMENT_RE.findall(text)
    # 굵은 조각 외의 글자가 남아 있으면 한 제목 안에 강조가 섞인 것이므로 나누지 않습니다.
    if not segments or _BOLD_SEGMENT_RE.sub("", text).strip():
        return [clean_header_text(text)]
    merged: List[str] = []
    for seg in (clean_header_text(s) for s in segments):
        if not seg:
            continue
        if merged and merged[-1].endswith(_CONTINUATION_ENDINGS):
            merged[-1] = f"{merged[-1]} {seg}"
        else:
            merged.append(seg)
    return merged


def restore_heading_levels(markdown: str, min_chapters: int = 2, flat_ratio: float = 0.9) -> Tuple[str, Dict[str, int]]:
    """평평해진 PDF 헤더를 번호 패턴으로 계층화합니다.

    규칙
        N장 / 제N장 / Chapter N / 부록        → #   (장: 최상위, 번호가 문서 전체에서 유일)
        장 안의 번호 없는 제목                  → ##
        "1. …" / "1.1 …" 번호 소제목            → ###
        첫 장 이전의 제목(표지, 저자의 말, 목차) → #
        N부 제목                                → 굵은 본문으로 강등
        "…다." / "…:"로 끝나는 강조 문장        → 굵은 본문으로 강등

    [왜 "부"를 계층에서 빼는가]
    추출된 PDF에서 "N부" 제목은 신뢰할 수 없습니다. 실제 사례에서 "6부" 배너가 5부 소속인
    19장과 같은 줄로 추출되었고, 같은 부 제목이 두 번 나오기도 했습니다. 잘못된 부를 경로에 넣으면
    인용이 틀리므로, 문서 전체에서 유일한 장 번호를 최상위로 씁니다(텍스트는 본문으로 남겨 검색 가능).

    [안전장치]
    이미 계층이 살아 있는 문서를 망가뜨리지 않도록, 헤더의 flat_ratio 이상이 한 수준에 몰려 있고
    장 패턴이 min_chapters개 이상일 때만 적용합니다.

    Returns:
        (변환된 markdown, 통계 dict). 적용하지 않았으면 원문과 {"applied": 0}.
    """
    lines = markdown.split("\n")
    heading_idx = [i for i, line in enumerate(lines) if _HEADING_LINE_RE.match(line)]
    if not heading_idx:
        return markdown, {"applied": 0}

    level_counts: Dict[int, int] = {}
    chapter_count = 0
    for i in heading_idx:
        m = _HEADING_LINE_RE.match(lines[i])
        level_counts[len(m.group(1))] = level_counts.get(len(m.group(1)), 0) + 1
        chapter_count += sum(1 for s in _split_heading_segments(m.group(2)) if _CHAPTER_RE.match(s))
    is_flat = max(level_counts.values()) / len(heading_idx) >= flat_ratio
    if not is_flat or chapter_count < min_chapters:
        return markdown, {"applied": 0}

    stats = {"applied": 1, "chapters": 0, "sections": 0, "subsections": 0, "demoted": 0, "parts_demoted": 0}
    out: List[str] = []
    in_chapter = False
    in_fence = False
    for line in lines:
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_LINE_RE.match(line)
        if not m:
            out.append(line)
            continue

        for seg in _split_heading_segments(m.group(2)):
            if _PART_RE.match(seg):
                out.extend([f"**{seg}**", ""])
                stats["parts_demoted"] += 1
            elif _CHAPTER_RE.match(seg):
                out.extend([f"# {seg}", ""])
                in_chapter = True
                stats["chapters"] += 1
            elif _SENTENCE_LIKE_RE.search(seg):
                out.extend([f"**{seg}**", ""])
                stats["demoted"] += 1
            elif not in_chapter:
                out.extend([f"# {seg}", ""])  # 표지·서문·목차 등 첫 장 이전
                stats["chapters"] += 1
            elif _SUBNUMBER_RE.match(seg):
                out.extend([f"### {seg}", ""])
                stats["subsections"] += 1
            else:
                out.extend([f"## {seg}", ""])
                stats["sections"] += 1
    return "\n".join(out), stats


# ---------------------------------------------------------------------------
# PDF 인라인 코드 오탐 제거
# ---------------------------------------------------------------------------
# 일부 PDF는 본문 속 영문·숫자·기호를 고정폭 글꼴로 조판합니다. PyMuPDF4LLM은 고정폭 글꼴을
# 코드로 보고 `10`, `→`, `(`처럼 백틱으로 감싸는데, 실제 사례에서 인라인 코드 3,395개 중 69%가
# 글자 없이 숫자·기호만 감싼 오탐이었습니다. 이런 백틱은 LLM 컨텍스트에 잡음을 더하고
# "10 스텝마다" 같은 정확한 문구 대조를 방해하므로 벗겨냅니다.
_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")


def _has_letter(text: str) -> bool:
    return any(unicodedata.category(ch).startswith("L") for ch in text)


def strip_symbolic_inline_code(markdown: str) -> Tuple[str, int]:
    """글자(한글·영문 등)가 없는 인라인 코드의 백틱을 제거합니다. 코드 블록(```) 안은 건드리지 않습니다.

    예외(백틱 유지):
      - 줄 맨 앞의 `#…` : 벗기면 Markdown 헤더가 되어 청킹이 깨짐
      - `|`가 들어간 것  : 벗기면 표 행으로 오인될 수 있음
    글자가 들어간 `ralph.set_goal(` 같은 조각은 실제 코드일 수 있어 보수적으로 남깁니다.

    Returns:
        (변환된 markdown, 제거한 인라인 코드 수)
    """
    removed = 0
    out: List[str] = []
    in_fence = False

    for line in markdown.split("\n"):
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or "`" not in line:
            out.append(line)
            continue

        def repl(match: "re.Match[str]") -> str:
            nonlocal removed
            content = match.group(1)
            at_line_start = not line[: match.start()].strip()
            if _has_letter(content) or "|" in content or (at_line_start and content.lstrip().startswith("#")):
                return match.group(0)
            removed += 1
            return content

        out.append(_INLINE_CODE_RE.sub(repl, line))
    return "\n".join(out), removed
