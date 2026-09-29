"""Parent-Child(Small-to-Big) 청커 (전략 B).

처리 순서
    1. 페이지 마커 전파       : 각 헤더 바로 뒤에 "현재 페이지" 마커를 복사해 섹션이 자기 시작 페이지를 알게 함
    2. 헤더 기반 1차 분할     : MarkdownHeaderTextSplitter (#, ##, ###) → 섹션(SectionUnit)
    3. 블록 파싱              : 섹션 본문을 문단/표/제목 블록으로 나누고 블록마다 페이지 범위 기록
    4. Parent 크기 보정       : 큰 섹션은 블록 경계로 분할, 작은 섹션은 같은 장(최상위 헤더) 안에서 병합
    5. Child 분할             : 블록을 예산 안에서 묶고, 긴 문단은 재귀 분할, 표는 TableAwareSplitter
    6. 메타데이터 태그 주입   : 모든 Parent/Child 본문 맨 앞에 "[문서명: …, 섹션: …]"

[왜 블록 단위로 직접 묶는가]
LangChain의 RecursiveCharacterTextSplitter에 섹션 전체를 넣으면 (1) 표를 행 중간에서 자르고
(2) 어느 조각이 몇 페이지에서 왔는지 추적할 수 없습니다. 그래서 문단·표 경계를 먼저 확정한
블록을 greedy하게 묶고, "한 블록이 예산보다 긴 경우"에만 재귀 분할기를 씁니다.
블록 경계 자체가 의미 경계이므로, 블록 사이에는 overlap을 두지 않고 긴 문단을 자를 때만
child_overlap_chars만큼 겹치게 합니다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.chunking.table_splitter import TableAwareSplitter, parse_markdown_table
from rag.config import ChunkerConfig
from rag.exceptions import ChunkingError
from rag.markdown_utils import (
    clean_header_text,
    is_table_row,
    is_table_separator,
    match_page_marker,
    page_marker,
)
from rag.schemas import ChunkingResult, ParsedDocument

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_FENCE_RE = re.compile(r"^(```|~~~)")

# 한국어 문장 경계까지 고려한 재귀 분할 구분자 (앞쪽일수록 우선)
_TEXT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "。", ", ", " ", ""]


# ---------------------------------------------------------------------------
# 내부 자료구조
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """섹션 본문의 최소 의미 단위. kind: 'text' | 'table' | 'heading'"""

    kind: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class SectionUnit:
    """헤더 하나가 관할하는 섹션 (또는 큰 섹션을 나눈 일부)."""

    path: Tuple[str, ...]  # ("2장 가격", "2.1 단가")
    headers: Dict[str, str]  # {"h1": "2장 가격", "h2": "2.1 단가"}
    blocks: List[Block]
    part: Optional[Tuple[int, int]] = None  # (현재 조각 번호, 전체 조각 수)

    @property
    def chars(self) -> int:
        # 블록 사이 구분자("\n\n")까지 포함해야 실제 Parent 본문 길이와 일치합니다.
        return sum(len(b.text) for b in self.blocks) + 2 * max(len(self.blocks) - 1, 0)

    @property
    def top_key(self) -> str:
        """병합 가능 여부를 판단하는 '장' 키. 서로 다른 장의 섹션은 절대 병합하지 않습니다."""
        return self.path[0] if self.path else ""

    @property
    def has_body(self) -> bool:
        return any(b.kind != "heading" for b in self.blocks)


@dataclass
class ChildPiece:
    text: str
    kind: str  # "text" | "table"
    page_start: Optional[int]
    page_end: Optional[int]


@dataclass
class _ParentGroup:
    units: List[SectionUnit] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(u.chars for u in self.units) + 2 * max(len(self.units) - 1, 0)


def _page_range(blocks: List[Block]) -> Tuple[Optional[int], Optional[int]]:
    starts = [b.page_start for b in blocks if b.page_start is not None]
    ends = [b.page_end for b in blocks if b.page_end is not None]
    return (min(starts) if starts else None, max(ends) if ends else None)


# ---------------------------------------------------------------------------
# 청커
# ---------------------------------------------------------------------------
class ParentChildChunker:
    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._validate_config()
        self._header_levels = {sep: len(sep) for sep, _ in self.config.headers_to_split_on}
        self._header_names = [name for _, name in self.config.headers_to_split_on]

    def _validate_config(self) -> None:
        c = self.config
        if c.child_chunk_chars >= c.parent_max_chars:
            # Child가 Parent보다 크면 Small-to-Big이 성립하지 않습니다 (설정 실수를 조기에 차단).
            raise ValueError("child_chunk_chars는 parent_max_chars보다 작아야 합니다")
        if c.parent_min_chars > c.parent_max_chars:
            raise ValueError("parent_min_chars는 parent_max_chars 이하여야 합니다")
        if c.child_overlap_chars >= c.child_chunk_chars:
            raise ValueError("child_overlap_chars는 child_chunk_chars보다 작아야 합니다")

    # ------------------------------------------------------------------ public
    def split(self, doc: ParsedDocument) -> ChunkingResult:
        """ParsedDocument → (Parent 목록, Child 목록).

        Raises:
            ChunkingError: 결과 청크가 하나도 없는 경우 (인덱싱하면 '검색되지 않는 문서'가 됨)
        """
        warnings: List[str] = []
        try:
            markdown = self._carry_page_markers(doc.markdown)
            sections = self._split_by_headers(markdown)
            sections = [s for s in sections if s.has_body]  # 제목만 있고 본문 없는 섹션 제거
            if not sections:
                raise ChunkingError("본문이 있는 섹션이 없습니다", source=doc.source_path)

            if all(not s.path for s in sections):
                warnings.append("헤더가 없는 문서 — 문서 전체를 하나의 섹션으로 처리합니다")

            sections = self._split_oversized_sections(sections)
            groups = self._merge_small_sections(sections)
            parents, children = self._build_documents(doc, groups)
        except ChunkingError:
            raise
        except Exception as exc:  # noqa: BLE001
            # 청킹 버그가 파이프라인 전체를 멈추지 않도록 파일 단위 도메인 예외로 감쌉니다.
            raise ChunkingError(f"청킹 중 오류: {type(exc).__name__}: {exc}", source=doc.source_path) from exc

        if not children:
            raise ChunkingError("생성된 Child 청크가 없습니다", source=doc.source_path)

        for w in warnings:
            logger.warning("[%s] %s", doc.source, w)
        logger.info("청킹 완료: %s → Parent %d개, Child %d개", doc.source, len(parents), len(children))
        return ChunkingResult(parents=parents, children=children, warnings=warnings)

    # ------------------------------------------------ 1. 페이지 마커 전파
    def _carry_page_markers(self, markdown: str) -> str:
        """분할 대상 헤더 바로 뒤에 '현재 페이지' 마커를 복사합니다.

        페이지 마커는 페이지 경계에만 있으므로, 12쪽 중간에서 시작하는 섹션은 본문에 마커가
        없어 자기 페이지를 모릅니다. 헤더로 자르기 전에 마커를 한 번 더 심어 두면 모든 섹션이
        "나는 12쪽에서 시작한다"는 정보를 갖게 됩니다.
        """
        out: List[str] = []
        current_page: Optional[int] = None
        in_fence = False
        for line in markdown.split("\n"):
            out.append(line)
            stripped = line.strip()
            if _FENCE_RE.match(stripped):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            page = match_page_marker(stripped)
            if page is not None:
                current_page = page
                continue
            m = _HEADING_RE.match(stripped)
            if m and m.group(1) in self._header_levels and current_page is not None:
                out.append("")
                out.append(page_marker(current_page))
                out.append("")
        return "\n".join(out)

    # ------------------------------------------------ 2. 헤더 기반 1차 분할
    def _split_by_headers(self, markdown: str) -> List[SectionUnit]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=list(self.config.headers_to_split_on),
            # 헤더 줄을 본문에 남겨야 Parent를 LLM에 줄 때 문서 구조(제목)가 보입니다.
            strip_headers=False,
        )
        sections: List[SectionUnit] = []
        for part in splitter.split_text(markdown):
            # [주의] MarkdownHeaderTextSplitter는 빈 줄(문단 경계)을 "  \n"으로 합쳐 버립니다.
            # 모든 줄을 strip()한 뒤 합치므로 "  \n"은 원래 빈 줄이었던 곳에서만 생깁니다.
            # 이를 다시 "\n\n"으로 되돌려야 아래 블록 파서가 문단 경계를 인식합니다.
            content = part.page_content.replace("  \n", "\n\n")
            headers = {
                name: clean_header_text(str(part.metadata[name]))
                for name in self._header_names
                if part.metadata.get(name)
            }
            path = tuple(v for v in headers.values() if v)
            sections.append(SectionUnit(path=path, headers=headers, blocks=self._parse_blocks(content)))
        return sections

    # ------------------------------------------------ 3. 블록 파싱
    def _parse_blocks(self, content: str) -> List[Block]:
        """섹션 본문을 heading / table / text 블록으로 나누고, 블록마다 페이지를 기록합니다."""
        blocks: List[Block] = []
        buf: List[str] = []
        buf_kind = "text"
        buf_page: Optional[int] = None
        current_page: Optional[int] = None
        in_fence = False

        def flush() -> None:
            nonlocal buf, buf_kind, buf_page
            text = "\n".join(buf).strip()
            if text:
                blocks.append(Block(buf_kind, text, buf_page, current_page))
            buf, buf_kind, buf_page = [], "text", None

        def start(kind: str) -> None:
            nonlocal buf_kind, buf_page
            buf_kind, buf_page = kind, current_page

        lines = content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()

            # 코드 블록 내부는 그대로 보존 (표/헤더로 오인 금지)
            if _FENCE_RE.match(stripped):
                if not in_fence:
                    flush()
                    start("text")
                buf.append(line)
                in_fence = not in_fence
                if not in_fence:
                    flush()
                continue
            if in_fence:
                buf.append(line)
                continue

            page = match_page_marker(stripped)
            if page is not None:
                # 페이지 경계는 블록 경계로 취급 → 블록의 페이지 범위가 정확해집니다.
                flush()
                current_page = page
                continue

            if not stripped:
                flush()
                continue

            if _HEADING_RE.match(stripped) and buf_kind != "table":
                flush()
                blocks.append(Block("heading", stripped, current_page, current_page))
                continue

            if buf_kind == "table":
                if is_table_row(stripped):
                    buf.append(stripped)
                    continue
                flush()  # 표가 끝났음 → 아래에서 일반 텍스트로 처리

            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if is_table_row(stripped) and is_table_separator(next_line):
                # "헤더행 + 구분선"이 연달아 나와야 표의 시작으로 인정합니다.
                flush()
                start("table")
                buf.append(stripped)
                continue

            if not buf:
                start("text")
            buf.append(line)

        flush()
        return blocks

    # ------------------------------------------------ 4-a. 큰 섹션 분할
    def _split_oversized_sections(self, sections: List[SectionUnit]) -> List[SectionUnit]:
        """parent_max_chars를 넘는 섹션을 블록 경계에서 여러 Parent로 나눕니다.

        섹션 하나가 수만 자(예: 부록 전체가 헤더 하나)인 경우, 그대로 Parent로 쓰면
        LLM 컨텍스트를 Parent 하나가 다 먹어 버립니다.
        """
        result: List[SectionUnit] = []
        limit = self.config.parent_max_chars
        for section in sections:
            if section.chars <= limit:
                result.append(section)
                continue

            # 블록 자체가 한도를 넘으면 먼저 쪼갭니다 (표는 헤더 재주입, 문단은 재귀 분할).
            fitted: List[Block] = []
            for block in section.blocks:
                caption: Optional[str] = None
                if block.kind == "table" and len(block.text) > limit and fitted and self._is_caption(fitted[-1]):
                    # 표가 여러 Parent로 쪼개질 때 캡션("(단위: 원)")이 첫 조각에만 남지 않도록
                    # 캡션 블록을 떼어 내 모든 표 조각에 복사합니다.
                    caption = fitted.pop().text.strip()
                fitted.extend(self._fit_block(block, limit, caption))

            groups: List[List[Block]] = []
            current: List[Block] = []
            for block in fitted:
                if current and self._joined_len(current + [block]) > limit:
                    # 제목과 표 캡션은 "다음 블록에 붙어 다녀야" 합니다. 그대로 끊으면 제목만 있는
                    # Parent나, 캡션과 표가 서로 다른 Parent로 갈라지는 문제가 생깁니다.
                    carry: List[Block] = []
                    while current and (
                        current[-1].kind == "heading"
                        or (not carry and block.kind == "table" and self._is_caption(current[-1]))
                    ):
                        carry.insert(0, current.pop())
                    if current:
                        groups.append(current)
                    current = carry
                current.append(block)
            if current:
                groups.append(current)

            total = len(groups)
            for idx, blocks in enumerate(groups, start=1):
                result.append(SectionUnit(section.path, dict(section.headers), blocks, part=(idx, total)))
        return result

    def _is_caption(self, block: Block) -> bool:
        return (
            block.kind == "text"
            and len(block.text.strip()) <= self.config.table_caption_max_chars
            and "\n\n" not in block.text
        )

    @staticmethod
    def _joined_len(blocks: List[Block]) -> int:
        return sum(len(b.text) for b in blocks) + 2 * max(len(blocks) - 1, 0)

    def _fit_block(self, block: Block, limit: int, caption: Optional[str] = None) -> List[Block]:
        if len(block.text) <= limit and caption is None:
            return [block]
        if block.kind == "table":
            table = parse_markdown_table(block.text)
            if table is not None:
                table.caption = caption or table.caption
                return [
                    Block("table", frag, block.page_start, block.page_end)
                    for frag in TableAwareSplitter(limit).split(table)
                ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=limit, chunk_overlap=0, separators=_TEXT_SEPARATORS, keep_separator="end"
        )
        return [Block("text", t, block.page_start, block.page_end) for t in splitter.split_text(block.text)]

    # ------------------------------------------------ 4-b. 작은 섹션 병합
    def _merge_small_sections(self, sections: List[SectionUnit]) -> List[_ParentGroup]:
        """너무 작은 섹션을 같은 장 안의 인접 섹션과 묶어 하나의 Parent로 만듭니다.

        "2.3 문의처" 같은 두세 줄짜리 섹션이 단독 Parent가 되면, 검색돼도 LLM에게 줄 문맥이
        빈약합니다. 반대로 다른 장(최상위 헤더)과 섞으면 무관한 내용이 문맥을 오염시키므로
        병합은 같은 top_key 안에서만 허용합니다.
        """
        c = self.config
        groups: List[_ParentGroup] = []
        for section in sections:
            if groups:
                last = groups[-1]
                same_chapter = last.units[-1].top_key == section.top_key
                small = last.chars < c.parent_min_chars or section.chars < c.parent_min_chars
                fits = last.chars + 2 + section.chars <= c.parent_max_chars
                if same_chapter and small and fits:
                    last.units.append(section)
                    continue
            groups.append(_ParentGroup(units=[section]))
        return groups

    # ------------------------------------------------ 5, 6. Document 생성
    def _build_documents(
        self, doc: ParsedDocument, groups: List[_ParentGroup]
    ) -> Tuple[List[Document], List[Document]]:
        parents: List[Document] = []
        children: List[Document] = []

        for p_idx, group in enumerate(groups):
            # 결정적 ID: 같은 문서를 다시 청킹하면 같은 ID가 나와야 upsert/삭제가 가능합니다.
            parent_id = f"{doc.doc_id}-p{p_idx:04d}"
            label = self._group_label(group.units)
            all_blocks = [b for u in group.units for b in u.blocks]
            p_start, p_end = _page_range(all_blocks)

            parent_body = "\n\n".join(b.text for b in all_blocks)
            parent_meta = self._base_metadata(doc, label, p_start, p_end)
            parent_meta.update(
                {
                    "chunk_level": "parent",
                    "parent_id": parent_id,
                    "subsection_count": len(group.units),
                }
            )
            if len(group.units) == 1 and group.units[0].part:
                cur, total = group.units[0].part
                parent_meta["part"] = f"{cur}/{total}"
            parents.append(
                Document(page_content=self._with_tag(doc.source, label, parent_body), metadata=parent_meta)
            )

            c_idx = 0
            for unit in group.units:
                # Child 태그는 병합된 그룹 라벨이 아니라 "자기 섹션"의 정확한 경로를 씁니다.
                # 검색 단계에서 "2.2 할인 규정"을 찾는 질의가 정확히 그 Child에 걸려야 하기 때문입니다.
                unit_label = self._unit_label(unit)
                tag = self._make_tag(doc.source, unit_label)
                budget = max(self.config.child_chunk_chars - len(tag) - 1, self.config.min_child_body_chars)

                for piece in self._split_children(unit, budget):
                    if len(re.sub(r"\s", "", piece.text)) < self.config.min_child_chars:
                        continue
                    child_meta = self._base_metadata(doc, unit_label, piece.page_start, piece.page_end)
                    child_meta.update(
                        {
                            "chunk_level": "child",
                            "parent_id": parent_id,
                            "child_id": f"{parent_id}-c{c_idx:03d}",
                            "chunk_index": c_idx,
                            "chunk_type": piece.kind,
                        }
                    )
                    child_meta.update(unit.headers)  # h1/h2/h3 → 메타데이터 필터 검색용
                    children.append(Document(page_content=f"{tag}\n{piece.text}", metadata=child_meta))
                    c_idx += 1

        return parents, children

    def _split_children(self, unit: SectionUnit, budget: int) -> List[ChildPiece]:
        """섹션 블록들을 Child 예산 안에서 묶습니다. 표는 항상 헤더를 가진 조각으로 분리됩니다."""
        pieces: List[ChildPiece] = []
        buffer: List[Block] = []
        buffer_len = 0
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=min(self.config.child_overlap_chars, budget // 4),
            separators=_TEXT_SEPARATORS,
            keep_separator="end",
        )

        def flush() -> None:
            nonlocal buffer, buffer_len
            # 제목 블록만 남은 버퍼("## 2.1 단가" 한 줄)는 Child로 만들지 않습니다.
            # 섹션명은 이미 태그에 들어 있어 검색 가치가 없는 중복 청크가 되기 때문입니다.
            if any(b.kind != "heading" for b in buffer):
                start, end = _page_range(buffer)
                pieces.append(ChildPiece("\n\n".join(b.text for b in buffer), "text", start, end))
            buffer, buffer_len = [], 0

        for block in unit.blocks:
            if block.kind == "table":
                table = parse_markdown_table(block.text)
                if table is not None and table.caption:
                    # Parent 단계에서 이미 캡션이 붙은 표 조각 → 그 캡션을 그대로 유지
                    flush()
                    for fragment in TableAwareSplitter(budget).split(table):
                        pieces.append(ChildPiece(fragment, "table", block.page_start, block.page_end))
                    continue
                if table is not None:
                    table.caption = self._caption_from(buffer)
                    if table.caption and all(b.kind == "heading" for b in buffer[:-1]):
                        # 버퍼가 "제목 + 캡션"뿐이면 캡션은 표 조각마다 복사되므로
                        # 따로 Child를 만들면 내용 없는 중복 청크가 됩니다 → 버립니다.
                        buffer, buffer_len = [], 0
                    flush()
                    for fragment in TableAwareSplitter(budget).split(table):
                        pieces.append(ChildPiece(fragment, "table", block.page_start, block.page_end))
                    continue
                # 표 모양이 깨진 경우 일반 텍스트로 처리 (아래로 진행)

            if len(block.text) > budget:
                flush()
                for text in text_splitter.split_text(block.text):
                    pieces.append(ChildPiece(text, "text", block.page_start, block.page_end))
                continue

            add_len = len(block.text) + (2 if buffer else 0)
            if buffer and buffer_len + add_len > budget:
                flush()
                add_len = len(block.text)
            buffer.append(block)
            buffer_len += add_len

        flush()
        return pieces

    def _caption_from(self, buffer: List[Block]) -> Optional[str]:
        """표 바로 앞 블록이 짧으면 표 캡션으로 사용합니다 ("표 3. 단가표", "(단위: 원)" 등).

        제목(heading) 블록은 캡션으로 쓰지 않습니다. 섹션명은 이미 메타데이터 태그에 들어 있어
        캡션으로 또 붙이면 모든 표 조각에 같은 문구가 두 번씩 들어갑니다.
        """
        if buffer and self._is_caption(buffer[-1]):
            return buffer[-1].text.strip() or None
        return None

    # ------------------------------------------------ 메타데이터 태그 / 라벨
    @staticmethod
    def _make_tag(source: str, section: str) -> str:
        # 요구사항의 형식을 정확히 따릅니다. 대괄호/쉼표가 섹션명에 섞여도 파싱할 일은 없고
        # (사람과 LLM이 읽는 용도) 임베딩·BM25에 문서명과 섹션명을 반영하는 것이 목적입니다.
        return f"[문서명: {source}, 섹션: {section}]"

    def _with_tag(self, source: str, section: str, body: str) -> str:
        return f"{self._make_tag(source, section)}\n{body}"

    def _unit_label(self, unit: SectionUnit) -> str:
        return " > ".join(unit.path) if unit.path else self.config.default_section

    def _group_label(self, units: List[SectionUnit]) -> str:
        """병합 Parent의 섹션 라벨: 공통 상위 경로 + 하위 섹션 나열.

        예) ("2장", "2.1 단가") + ("2장", "2.2 할인") → "2장 > 2.1 단가, 2.2 할인"
        """
        if len(units) == 1:
            return self._unit_label(units[0])

        paths = [u.path for u in units]
        prefix: List[str] = []
        for level_values in zip(*paths):
            if len(set(level_values)) != 1:
                break
            prefix.append(level_values[0])

        remainders: List[str] = []
        includes_intro = False  # 공통 상위 섹션 자신의 도입부 본문이 포함되었는가
        for path in paths:
            rest = " > ".join(path[len(prefix):])
            if not rest:
                includes_intro = True
            elif rest not in remainders:
                remainders.append(rest)

        prefix_label = " > ".join(prefix)
        if not remainders:
            return prefix_label or self.config.default_section

        limit = self.config.max_section_label_items
        items = ", ".join(remainders[:limit])
        if len(remainders) > limit:
            items += f" 외 {len(remainders) - limit}개"
        if not prefix_label:
            return items
        # 예) "1장 개요"의 도입부 + "1.1 적용 범위" 병합 → "1장 개요 (1.1 적용 범위 포함)"
        #     "1장 개요 > 1.1 적용 범위"라고 쓰면 도입부가 1.1절 소속인 것처럼 잘못 인용됩니다.
        if includes_intro:
            return f"{prefix_label} ({items} 포함)"
        return f"{prefix_label} > {items}"

    @staticmethod
    def _base_metadata(
        doc: ParsedDocument, section: str, page_start: Optional[int], page_end: Optional[int]
    ) -> Dict[str, object]:
        """Chroma 호환 메타데이터 (str/int/float/bool만 허용, None 금지).

        Chroma는 None이나 list 값이 들어오면 upsert 자체가 실패하므로, 페이지 정보가 없는
        문서(DOCX, Excel)는 키를 아예 넣지 않습니다.
        """
        meta: Dict[str, object] = {
            "doc_id": doc.doc_id,
            "source": doc.source,
            "source_path": doc.source_path,
            "file_type": doc.file_type,
            "parser_used": doc.parser_used,
            "section": section,
        }
        if page_start is not None:
            end = page_end if page_end is not None else page_start
            meta["page_start"] = page_start
            meta["page_end"] = end
            meta["pages"] = str(page_start) if page_start == end else f"{page_start}-{end}"
        return meta
