"""표 파편화 방지 분할기 (전략 B — 표 헤더 재주입).

[문제]
    | 항목 | 단가 | 수량 |      ← 청크 1에만 헤더가 있음
    |------|------|------|
    | A    | 100  | 3    |
    -------- 청크 경계 --------
    | C    | 300  | 1    |      ← 청크 2: "300"이 단가인지 수량인지 알 수 없음
일반 텍스트 분할기는 표를 글자 수로만 자르므로 (1) 행 중간이 잘리고 (2) 두 번째 조각부터
헤더가 사라집니다. 헤더 없는 숫자 나열은 임베딩도 무의미하고 LLM도 해석하지 못합니다.

[해결]
1. 표를 헤더(헤더행 + 구분선)와 데이터 행으로 파싱합니다.
2. 자르는 위치는 항상 "행 경계"입니다 (행 중간 절단 금지).
3. 모든 조각의 맨 앞에 헤더 두 줄을 복사해 넣어, 각 조각이 독립된 완전한 표가 되게 합니다.
4. 표 바로 위의 짧은 문단(예: "표 3. 2025년 단가표")을 캡션으로 받아 모든 조각에 붙이고,
   두 번째 조각부터는 "(계속)"을 표시해 LLM이 같은 표의 연속임을 알게 합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from rag.markdown_utils import is_table_row, is_table_separator

logger = logging.getLogger(__name__)


@dataclass
class MarkdownTable:
    header: str
    separator: str
    rows: List[str]
    caption: Optional[str] = None

    @property
    def header_block(self) -> str:
        return f"{self.header}\n{self.separator}"

    def render(self, rows: List[str], caption: Optional[str] = None) -> str:
        body = "\n".join([self.header_block] + rows)
        return f"{caption}\n\n{body}" if caption else body


def parse_markdown_table(text: str) -> Optional[MarkdownTable]:
    """Markdown 표 문자열을 파싱합니다. 표 형식이 아니면 None.

    표 앞에 캡션 줄이 붙어 있는 형태(`render(..., caption)`의 출력)도 받아들입니다.
    Parent 단계에서 이미 캡션과 함께 잘린 표 조각을 Child 단계에서 다시 자를 수 있어야 하기 때문입니다.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    start = next(
        (i for i in range(len(lines) - 1) if is_table_row(lines[i]) and is_table_separator(lines[i + 1])),
        None,
    )
    if start is None:
        return None
    rows = lines[start + 2 :]
    if not all(is_table_row(row) for row in rows):
        return None  # 표 뒤에 일반 텍스트가 섞여 있으면 순수한 표 블록이 아님
    caption = "\n".join(lines[:start]) or None
    return MarkdownTable(header=lines[start], separator=lines[start + 1], rows=rows, caption=caption)


class TableAwareSplitter:
    """행 경계에서만 자르고, 모든 조각에 헤더를 재주입하는 표 분할기."""

    CONTINUED_SUFFIX = " (계속)"

    def __init__(self, chunk_size: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size는 양수여야 합니다")
        self.chunk_size = chunk_size

    def split(self, table: MarkdownTable) -> List[str]:
        """표를 chunk_size 이하의 조각들로 나눕니다. 각 조각은 헤더를 포함한 완전한 표입니다."""
        whole = table.render(table.rows, table.caption)
        if len(whole) <= self.chunk_size or not table.rows:
            return [whole]

        # 헤더와 캡션이 차지하는 고정 비용을 먼저 빼야 "헤더 복사 때문에 예산 초과"가 안 납니다.
        # 두 번째 조각부터는 캡션에 "(계속)"이 붙으므로 그 길이 기준으로 보수적으로 계산합니다.
        caption_cost = len(table.caption) + len(self.CONTINUED_SUFFIX) + 2 if table.caption else 0
        fixed_cost = len(table.header_block) + 1 + caption_cost
        row_budget = self.chunk_size - fixed_cost

        if row_budget <= 0:
            # 헤더만으로 예산을 넘는 초광폭 표. 헤더를 빼면 의미가 없으므로 예산 초과를 감수하고
            # 한 행씩 헤더와 함께 내보냅니다 (정보 보존 > 크기 제한).
            logger.warning("표 헤더가 청크 예산(%d자)보다 깁니다. 행 단위로 분할합니다.", self.chunk_size)
            row_budget = 1

        groups: List[List[str]] = []
        current: List[str] = []
        current_len = 0
        for row in table.rows:
            row_len = len(row) + 1  # 줄바꿈 포함
            if current and current_len + row_len > row_budget:
                groups.append(current)
                current, current_len = [], 0
            if not current and row_len > row_budget:
                # 행 하나가 예산보다 긴 경우: 행을 자르면 셀 경계가 깨지므로 단독 조각으로 둡니다.
                logger.debug("단일 행이 예산 초과(%d자) — 단독 조각으로 유지", row_len)
            current.append(row)
            current_len += row_len
        if current:
            groups.append(current)

        fragments: List[str] = []
        for i, rows in enumerate(groups):
            caption = table.caption
            if caption and i > 0 and not caption.endswith(self.CONTINUED_SUFFIX):
                caption = f"{caption}{self.CONTINUED_SUFFIX}"
            # 핵심: 모든 조각이 table.render()를 거치므로 헤더 두 줄이 항상 맨 앞에 붙습니다.
            fragments.append(table.render(rows, caption))
        return fragments
