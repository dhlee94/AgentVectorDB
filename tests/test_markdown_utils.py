"""Markdown 공통 규약: 인라인 코드 오탐 제거, PDF 헤더 수준 복원, 표 구분선 판별."""

from __future__ import annotations

import re

import pytest

from rag.markdown_utils import (
    is_table_separator,
    normalize_text,
    restore_heading_levels,
    strip_symbolic_inline_code,
)


# ------------------------------------------------------ 인라인 코드 백틱 제거
@pytest.mark.parametrize(
    "src, expected",
    [
        ("`→` 초반 원칙을 잊지 않도록 `10` 스텝마다", "→ 초반 원칙을 잊지 않도록 10 스텝마다"),
        ("`→ 100` 개 파일", "→ 100 개 파일"),
        ("`1.` 장기 리팩토링", "1. 장기 리팩토링"),
        ("텍스트 중간 `#` 기호", "텍스트 중간 # 기호"),
        # 아래는 모두 유지되어야 하는 경우
        ("`#` 줄 맨 앞", "`#` 줄 맨 앞"),  # 벗기면 Markdown 헤더가 됨
        ("  `##` 들여쓴 줄", "  `##` 들여쓴 줄"),
        ("`|` 파이프", "`|` 파이프"),  # 표 행으로 오인 방지
        ('`ralph.set_goal("src/` 디렉토리', '`ralph.set_goal("src/` 디렉토리'),  # 글자 포함 → 코드일 수 있음
        ("```\n`10` 코드블록 안\n```", "```\n`10` 코드블록 안\n```"),
        ("``double`` ``42``", "``double`` ``42``"),
    ],
)
def test_strip_symbolic_inline_code(src, expected):
    assert strip_symbolic_inline_code(src)[0] == expected


def test_strip_symbolic_inline_code_is_idempotent():
    once, n = strip_symbolic_inline_code("`10` 스텝, `→` 화살표, 줄 중간 `#` 기호")
    assert n == 3  # 줄 중간의 `#`는 헤더가 될 위험이 없으므로 제거 대상
    assert strip_symbolic_inline_code(once) == (once, 0)


# ---------------------------------------------------------- 헤더 수준 복원
FLAT_BOOK = """## **책 제목**

## **저자의 말**

서문 본문.

## **1장. 시작** **시작하며: 부제**

본문 1.

## **그러나 현실은 냉혹했다.**

## **실전 체크리스트**

## **1. 첫째 원칙**

## **6부. 다음 부 배너** **2장. 규칙 파일: AGENTS.md,** **CLAUDE.md** **규칙 파일이란**

본문 2.

## **실전 체크리스트**
"""


def _headings(md: str):
    return [line for line in md.split("\n") if re.match(r"^#{1,6} ", line)]


def test_restore_heading_levels_builds_hierarchy():
    md, stats = restore_heading_levels(FLAT_BOOK)
    assert stats["applied"] == 1
    assert _headings(md) == [
        "# 책 제목",  # 첫 장 이전 → 최상위
        "# 저자의 말",
        "# 1장. 시작",  # 한 줄에 합쳐진 장 제목 + 부제 분리
        "## 시작하며: 부제",
        "## 실전 체크리스트",
        "### 1. 첫째 원칙",
        "# 2장. 규칙 파일: AGENTS.md, CLAUDE.md",  # 쉼표로 끝난 조각은 다음 조각과 합침
        "## 규칙 파일이란",
        "## 실전 체크리스트",
    ]
    # 부 배너와 강조 문장은 헤더가 아닌 본문으로 강등 (텍스트는 보존)
    assert "**6부. 다음 부 배너**" in md
    assert "**그러나 현실은 냉혹했다.**" in md
    assert stats["parts_demoted"] == 1 and stats["demoted"] == 1


def test_restore_heading_levels_is_idempotent():
    md, _ = restore_heading_levels(FLAT_BOOK)
    assert restore_heading_levels(md) == (md, {"applied": 0})


def test_restore_heading_levels_skips_hierarchical_docs():
    doc = "# 1장. 개요\n\n## 배경\n\n### 세부\n\n# 2장. 방법\n\n## 절차\n"
    assert restore_heading_levels(doc) == (doc, {"applied": 0})


def test_restore_heading_levels_skips_docs_without_chapters():
    doc = "## 소개\n\n본문\n\n## 방법\n\n본문\n"
    assert restore_heading_levels(doc)[1] == {"applied": 0}


# ------------------------------------------------------------------ 기타
@pytest.mark.parametrize("line", ["|---|---|", "|:-----|------:|", "| --- | :---: |"])
def test_table_separator_variants(line):
    assert is_table_separator(line)


@pytest.mark.parametrize("line", ["| a | b |", "---", "| 1 | 2 |"])
def test_not_table_separator(line):
    assert not is_table_separator(line)


def test_normalize_text_nfc_and_control_chars():
    import unicodedata

    nfd = unicodedata.normalize("NFD", "견적서")
    out = normalize_text(f"{nfd}\x00\x0c 본문\n\n\n\n끝")
    assert out == "견적서 본문\n\n끝"
