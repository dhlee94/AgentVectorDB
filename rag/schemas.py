"""모듈 간에 주고받는 데이터 구조.

[설계 의도]
- Parser와 Chunker 사이의 계약은 `ParsedDocument` 하나뿐입니다.
  Chunker는 원본이 PDF인지 Excel인지 전혀 모르고 "페이지 마커가 들어 있는 Markdown"만
  다룹니다. 이 경계 덕분에 파서를 추가/교체해도 청킹 로직은 손대지 않아도 됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class ParsedDocument:
    """파서의 공통 출력.

    Attributes:
        doc_id: 파일 경로에서 유도한 결정적(deterministic) ID. 재인덱싱 시 기존 청크를
            정확히 찾아 지우기 위해 랜덤 UUID가 아닌 해시를 씁니다.
        source: 사람이 읽는 문서명(파일명). 인용과 메타데이터 태그에 그대로 노출됩니다.
        source_path: 절대 경로.
        file_type: "pdf" | "xlsx" | "docx" 등.
        markdown: 페이지 마커(`<!-- page: N -->`)가 포함될 수 있는 Markdown 본문.
        parser_used: 실제로 성공한 파서 이름 (Fallback 여부 추적용).
        metadata: 페이지 수, 시트 목록 등 포맷별 부가 정보.
        warnings: 치명적이지 않은 문제(Fallback 발생, 빈 시트 skip 등) 기록.
    """

    doc_id: str
    source: str
    source_path: str
    file_type: str
    markdown: str
    parser_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ParseFailure:
    """파싱 실패 리포트 한 건."""

    source_path: str
    error_type: str
    reason: str


@dataclass
class ChunkingResult:
    """청커의 출력. Parent는 Docstore로, Child는 Vector DB와 BM25로 갑니다."""

    parents: List[Document]
    children: List[Document]
    warnings: List[str] = field(default_factory=list)


@dataclass
class RetrievedParent:
    """하이브리드 검색 결과 한 건 (Parent 단위)."""

    parent_id: str
    document: Document
    score: float
    # 이 Parent를 끌어올린 Child들과, 어느 검색기(dense/sparse)가 찾았는지 — 튜닝/디버깅용
    matched_child_ids: List[str] = field(default_factory=list)
    retrievers: List[str] = field(default_factory=list)


@dataclass
class Citation:
    index: int  # 답변 본문의 [n] 번호
    source: str
    section: str
    pages: Optional[str]
    parent_id: str
    source_path: str

    def format(self) -> str:
        page = f" | p.{self.pages}" if self.pages else ""
        return f"[{self.index}] {self.source} | {self.section}{page}"


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    # LLM에 전달된 전체 컨텍스트 목록 (인용되지 않은 것 포함)
    contexts: List[Citation] = field(default_factory=list)
    # 답변이 인용 근거를 갖췄는가. False면 UI에서 "근거 부족" 경고를 표시해야 합니다.
    grounded: bool = False
    warnings: List[str] = field(default_factory=list)
    model: Optional[str] = None
    stop_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)

    def format(self) -> str:
        lines = [self.answer]
        if self.citations:
            lines += ["", "참고 문서"] + [f"- {c.format()}" for c in self.citations]
        if self.warnings:
            lines += ["", "경고"] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class IngestReport:
    indexed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    failures: List[ParseFailure] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"인덱싱 {len(self.indexed)}건, 변경 없음 {len(self.unchanged)}건, "
            f"삭제 {len(self.removed)}건, 실패 {len(self.failures)}건"
        )
