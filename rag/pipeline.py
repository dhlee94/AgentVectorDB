"""RAGPipeline: 폴더 인제스트(로드→파싱→청킹→인덱싱)와 질의(하이브리드 검색→인용 답변)를 묶는 진입점.

사용 예
    from rag.pipeline import RAGPipeline
    pipe = RAGPipeline()                        # ./config.yaml 설정으로 생성
    print(pipe.ingest("./data/inbox").summary())
    print(pipe.query("A4용지 단가는?").format())

CLI
    python -m rag ingest ./data/inbox
    python -m rag query "A4용지 단가는?"
    python -m rag --config prod.yaml query "..."   # 다른 설정 파일 사용

[설계 의도]
- 인제스트는 파일 단위로 격리합니다. 파싱·청킹·인덱싱 어느 단계에서 실패해도 그 파일만
  실패 목록에 들어가고 나머지는 계속 처리됩니다.
- 파일 해시(SHA-256)가 매니페스트와 같으면 건너뜁니다. 폴더에 파일을 계속 쌓는 운영에서
  매번 전체를 재임베딩하지 않게 하는 핵심 장치입니다.
- 질의 시 검색 결과가 0건이면 LLM을 호출하지 않습니다. 빈 컨텍스트를 주면 모델이
  상식으로 답할 여지가 생기고(환각), 비용만 듭니다.
- LLM 호출이 최종 실패해도 예외를 던지지 않고 "검색된 문서 목록"을 담은 답변을 돌려줍니다.
  사용자는 최소한 어느 문서를 보면 되는지는 알 수 있어야 합니다.
"""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings

from rag.chunking import ParentChildChunker
from rag.config import PipelineConfig, load_config, load_env
from rag.exceptions import GenerationError, RAGError, RetrievalError
from rag.generation import NO_ANSWER, AnswerGenerator
from rag.indexing import IndexStore, KoreanTokenizer, build_default_embeddings
from rag.parsers import ParserRouter
from rag.retrieval import HybridParentRetriever
from rag.schemas import Citation, IngestReport, ParseFailure, RAGAnswer

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        embeddings: Optional[Embeddings] = None,
        llm_client: Optional[Any] = None,
        reranker: Optional[Any] = None,
    ) -> None:
        """
        Args:
            embeddings: LangChain Embeddings 구현. None이면 bge-m3(로컬)를 로드합니다.
                주입한 경우 config.store.embedding_model에 그 모델 이름을 맞춰 주세요
                (매니페스트의 모델 불일치 검사에 쓰입니다).
            llm_client: anthropic.Anthropic 호환 클라이언트 (테스트용 주입).
            reranker: `score(query, texts) -> List[float]`를 가진 재순위기 (선택).
        """
        # .env(API 키 등)를 먼저 환경변수로 올립니다. config를 주지 않으면 config.yaml(또는
        # RAG_CONFIG)을 읽고, 없으면 코드 기본값을 씁니다.
        load_env()
        self.config = config or load_config()
        self.config.validate()  # 잘못된 모델/파라미터 조합을 첫 질의 전에 차단
        self.router = ParserRouter(self.config.parser)
        self.chunker = ParentChildChunker(self.config.chunker)
        self.store = IndexStore(
            self.config.store,
            embeddings or build_default_embeddings(self.config.store),
            KoreanTokenizer(),
        )
        self.retriever = HybridParentRetriever(
            store=self.store, config=self.config.retriever, reranker=reranker
        )
        self.generator = AnswerGenerator(self.config.generation, client=llm_client)

    # ================================================================ ingest
    def ingest(self, folder: str, remove_missing: bool = True) -> IngestReport:
        """폴더의 문서를 인덱스와 동기화합니다 (신규/변경만 처리, 사라진 파일은 인덱스에서 제거).

        Args:
            remove_missing: True면 폴더에서 사라진 파일의 청크를 인덱스에서 삭제합니다.
        """
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise RAGError("인제스트 대상 폴더가 없습니다", source=root)

        report = IngestReport()
        files = self._scan(root)
        seen = set()

        for path in files:
            key = _path_key(path)
            seen.add(key)
            try:
                file_hash = _sha256(path)
            except OSError as exc:
                report.failures.append(ParseFailure(key, type(exc).__name__, f"파일 읽기 실패: {exc}"))
                continue

            entry = self.store.get_manifest_entry(key)
            if entry and entry.get("file_hash") == file_hash:
                report.unchanged.append(key)
                continue

            try:
                parsed = self.router.parse(path)
                chunks = self.chunker.split(parsed)
                self.store.upsert_document(key, parsed.doc_id, file_hash, chunks)
                report.indexed.append(key)
                logger.info(
                    "인덱싱 완료: %s (Parent %d, Child %d)", path.name, len(chunks.parents), len(chunks.children)
                )
            except RAGError as exc:
                # 예상된 실패(빈 문서, 암호, 청킹/인덱싱 오류) — 이 파일만 건너뜁니다.
                logger.warning("인제스트 건너뜀: %s", exc)
                report.failures.append(ParseFailure(key, type(exc).__name__, exc.reason))
            except Exception as exc:  # noqa: BLE001
                logger.exception("인제스트 중 예기치 못한 오류: %s", path)
                report.failures.append(ParseFailure(key, type(exc).__name__, str(exc)))

        if remove_missing:
            # 폴더 밖(다른 루트)에서 인덱싱된 문서까지 지우지 않도록 같은 루트 아래 경로만 대상으로 합니다.
            prefix = _path_key(root) + "/"
            for key in self.store.indexed_paths():
                if key.startswith(prefix) and key not in seen:
                    self.store.delete_document(key)
                    report.removed.append(key)

        # 모든 쓰기가 끝난 뒤 BM25를 한 번만 재구축합니다 (파일마다 재구축하면 O(N^2)).
        if report.indexed or report.removed:
            self.store.rebuild_bm25()
        logger.info("인제스트 결과: %s", report.summary())
        return report

    def _scan(self, root: Path) -> List[Path]:
        files: List[Path] = []
        for path in sorted(root.rglob("*")):
            rel_parts = path.relative_to(root).parts
            # 숨김 파일/폴더(.git, .DS_Store 등)와 Office 잠금 파일(~$)은 제외
            if any(part.startswith(".") for part in rel_parts):
                continue
            if path.is_file() and self.router.is_supported(path):
                files.append(path)
        return files

    # ================================================================= query
    def query(self, question: str, where: Optional[Dict[str, Any]] = None) -> RAGAnswer:
        """질문에 대한 인용 포함 답변.

        Args:
            where: 메타데이터 필터. 예) {"source": "견적_테스트.xlsx"}, {"file_type": "pdf"}
        """
        question = (question or "").strip()
        if not question:
            return RAGAnswer(question="", answer="질문이 비어 있습니다.", warnings=["빈 질문"])

        try:
            parents = self.retriever.retrieve(question, where=where)
        except RetrievalError as exc:
            logger.error("검색 실패: %s", exc)
            return RAGAnswer(question=question, answer="문서 검색에 실패했습니다.", warnings=[exc.reason])

        if not parents:
            # 근거가 0건이면 LLM을 부르지 않습니다 (환각 차단 + 비용 절감).
            return RAGAnswer(
                question=question,
                answer=NO_ANSWER,
                warnings=["질문과 관련된 문서를 찾지 못했습니다"],
            )

        try:
            return self.generator.generate(question, parents)
        except GenerationError as exc:
            logger.error("답변 생성 실패(retryable=%s): %s", exc.retryable, exc)
            contexts = [
                Citation(
                    index=i,
                    source=str(p.document.metadata.get("source", "")),
                    section=str(p.document.metadata.get("section", "")),
                    pages=p.document.metadata.get("pages"),
                    parent_id=p.parent_id,
                    source_path=str(p.document.metadata.get("source_path", "")),
                )
                for i, p in enumerate(parents, start=1)
            ]
            hint = " 잠시 후 다시 시도해 주세요." if exc.retryable else ""
            return RAGAnswer(
                question=question,
                answer=f"답변 생성에 실패했습니다.{hint} 아래 검색된 문서를 직접 확인하세요.",
                citations=contexts,  # 검색 결과라도 출처 목록으로 제공
                contexts=contexts,
                grounded=False,
                warnings=[exc.reason],
            )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _path_key(path: Path) -> str:
    """매니페스트 키. macOS NFD 경로를 NFC로 통일해야 같은 파일을 다른 파일로 오인하지 않습니다."""
    return unicodedata.normalize("NFC", str(path.resolve()))


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """대용량 파일도 메모리에 전부 올리지 않도록 1MB씩 읽어 해시합니다."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
