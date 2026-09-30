"""인덱스 저장소: Chroma(Child 벡터) + Docstore(Parent) + BM25(Child 키워드) + 매니페스트.

[설계 의도]
- 네 저장소는 반드시 "같은 청크 집합"을 가리켜야 합니다. 하나라도 어긋나면
  (예: Chroma엔 Child가 있는데 Docstore에 Parent가 없음) 검색은 되는데 답변 문맥이 비는
  찾기 어려운 버그가 생깁니다. 그래서 쓰기/삭제는 모두 이 클래스 한 곳을 거치고,
  문서 단위로 "전부 성공 아니면 롤백"하도록 만들었습니다.
- BM25는 별도 파일로 저장하지 않고 Chroma의 Child를 읽어 메모리에서 재구축합니다.
  원천 데이터를 Chroma 하나로 유지해야 두 인덱스가 어긋날 여지가 없습니다.
- 매니페스트(manifest.json)는 "파일 경로 → 해시, 청크 ID 목록"을 기록합니다.
  변경된 파일만 재인덱싱하고, 삭제된 파일의 청크를 정확히 지우는 근거가 됩니다.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config import StoreConfig
from rag.exceptions import IndexConfigMismatchError, IndexingError
from rag.indexing.korean_tokenizer import KoreanTokenizer
from rag.schemas import ChunkingResult

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1


def build_default_embeddings(config: StoreConfig) -> Embeddings:
    """기본 임베딩(bge-m3, 로컬 실행). 무거운 의존성이므로 실제로 필요할 때만 import 합니다."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise IndexingError(
            "기본 임베딩(bge-m3)에는 `pip install langchain-huggingface sentence-transformers`가 필요합니다. "
            "또는 RAGPipeline(embeddings=...)로 다른 Embeddings 구현을 주입하세요."
        ) from exc

    def load(device: Optional[str]) -> Embeddings:
        return HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": device} if device else {},
            # 정규화해야 코사인 유사도가 내적과 같아지고, 문서 길이에 따른 점수 편향이 줄어듭니다.
            encode_kwargs={"normalize_embeddings": True},
        )

    embeddings = load(config.embedding_device)
    device = str(getattr(getattr(embeddings, "_client", None), "device", config.embedding_device or "?"))
    worst = embedding_self_test(embeddings)
    if worst >= _SELF_TEST_MIN_COSINE:
        logger.info("임베딩 자기검사 통과 (device=%s, 최소 cos=%.4f)", device, worst)
        return embeddings

    if device.startswith("cpu"):
        raise IndexingError(
            f"임베딩 자기검사 실패: CPU에서도 단건/배치 인코딩 결과가 다릅니다 (최소 cos={worst:.3f}). "
            "torch / sentence-transformers 버전을 확인하세요."
        )
    # 실제 사례: torch 2.8 + Apple MPS에서 bge-m3로 "짧은 문장을 1건만" 인코딩하면 엉뚱한 벡터가
    # 나옵니다(cos 0.16~0.24). 질의는 항상 짧은 문장 1건이라 Dense 검색이 오류 없이 망가지고,
    # BM25가 일부를 가려 주기 때문에 발견도 어렵습니다. 속도보다 정확성이 우선이므로 CPU로 전환합니다.
    logger.warning(
        "임베딩 자기검사 실패 (device=%s, 최소 cos=%.3f) → CPU로 전환합니다. "
        "이 장치의 단건 인코딩 결과가 배치 결과와 다릅니다.", device, worst,
    )
    embeddings = load("cpu")
    worst = embedding_self_test(embeddings)
    if worst < _SELF_TEST_MIN_COSINE:
        raise IndexingError(f"임베딩 자기검사 실패: CPU 전환 후에도 불일치 (최소 cos={worst:.3f})")
    return embeddings


# 같은 문장의 단건/배치 인코딩은 수치 오차 수준(>0.999)으로 같아야 합니다. 여유를 두고 0.99.
_SELF_TEST_MIN_COSINE = 0.99
# 문제는 짧은 문장에서만 나타나므로 짧은 질의형 문장 위주로 구성합니다.
_SELF_TEST_PROBES = [
    "볼펜 할인율은?",
    "짧음",
    "단가 알려줘",
    "SKU-00123 재고",
    "재택근무는 주 2회까지 허용된다",
    "Chapter 2 unit price",
]


def embedding_self_test(embeddings: Embeddings) -> float:
    """단건 인코딩(embed_query 경로)과 배치 인코딩(embed_documents 경로)의 일치도를 검사합니다.

    Returns:
        프로브 문장들 중 최소 코사인 유사도 (1.0에 가까워야 정상)
    """
    batch = embeddings.embed_documents(_SELF_TEST_PROBES)
    worst = 1.0
    for text, batch_vec in zip(_SELF_TEST_PROBES, batch):
        single_vec = embeddings.embed_query(text)
        dot = sum(a * b for a, b in zip(single_vec, batch_vec))
        norm = (sum(a * a for a in single_vec) ** 0.5) * (sum(b * b for b in batch_vec) ** 0.5)
        worst = min(worst, dot / norm if norm else 0.0)
    return worst


class IndexStore:
    def __init__(
        self,
        config: StoreConfig,
        embeddings: Embeddings,
        tokenizer: Optional[KoreanTokenizer] = None,
    ) -> None:
        from langchain.storage import LocalFileStore, create_kv_docstore
        from langchain_chroma import Chroma

        self.config = config
        self.tokenizer = tokenizer or KoreanTokenizer()
        self._persist_dir = Path(config.persist_dir).expanduser().resolve()
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._persist_dir / "manifest.json"
        self._manifest = self._load_manifest()
        self._check_embedding_model()

        self._vectorstore = Chroma(
            collection_name=config.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self._persist_dir / "chroma"),
            # 기본값(L2)이 아닌 코사인 거리: 정규화된 문장 임베딩의 표준 비교 방식
            collection_metadata={"hnsw:space": "cosine"},
        )
        # InMemoryStore는 프로세스가 끝나면 Parent가 사라집니다 → 디스크 영속 저장소 사용
        self._docstore = create_kv_docstore(LocalFileStore(str(self._persist_dir / "parents")))

        # BM25는 쓰기가 일어나면 "dirty"로 표시했다가, 다음 검색 때 한 번만 재구축합니다.
        # (문서 100개를 인제스트하면서 매번 재구축하면 O(N^2)이 되기 때문)
        self._bm25: Optional[Any] = None
        self._bm25_docs: List[Document] = []
        self._bm25_dirty = True
        self._bm25_lock = threading.Lock()

    # ================================================================ manifest
    def _load_manifest(self) -> Dict[str, Any]:
        if not self._manifest_path.exists():
            return {"version": _MANIFEST_VERSION, "embedding_model": None, "documents": {}}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # 매니페스트가 깨졌다고 조용히 빈 것으로 시작하면, 모든 파일을 "신규"로 보고
            # 중복 인덱싱합니다. 사람이 판단하도록 명확히 실패시킵니다.
            raise IndexingError(f"매니페스트를 읽을 수 없습니다: {exc}", source=self._manifest_path) from exc
        data.setdefault("documents", {})
        return data

    def _save_manifest(self) -> None:
        """임시 파일에 쓴 뒤 교체(atomic rename) — 쓰는 도중 프로세스가 죽어도 파일이 반쯤 깨지지 않습니다."""
        payload = json.dumps(self._manifest, ensure_ascii=False, indent=2, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._persist_dir), prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self._manifest_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _check_embedding_model(self) -> None:
        recorded = self._manifest.get("embedding_model")
        current = self.config.embedding_model
        if recorded and recorded != current and self._manifest["documents"]:
            raise IndexConfigMismatchError(
                f"기존 인덱스는 '{recorded}'로 임베딩되었는데 현재 설정은 '{current}'입니다. "
                f"벡터 공간이 달라 검색이 무의미해지므로 persist_dir을 비우고 재인덱싱하세요.",
                source=self._persist_dir,
            )
        self._manifest["embedding_model"] = current

    def get_manifest_entry(self, source_path: str) -> Optional[Dict[str, Any]]:
        return self._manifest["documents"].get(source_path)

    def indexed_paths(self) -> List[str]:
        return list(self._manifest["documents"])

    # ================================================================== write
    def upsert_document(self, source_path: str, doc_id: str, file_hash: str, chunks: ChunkingResult) -> None:
        """문서 하나의 청크를 교체 저장합니다 (기존 청크 삭제 → 신규 저장 → 매니페스트 기록).

        실패하면 이번에 쓴 것을 지우고 IndexingError를 던집니다. 매니페스트는 성공했을 때만
        갱신되므로, 다음 인제스트 때 이 파일은 "변경됨"으로 다시 시도됩니다.
        """
        if self.get_manifest_entry(source_path):
            self.delete_document(source_path)

        child_ids = [str(c.metadata["child_id"]) for c in chunks.children]
        parent_ids = [str(p.metadata["parent_id"]) for p in chunks.parents]
        if len(set(child_ids)) != len(child_ids):
            raise IndexingError("Child ID가 중복되었습니다 (청커 버그)", source=source_path)

        try:
            # Parent 먼저 저장: Child가 검색됐는데 Parent가 없는 순간이 생기지 않도록
            self._docstore.mset(list(zip(parent_ids, chunks.parents)))
            batch = self.config.add_batch_size
            for i in range(0, len(chunks.children), batch):
                self._vectorstore.add_documents(
                    chunks.children[i : i + batch], ids=child_ids[i : i + batch]
                )
        except Exception as exc:  # noqa: BLE001 — 임베딩 모델/Chroma/파일시스템 오류 전부
            logger.error("인덱싱 실패 → 롤백: %s (%s)", source_path, exc)
            self._safe_delete(child_ids, parent_ids)
            raise IndexingError(f"인덱싱 실패: {type(exc).__name__}: {exc}", source=source_path) from exc

        self._manifest["documents"][source_path] = {
            "doc_id": doc_id,
            "file_hash": file_hash,
            "parent_ids": parent_ids,
            "child_ids": child_ids,
            "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._save_manifest()
        self._bm25_dirty = True

    def delete_document(self, source_path: str) -> bool:
        entry = self._manifest["documents"].get(source_path)
        if not entry:
            return False
        self._safe_delete(entry.get("child_ids", []), entry.get("parent_ids", []))
        del self._manifest["documents"][source_path]
        self._save_manifest()
        self._bm25_dirty = True
        return True

    def _safe_delete(self, child_ids: Sequence[str], parent_ids: Sequence[str]) -> None:
        # 롤백/삭제 경로에서 또 예외가 나면 원래 오류가 가려지므로 로그만 남깁니다.
        try:
            if child_ids:
                self._vectorstore.delete(ids=list(child_ids))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma 삭제 실패(고아 청크가 남을 수 있음): %s", exc)
        try:
            if parent_ids:
                self._docstore.mdelete(list(parent_ids))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Docstore 삭제 실패: %s", exc)

    # ================================================================== read
    def count_children(self) -> int:
        return sum(len(e.get("child_ids", [])) for e in self._manifest["documents"].values())

    def get_parents(self, parent_ids: Sequence[str]) -> List[Optional[Document]]:
        return self._docstore.mget(list(parent_ids))

    def dense_search(
        self, query: str, k: int, where: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """벡터 유사도 순 Child 목록 (RRF는 순위만 쓰므로 점수는 반환하지 않습니다)."""
        if self.count_children() == 0:
            return []
        k = min(k, self.count_children())  # Chroma는 k > 전체 개수면 경고를 냅니다
        return self._vectorstore.similarity_search(query, k=k, filter=_to_chroma_filter(where))

    def sparse_search(
        self, query: str, k: int, where: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """BM25 점수 순 Child 목록. 점수 0(공통 토큰 없음)인 문서는 제외합니다."""
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        tokens = self.tokenizer(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[Document] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            doc = self._bm25_docs[i]
            if where and not _match_where(doc.metadata, where):
                continue
            results.append(doc)
            if len(results) >= k:
                break
        return results

    # ================================================================== BM25
    def rebuild_bm25(self) -> None:
        """Chroma에 저장된 Child 전체로 BM25 인덱스를 다시 만듭니다."""
        from rank_bm25 import BM25Okapi

        with self._bm25_lock:
            docs: List[Document] = []
            collection = self._vectorstore._collection  # 페이지네이션을 위해 원본 컬렉션 사용
            total = collection.count()
            page = 5000
            for offset in range(0, total, page):
                batch = collection.get(include=["documents", "metadatas"], limit=page, offset=offset)
                for text, meta in zip(batch["documents"], batch["metadatas"]):
                    docs.append(Document(page_content=text or "", metadata=dict(meta or {})))

            if not docs:
                self._bm25, self._bm25_docs = None, []
            else:
                corpus = self.tokenizer.tokenize_many(d.page_content for d in docs)
                # 토큰이 하나도 없는 문서가 섞이면 BM25Okapi 내부 평균 길이 계산은 되지만
                # 전부 비어 있으면 0으로 나누기 오류 → 방어
                if not any(corpus):
                    self._bm25, self._bm25_docs = None, []
                else:
                    self._bm25, self._bm25_docs = BM25Okapi(corpus), docs
            self._bm25_dirty = False
            logger.info("BM25 인덱스 재구축: Child %d개", len(self._bm25_docs))

    def _ensure_bm25(self) -> None:
        if self._bm25_dirty:
            self.rebuild_bm25()


# ---------------------------------------------------------------------------
# 메타데이터 필터 (Chroma와 BM25가 같은 의미로 동작하도록 공통 규칙 사용)
# ---------------------------------------------------------------------------
def _to_chroma_filter(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """{"source": "a.pdf", "file_type": "pdf"} → Chroma 문법 ({"$and": [...]}).

    Chroma는 조건이 2개 이상이면 반드시 $and로 감싸야 하고, 1개면 감싸면 안 됩니다.
    """
    if not where:
        return None
    clauses = [{key: {"$eq": value}} for key, value in where.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _match_where(metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in where.items())
