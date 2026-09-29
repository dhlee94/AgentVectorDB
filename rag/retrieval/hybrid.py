"""HybridParentRetriever: Dense + BM25 → RRF(Child 단위) → Parent 승격 (전략 B + C).

흐름
    질문 ─┬─ Dense (Chroma, bge-m3) → Child 후보 dense_k개 (순위 r_d)
          └─ Sparse(BM25, Kiwi)     → Child 후보 sparse_k개 (순위 r_s)
    RRF(child) = w_d/(rrf_k + r_d) + w_s/(rrf_k + r_s)       ← 한쪽에만 있으면 그쪽 항만
    Parent 점수 = 소속 Child들의 RRF 최댓값 → 상위 top_n개 Parent를 Docstore에서 조회
    (선택) Cross-Encoder로 Parent 재순위

[왜 Child 단계에서 융합하는가]
LangChain `ParentDocumentRetriever`는 Dense만 지원하고, `EnsembleRetriever`로 Parent끼리
합치면 BM25도 Parent(긴 텍스트)를 대상으로 해야 합니다. 긴 텍스트는 BM25 길이 정규화 때문에
점수가 희석되어, 짧은 Child에서 정확히 걸리던 제품코드가 묻힙니다. 두 검색기 모두 "작은 조각"
위에서 경쟁시키고, 결과만 Parent로 올리는 것이 Small-to-Big의 원래 의도에 맞습니다.

[왜 점수 합산이 아니라 RRF인가]
코사인 유사도(0~1)와 BM25 점수(0~수십)는 스케일이 달라 가중합하려면 정규화가 필요하고,
그 정규화는 질의마다 분포가 달라 불안정합니다. RRF는 순위만 쓰므로 정규화가 필요 없습니다.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag.config import RetrieverConfig
from rag.exceptions import RetrievalError
from rag.schemas import RetrievedParent

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """(선택) Parent 재순위기. 질문과 Parent 본문을 쌍으로 직접 비교해 RRF보다 정밀하게 정렬합니다.

    RRF는 "어느 검색기에서 몇 등이었나"만 보므로, 최종 3~5개를 고르는 단계에서는
    Cross-Encoder가 정확도를 크게 올립니다. 대신 후보마다 모델 추론이 필요해 느립니다.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("재순위에는 `pip install sentence-transformers`가 필요합니다") from exc
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


class HybridParentRetriever(BaseRetriever):
    """LangChain `BaseRetriever` 호환 — LCEL 체인에 그대로 꽂아 쓸 수 있습니다."""

    # IndexStore / RetrieverConfig는 pydantic 모델이 아니므로 임의 타입을 허용합니다.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any
    config: Any = None
    reranker: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        if self.config is None:
            self.config = RetrieverConfig()

    # ------------------------------------------------------------ LangChain API
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.retrieve(query)
        docs: List[Document] = []
        for r in results:
            meta = dict(r.document.metadata)
            meta["retrieval_score"] = r.score
            meta["retrievers"] = ",".join(r.retrievers)
            docs.append(Document(page_content=r.document.page_content, metadata=meta))
        return docs

    # ---------------------------------------------------------------- main
    def retrieve(self, query: str, where: Optional[Dict[str, Any]] = None) -> List[RetrievedParent]:
        """질문에 대한 상위 Parent 목록 (점수 내림차순).

        Args:
            where: 메타데이터 동등 조건 필터 (예: {"source": "견적서.xlsx"}).
                   Dense와 BM25에 똑같이 적용되어야 두 결과를 공정하게 합칠 수 있습니다.
        Raises:
            RetrievalError: Dense와 Sparse가 모두 실패한 경우
        """
        cfg = self.config
        query = (query or "").strip()
        if not query:
            return []

        # 1) 두 검색기 실행 — 한쪽이 실패해도 다른 쪽으로 계속 (부분 성공 > 전체 실패)
        dense, dense_err = self._safe_search("dense", self.store.dense_search, query, cfg.dense_k, where)
        sparse, sparse_err = self._safe_search("sparse", self.store.sparse_search, query, cfg.sparse_k, where)
        if dense_err and sparse_err:
            raise RetrievalError(f"Dense/Sparse 검색 모두 실패 (dense: {dense_err} / sparse: {sparse_err})")

        # 2) Child 단위 RRF 융합
        fused: Dict[str, float] = defaultdict(float)
        child_parent: Dict[str, str] = {}
        child_hits: Dict[str, List[str]] = defaultdict(list)
        for name, docs, weight in (("dense", dense, cfg.dense_weight), ("sparse", sparse, cfg.sparse_weight)):
            for rank, doc in enumerate(docs, start=1):
                child_id = doc.metadata.get("child_id")
                parent_id = doc.metadata.get("parent_id")
                if not child_id or not parent_id:
                    continue  # 스키마가 다른(구버전) 청크 방어
                fused[child_id] += weight / (cfg.rrf_k + rank)
                child_parent[child_id] = parent_id
                if name not in child_hits[child_id]:
                    child_hits[child_id].append(name)

        if not fused:
            return []

        # 3) Parent로 승격 — 소속 Child 중 최고 점수를 Parent 점수로 사용
        #    (합산을 쓰면 표가 20조각으로 잘린 Parent가 Child 수만으로 상위를 독식합니다)
        parent_score: Dict[str, float] = {}
        parent_children: Dict[str, List[str]] = defaultdict(list)
        parent_retrievers: Dict[str, List[str]] = defaultdict(list)
        for child_id, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            pid = child_parent[child_id]
            parent_score[pid] = max(parent_score.get(pid, 0.0), score)
            parent_children[pid].append(child_id)
            for name in child_hits[child_id]:
                if name not in parent_retrievers[pid]:
                    parent_retrievers[pid].append(name)

        ranked_ids = sorted(parent_score, key=lambda pid: parent_score[pid], reverse=True)
        # 재순위기가 있으면 후보를 넉넉히(2배) 가져와 재정렬 후 top_n만 남깁니다.
        n_candidates = cfg.top_n * 2 if self.reranker is not None else cfg.top_n
        candidate_ids = ranked_ids[:n_candidates]

        # 4) Docstore에서 Parent 본문 조회
        results: List[RetrievedParent] = []
        for pid, doc in zip(candidate_ids, self.store.get_parents(candidate_ids)):
            if doc is None:
                # Chroma와 Docstore가 어긋난 상태(부분 실패 후 수동 삭제 등). 조용히 넘기면
                # 원인 추적이 불가능하므로 경고를 남기고 건너뜁니다.
                logger.warning("Parent가 Docstore에 없습니다: %s (재인덱싱 권장)", pid)
                continue
            results.append(
                RetrievedParent(
                    parent_id=pid,
                    document=doc,
                    score=parent_score[pid],
                    matched_child_ids=parent_children[pid],
                    retrievers=parent_retrievers[pid],
                )
            )

        # 5) (선택) Cross-Encoder 재순위
        if self.reranker is not None and results:
            try:
                scores = self.reranker.score(query, [r.document.page_content for r in results])
                for r, s in zip(results, scores):
                    r.score = s
                results.sort(key=lambda r: r.score, reverse=True)
            except Exception as exc:  # noqa: BLE001 — 재순위 실패 시 RRF 순서 그대로 사용
                logger.warning("재순위 실패 → RRF 순서 유지: %s", exc)

        return results[: cfg.top_n]

    @staticmethod
    def _safe_search(name: str, fn: Any, query: str, k: int, where: Optional[Dict[str, Any]]):
        try:
            return fn(query, k, where), None
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s 검색 실패 → 다른 검색기 결과만 사용: %s", name, exc)
            return [], f"{type(exc).__name__}: {exc}"
