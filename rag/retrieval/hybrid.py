"""HybridParentRetriever: Dense + BM25 → RRF(Child 단위) → Parent 승격 (전략 B + C).

흐름
    질문 ─┬─ Dense (Chroma, bge-m3) → Child 후보 dense_k개 (순위 r_d)
          └─ Sparse(BM25, Kiwi)     → Child 후보 sparse_k개 (순위 r_s)
    RRF(child) = w_d/(rrf_k + r_d) + w_s/(rrf_k + r_s)       ← 한쪽에만 있으면 그쪽 항만
    (선택) RRF 상위 Child를 Cross-Encoder로 재채점
    Parent 점수 = 소속 Child들의 점수 최댓값 → 상위 top_n개 Parent를 Docstore에서 조회

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
    """Cross-Encoder 재순위기. 질문과 청크를 한 쌍으로 모델에 넣어 관련도를 직접 계산합니다.

    임베딩(Bi-Encoder)은 질문과 문서를 따로 벡터화해 비교하므로 빠르지만 거칩니다.
    Cross-Encoder는 둘을 함께 읽어 "이 청크가 이 질문에 답하는가"를 판단하므로 훨씬 정확하지만,
    후보마다 모델 추론이 필요해 느립니다. 그래서 RRF로 좁힌 후보에만 적용합니다.
    """

    # 같은 쌍의 단건/배치 점수 차이 허용치 (점수는 0~1 sigmoid)
    _SELF_TEST_TOLERANCE = 0.01
    _SELF_TEST_PAIRS = [
        ("볼펜 할인율은?", "[행5] 품목: 볼펜, 할인율: 10%"),
        ("보안 위협", "프롬프트 인젝션 방어 레이어"),
        ("짧음", "재택근무는 주 2회까지 허용된다"),
        ("Ralph Loop", "컨텍스트 불안을 극복하는 방법"),
    ]

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("재순위에는 `pip install sentence-transformers`가 필요합니다") from exc

        self._model = CrossEncoder(model_name, device=device, max_length=max_length)
        worst = self.self_test()
        current = str(getattr(self._model, "device", device or "?"))
        if worst > self._SELF_TEST_TOLERANCE:
            if current.startswith("cpu"):
                raise RuntimeError(f"재순위기 자기검사 실패: CPU에서도 단건/배치 점수가 다릅니다 (최대 차이 {worst:.3f})")
            # 임베딩과 같은 이유: 일부 GPU 백엔드(Apple MPS 등)에서 입력 1건 추론이 틀리는 사례가 있음
            logger.warning("재순위기 자기검사 실패 (device=%s, 최대 차이 %.3f) → CPU로 전환", current, worst)
            self._model = CrossEncoder(model_name, device="cpu", max_length=max_length)
            if self.self_test() > self._SELF_TEST_TOLERANCE:
                raise RuntimeError("재순위기 자기검사 실패: CPU 전환 후에도 불일치")
        logger.info("재순위기 준비 완료: %s (device=%s)", model_name, getattr(self._model, "device", "?"))

    def self_test(self) -> float:
        """같은 (질문, 문서) 쌍을 배치로 채점한 값과 1건씩 채점한 값의 최대 차이."""
        batch = self._model.predict(self._SELF_TEST_PAIRS)
        return max(abs(float(self._model.predict([pair])[0]) - float(b)) for pair, b in zip(self._SELF_TEST_PAIRS, batch))

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
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
        child_docs: Dict[str, Document] = {}
        child_hits: Dict[str, List[str]] = defaultdict(list)
        for name, docs, weight in (("dense", dense, cfg.dense_weight), ("sparse", sparse, cfg.sparse_weight)):
            for rank, doc in enumerate(docs, start=1):
                child_id = doc.metadata.get("child_id")
                if not child_id or not doc.metadata.get("parent_id"):
                    continue  # 스키마가 다른(구버전) 청크 방어
                fused[child_id] += weight / (cfg.rrf_k + rank)
                child_docs[child_id] = doc
                if name not in child_hits[child_id]:
                    child_hits[child_id].append(name)

        if not fused:
            return []

        # 3) (선택) Child 단위 Cross-Encoder 재순위
        #    Parent(최대 3,000자)가 아니라 Child(최대 500자)를 채점합니다. Parent를 넣으면 재순위기
        #    입력 한도(512토큰)에서 뒷부분이 잘려, 정작 답이 있는 부분을 못 보고 점수를 매기게 됩니다.
        #    검색 단위(Child)끼리 경쟁시키고 승자를 Parent로 올리는 Small-to-Big 원칙과도 일치합니다.
        child_score: Dict[str, float] = dict(fused)
        reranked = False
        if self.reranker is not None:
            candidates = sorted(fused, key=lambda cid: fused[cid], reverse=True)[: cfg.rerank_candidates]
            try:
                scores = self.reranker.score(query, [child_docs[cid].page_content for cid in candidates])
                # 후보 밖 Child는 버립니다: RRF 하위권이 재순위 점수 없이 섞이면 비교가 불가능
                child_score = dict(zip(candidates, scores))
                for cid in candidates:
                    child_hits[cid].append("rerank")
                reranked = True
            except Exception as exc:  # noqa: BLE001 — 재순위 실패 시 RRF 점수 그대로 사용
                logger.warning("재순위 실패 → RRF 순서 유지: %s", exc)

        # 4) Parent로 승격 — 소속 Child 중 최고 점수를 Parent 점수로 사용
        #    (합산을 쓰면 표가 20조각으로 잘린 Parent가 Child 수만으로 상위를 독식합니다)
        parent_score: Dict[str, float] = {}
        parent_children: Dict[str, List[str]] = defaultdict(list)
        parent_retrievers: Dict[str, List[str]] = defaultdict(list)
        for child_id, score in sorted(child_score.items(), key=lambda kv: kv[1], reverse=True):
            pid = child_docs[child_id].metadata["parent_id"]
            parent_score[pid] = max(parent_score.get(pid, float("-inf")), score)
            parent_children[pid].append(child_id)
            for name in child_hits[child_id]:
                if name not in parent_retrievers[pid]:
                    parent_retrievers[pid].append(name)

        top_ids = sorted(parent_score, key=lambda pid: parent_score[pid], reverse=True)[: cfg.top_n]
        if reranked and cfg.rerank_min_score > 0:
            # 재순위 점수는 0~1 절대 척도라 "관련 없음"을 판단할 수 있습니다 (RRF 점수는 상대 순위라 불가).
            # 잡음 문서를 LLM에 넘기면 토큰만 쓰고 출처 혼동을 일으키므로 걸러냅니다.
            kept = [pid for pid in top_ids if parent_score[pid] >= cfg.rerank_min_score]
            if len(kept) < len(top_ids):
                logger.info("재순위 점수 %.2f 미만 Parent %d개 제외", cfg.rerank_min_score, len(top_ids) - len(kept))
            top_ids = kept

        # 5) Docstore에서 Parent 본문 조회
        results: List[RetrievedParent] = []
        for pid, doc in zip(top_ids, self.store.get_parents(top_ids)):
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
        return results

    @staticmethod
    def _safe_search(name: str, fn: Any, query: str, k: int, where: Optional[Dict[str, Any]]):
        try:
            return fn(query, k, where), None
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s 검색 실패 → 다른 검색기 결과만 사용: %s", name, exc)
            return [], f"{type(exc).__name__}: {exc}"
