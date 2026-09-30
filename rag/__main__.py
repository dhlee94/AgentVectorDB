"""CLI: `python -m rag ingest <folder>` / `python -m rag query "<질문>"`."""

from __future__ import annotations

import argparse
import logging
import sys

from rag.config import load_config
from rag.exceptions import RAGError
from rag.pipeline import RAGPipeline


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag", description="엔터프라이즈 RAG 파이프라인")
    parser.add_argument("--config", help="설정 파일 경로 (기본: 환경변수 RAG_CONFIG → ./config.yaml)")
    parser.add_argument("--store", help="인덱스 저장 디렉터리 (설정 파일의 store.persist_dir보다 우선)")
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO 로그 출력")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="폴더의 문서를 인덱스와 동기화")
    p_ingest.add_argument("folder")

    p_query = sub.add_parser("query", help="질문하기")
    p_query.add_argument("question")
    p_query.add_argument("--source", help="특정 파일명으로 검색 범위 제한")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
        if args.store:
            config.store.persist_dir = args.store
        pipeline = RAGPipeline(config)
        if args.command == "ingest":
            report = pipeline.ingest(args.folder)
    except RAGError as exc:
        # 설정 오류, 의존성 누락, 폴더 없음 등 사용자가 고칠 수 있는 오류는 traceback 대신 안내만 출력
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    if args.command == "ingest":
        print(report.summary())
        for failure in report.failures:
            print(f"  실패: {failure.source_path} — {failure.error_type}: {failure.reason}")
        return 1 if report.failures and not report.indexed and not report.unchanged else 0

    where = {"source": args.source} if args.source else None
    answer = pipeline.query(args.question, where=where)
    print(answer.format())
    return 0


if __name__ == "__main__":
    sys.exit(main())
