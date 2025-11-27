"""
📘 Step 2: 임베딩 및 Vector DB 저장 시스템 (개선 버전)

기능 요약
- 약관 JSON(페이지 단위)을 로드하여 청크 단위로 분할
- OpenAI Embeddings(text-embedding-3-large)로 임베딩 계산
- ChromaDB PersistentClient에 벡터 + 메타데이터 저장
- 쿼리 시 동일 임베딩 모델로 검색하여 상위 문서 반환

주요 클래스
- VectorStore: 로딩, 청크 분할, 벡터 저장, 검색 전부 담당
"""

import os
import json
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


# .env 로딩 (OPENAI_API_KEY 등)
load_dotenv()


class VectorStore:
    """
    보험 약관용 Vector DB 래퍼 클래스.

    사용 순서 예시:
        vs = VectorStore()
        pages = vs.load_pages_from_json("processed_data/약관_pages.json")
        chunks = vs.process_all_pages(pages)
        vs.store_in_vector_db(chunks, reset=True)
        results = vs.search_similar("계약 해지하면 환급금 얼마나 나와?")
    """

    def __init__(
        self,
        collection_name: str = "insurance_terms",
        persist_dir: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-large",
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY 가 설정되어 있지 않습니다 (.env 확인).")

        # Chroma Persistent Client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # LangChain OpenAI 임베딩 (직접 embeddings 인자로 넘길 예정)
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )

        # 컬렉션 생성 또는 로드
        self.collection = self._get_or_create_collection()
        print(f"✅ VectorStore 초기화 완료 (collection='{self.collection_name}')")

    # --------------------------------------------------------------------- #
    # 내부 유틸
    # --------------------------------------------------------------------- #

    def _get_or_create_collection(self):
        """
        컬렉션이 존재하면 로드, 없으면 생성.
        embedding_function 은 사용하지 않고, 항상 직접 embeddings 를 넘긴다.
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            print(f"📂 기존 컬렉션 로드: {self.collection_name}")
            return collection
        except Exception:
            print(f"🆕 새 컬렉션 생성: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "보험 약관 문서 벡터 저장소"},
            )

    @staticmethod
    def _safe_basename(path: str) -> str:
        """파일 경로에서 확장자 제거한 안전한 basename 리턴."""
        base = os.path.basename(path)
        return os.path.splitext(base)[0]

    # --------------------------------------------------------------------- #
    # 1) JSON 로딩
    # --------------------------------------------------------------------- #

    def load_pages_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        약관 JSON 파일을 로드한다.
        구조 예시:
            [
              {"page": 1, "text": "...", "source": "약관.pdf"},
              {"page": 2, "text": "...", "source": "약관.pdf"},
              ...
            ]
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 간단 검증
        if not isinstance(data, list):
            raise ValueError("❌ JSON 최상위 구조는 list 여야 합니다.")

        print(f"📄 JSON 페이지 로드 완료: {len(data)} pages from '{json_path}'")
        return data

    # --------------------------------------------------------------------- #
    # 2) 페이지 → 청크
    # --------------------------------------------------------------------- #

    def process_all_pages(
        self,
        pages: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        모든 페이지 텍스트를 청크로 나누고 메타데이터를 붙여 반환한다.

        반환 형식:
            [
              {
                "id": "약관_p1_c0",
                "text": "청크 내용...",
                "metadata": {
                    "page": 1,
                    "source": "약관.pdf",
                    "chunk_id": 0,
                    "total_chunks": 3,
                }
              },
              ...
            ]
        """
        if not pages:
            print("⚠️ process_all_pages: 입력 페이지가 비어 있습니다.")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        all_chunks: List[Dict[str, Any]] = []

        for page_entry in pages:
            page_num = page_entry.get("page")
            text = (page_entry.get("text") or "").strip()
            source = page_entry.get("source") or "unknown"

            if not text:
                continue

            chunks = splitter.split_text(text)
            total_chunks = len(chunks)
            base = self._safe_basename(source)

            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{base}_p{page_num}_c{idx}"

                metadata = {
                    "page": page_num,
                    "source": source,
                    "chunk_id": idx,
                    "total_chunks": total_chunks,
                }

                all_chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": metadata,
                    }
                )

        print(f"✂️ 텍스트 청킹 완료: 총 {len(all_chunks)} chunks 생성")
        return all_chunks

    # --------------------------------------------------------------------- #
    # 3) Vector DB 저장
    # --------------------------------------------------------------------- #

    def store_in_vector_db(
        self,
        chunks: List[Dict[str, Any]],
        reset: bool = False,
        batch_size: int = 50,
    ) -> None:
        """
        청크 리스트를 임베딩 계산 후 ChromaDB 컬렉션에 저장한다.

        - reset=True 이면 기존 컬렉션을 삭제하고 새로 생성
        - batch_size 단위로 나누어 embeddings + add 수행
        """
        if not chunks:
            print("⚠️ 저장할 청크가 없습니다.")
            return

        # 필요하면 기존 데이터 삭제 후 컬렉션 재생성
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                print("🗑️ 기존 컬렉션 삭제 완료")
            except Exception as e:
                print(f"⚠️ 기존 컬렉션 삭제 실패 (무시함): {e}")
            finally:
                self.collection = self._get_or_create_collection()

        total = len(chunks)
        print(f"💾 벡터 저장 시작: 총 {total} chunks (batch_size={batch_size})")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = chunks[start:end]

            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            ids = [c["id"] for c in batch]

            try:
                # 1) 임베딩 계산
                vectors = self.embeddings.embed_documents(texts)

                # 2) 컬렉션에 저장
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=vectors,
                )

                print(f"  🔹 저장 완료: {start} ~ {end - 1} (누적 {end}/{total})")
            except Exception as e:
                print(f"❌ batch {start}~{end} 저장 중 오류: {e}")

        print("✅ 모든 청크 벡터 저장 완료!")

    # --------------------------------------------------------------------- #
    # 4) 검색 (RAG에서 직접 사용)
    # --------------------------------------------------------------------- #

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        쿼리 문장을 임베딩 후, 가장 유사한 청크 top_k개를 반환한다.

        반환 형식:
            [
              {
                "text": "...",
                "page": 12,
                "source": "약관.pdf",
                "score": 0.87,  # 0~1 (1에 가까울수록 유사)
                "metadata": {...}
              },
              ...
            ]
        """
        if not query.strip():
            return []

        # 쿼리 임베딩 계산
        query_vec = self.embeddings.embed_query(query)

        try:
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"❌ search_similar 쿼리 오류: {e}")
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not docs:
            print("⚠️ 검색 결과가 없습니다.")
            return []

        # 거리(distance)를 0~1 사이의 점수(score)로 정규화 (작을수록 유사)
        max_dist = max(dists) if dists else 1.0
        if max_dist == 0:
            max_dist = 1.0

        scored_items = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 - (dist / max_dist)
            scored_items.append(
                {
                    "text": doc,
                    "page": meta.get("page"),
                    "source": meta.get("source"),
                    "score": float(score),
                    "metadata": meta,
                }
            )

        # score 기준 내림차순 정렬
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items

    # --------------------------------------------------------------------- #
    # 5) 디버그/정보 함수
    # --------------------------------------------------------------------- #

    def get_collection_info(self) -> int:
        """
        컬렉션에 저장된 문서(청크) 개수를 반환하고,
        간단한 요약 로그를 출력한다.
        """
        info = self.collection.get()
        num = len(info.get("ids", []))
        print(f"📊 컬렉션 '{self.collection_name}' 문서 수: {num}")
        return num


# ------------------------------------------------------------------------- #
#  단독 실행용 main (테스트 용도)
# ------------------------------------------------------------------------- #

def main():
    """
    python vector_store.py 를 직접 실행했을 때:
    - processed_data/약관_pages.json 을 읽어서
    - 청크 생성 후
    - 벡터 DB를 reset 하고 다시 빌드
    """
    json_file = "processed_data/약관_pages.json"

    vs = VectorStore(
        collection_name="insurance_terms",
        persist_dir="./chroma_db",
        embedding_model="text-embedding-3-large",
    )

    pages = vs.load_pages_from_json(json_file)
    chunks = vs.process_all_pages(pages)

    if not chunks:
        print("⚠️ 생성된 청크가 없어 벡터 저장을 중단합니다.")
        return

    vs.store_in_vector_db(chunks, reset=True)
    vs.get_collection_info()

    print("\n🎉 벡터 저장소 구축 완료!")
    print(f"📁 저장 위치: {vs.persist_dir}")
    print(f"📚 컬렉션: {vs.collection_name}")


if __name__ == "__main__":
    main()
