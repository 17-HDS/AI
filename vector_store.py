"""
📘 Step 2: 임베딩 및 Vector DB 저장 시스템 (개선 버전)
- ID 중복 방지
- ChromaDB 영구 저장 (persist)
- 컬렉션 초기화 제어 가능
- 검색 정확도 향상 (거리 정규화 적용)
"""

import json
import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


class VectorStore:
    def __init__(self, collection_name: str = "insurance_terms"):
        self.collection_name = collection_name

        # OpenAI API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key
        )

        # ChromaDB 클라이언트 초기화 (Persistent)
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )

        # 컬렉션 생성 또는 로드
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"📚 기존 컬렉션 로드: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "보험 약관 문서 벡터 저장소"}
            )
            print(f"📚 새 컬렉션 생성: {collection_name}")

    def load_processed_data(self, json_file: str) -> List[Dict]:
        """처리된 JSON 데이터 로드"""
        print(f"📖 데이터 로드 중: {json_file}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"✅ {len(data)}페이지 데이터 로드 완료")
            return data
        except Exception as e:
            print(f"❌ 데이터 로드 오류: {str(e)}")
            return []

    def chunk_text(self, text: str, page: int, source: str) -> List[Dict]:
        """텍스트를 청크로 분할"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_text(text)
        chunk_data = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                chunk_data.append(
                    {
                        "content": chunk.strip(),
                        "metadata": {
                            "page": page,
                            "source": source,
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                        },
                    }
                )
        return chunk_data

    def process_all_pages(self, pages_data: List[Dict]) -> List[Dict]:
        """모든 페이지를 청크로 분할"""
        print("✂️ 텍스트 청킹 중...")
        all_chunks = []
        for page_data in pages_data:
            page_chunks = self.chunk_text(
                page_data["text"], page_data["page"], page_data["source"]
            )
            all_chunks.extend(page_chunks)
            print(f"   ✅ 페이지 {page_data['page']}: {len(page_chunks)}개 청크")
        print(f"✂️ 총 {len(all_chunks)}개 청크 생성 완료")
        return all_chunks

    def store_in_vector_db(self, chunks: List[Dict], reset: bool = True):
        """청크들을 벡터 DB에 저장"""
        print("💾 벡터 DB에 저장 중...")

        try:
            # 필요 시 기존 데이터 삭제
            if reset:
                try:
                    self.client.delete_collection(self.collection_name)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "보험 약관 문서 벡터 저장소"},
                    )
                    print("🗑️ 기존 데이터 삭제 후 새 컬렉션 생성 완료")
                except Exception as e:
                    print(f"⚠️ 기존 컬렉션 삭제 실패 (무시됨): {e}")

            # 청크를 배치 단위로 저장
            batch_size = 50  # ✅ 안정성 향상
            global_counter = 0
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                documents = [chunk["content"] for chunk in batch]
                metadatas = [chunk["metadata"] for chunk in batch]
                ids = [f"chunk_{global_counter + k}" for k in range(len(batch))]
                global_counter += len(batch)

                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                print(f"   ✅ 배치 {i // batch_size + 1} 저장 완료 ({len(batch)}개 청크)")

            print("💾 ChromaDB 영구 저장 완료")
            print(f"💾 총 {len(chunks)}개 청크 저장 완료")

        except Exception as e:
            print(f"❌ 벡터 DB 저장 오류: {str(e)}")

    def search_similar(self, query: str, top_k: int = 10) -> List[Dict]:
        """유사한 문서 검색 (distance 정규화 포함)"""
        try:
            results = self.collection.query(query_texts=[query], n_results=top_k)
            search_results = []
            if not results or not results["documents"]:
                print("⚠️ 검색 결과가 없습니다.")
                return []

            distances = results["distances"][0]
            # ✅ 거리 → 유사도로 변환 (정규화)
            max_d = max(distances)
            min_d = min(distances)
            norm_sim = [(max_d - d) / (max_d - min_d + 1e-9) for d in distances]

            for i, doc in enumerate(results["documents"][0]):
                search_results.append(
                    {
                        "content": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": distances[i],
                        "similarity": round(norm_sim[i], 4),
                    }
                )

            return sorted(search_results, key=lambda x: x["similarity"], reverse=True)

        except Exception as e:
            print(f"❌ 검색 오류: {str(e)}")
            return []

    def get_collection_info(self):
        """컬렉션 정보 조회"""
        try:
            count = self.collection.count()
            print("📊 컬렉션 정보:")
            print(f"   - 이름: {self.collection_name}")
            print(f"   - 총 문서 수: {count}")
            return count
        except Exception as e:
            print(f"❌ 컬렉션 정보 조회 오류: {str(e)}")
            return 0


def main():
    """메인 프로그램"""
    print("📘 벡터 저장소 구축 시스템")
    print("=" * 60)

    json_file = "processed_data/약관_pages.json"
    if not os.path.exists(json_file):
        print(f"❌ 처리된 데이터 파일이 없습니다: {json_file}")
        print("💡 먼저 pdf_preprocessor.py를 실행하세요.")
        return

    vector_manager = VectorStore()
    pages_data = vector_manager.load_processed_data(json_file)
    if not pages_data:
        return

    chunks = vector_manager.process_all_pages(pages_data)
    if not chunks:
        return

    vector_manager.store_in_vector_db(chunks, reset=True)
    vector_manager.get_collection_info()

    print("\n🎉 벡터 저장소 구축 완료!")
    print("📁 저장 위치: ./chroma_db")
    print(f"📚 컬렉션: {vector_manager.collection_name}")


if __name__ == "__main__":
    main()
