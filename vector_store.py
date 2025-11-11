"""
📘 Step 2: 임베딩 및 Vector DB 저장 시스템
페이지별 텍스트를 문단 단위로 쪼개어 ChromaDB에 저장
"""

import json
import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str = "insurance_terms"):
        self.collection_name = collection_name
        
        # BGE-M3 임베딩 함수 초기화
        # BGE-M3는 멀티 언어 지원 및 높은 성능의 오픈소스 임베딩 모델
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3"
        )
        print("BGE-M3 임베딩 모델 로드 완료")
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 기존 컬렉션 삭제 후 새로 생성
        try:
            self.client.delete_collection(name=collection_name)
            print(f"기존 컬렉션 삭제: {collection_name}")
        except:
            pass
        
        # 새 컬렉션 생성 (BGE-M3 임베딩 함수 지정)
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "보험 약관 문서 벡터 저장소", "embedding_model": "BAAI/bge-m3"}
        )
        print(f"새 컬렉션 생성: {collection_name} (임베딩 모델: BGE-M3)")
    
    def load_processed_data(self, json_file: str) -> List[Dict]:
        """처리된 JSON 데이터 로드"""
        print(f"데이터 로드 중: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"{len(data)}페이지 데이터 로드 완료")
            return data
            
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            return []
    
    def chunk_text(self, text: str, page: int, source: str) -> List[Dict]:
        """텍스트를 청크로 분할"""
        # 텍스트 분할기 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # 300 토큰으로 줄임
            chunk_overlap=100,  # 100 토큰 오버랩으로 늘림
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # 텍스트 분할
        chunks = text_splitter.split_text(text)
        
        # 청크 데이터 생성
        chunk_data = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # 빈 청크 제외
                chunk_data.append({
                    "content": chunk.strip(),
                    "metadata": {
                        "page": page,
                        "source": source,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                })
        
        return chunk_data
    
    def process_all_pages(self, pages_data: List[Dict]) -> List[Dict]:
        """모든 페이지를 청크로 분할"""
        print("텍스트 청킹 중...")
        
        all_chunks = []
        for page_data in pages_data:
            page_chunks = self.chunk_text(
                page_data["text"],
                page_data["page"],
                page_data["source"]
            )
            all_chunks.extend(page_chunks)
            
            print(f"   페이지 {page_data['page']}: {len(page_chunks)}개 청크")
        
        print(f"총 {len(all_chunks)}개 청크 생성 완료")
        return all_chunks
    
    def store_in_vector_db(self, chunks: List[Dict]):
        """청크들을 벡터 DB에 저장"""
        print("벡터 DB에 저장 중...")
        
        try:
            # 기존 데이터 삭제 (새로 시작)
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "보험 약관 문서 벡터 저장소", "embedding_model": "BAAI/bge-m3"}
                )
                print("기존 데이터 삭제 완료")
            except:
                pass
            
            # 청크들을 배치로 저장
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # 데이터 준비
                documents = [chunk["content"] for chunk in batch]
                metadatas = [chunk["metadata"] for chunk in batch]
                ids = [f"chunk_{i}_{j}" for j in range(len(batch))]
                
                # 벡터 DB에 추가
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                print(f"   배치 {i//batch_size + 1} 저장 완료 ({len(batch)}개 청크)")
            
            print(f"총 {len(chunks)}개 청크 저장 완료")
            
        except Exception as e:
            print(f"벡터 DB 저장 오류: {str(e)}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """유사한 문서 검색"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # 결과 정리
            search_results = []
            for i in range(len(results['documents'][0])):
                search_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
            
            return search_results
            
        except Exception as e:
            print(f"검색 오류: {str(e)}")
            return []
    
    def get_collection_info(self):
        """컬렉션 정보 조회"""
        try:
            count = self.collection.count()
            print(f"컬렉션 정보:")
            print(f"   - 이름: {self.collection_name}")
            print(f"   - 총 문서 수: {count}")
            
            return count
            
        except Exception as e:
            print(f"컬렉션 정보 조회 오류: {str(e)}")
            return 0

def main():
    """메인 프로그램"""
    print("벡터 저장소 구축 시스템")
    print("=" * 60)
    
    # 1. 처리된 데이터 로드 (통합 파일 우선, 개별 파일 대체)
    json_files = [
        "processed_data/all_pdfs_pages.json",  # 통합 파일 우선
        "processed_data/약관_pages.json"        # 개별 파일 대체
    ]
    
    json_file = None
    for file_path in json_files:
        if os.path.exists(file_path):
            json_file = file_path
            break
    
    if not json_file:
        print(f"처리된 데이터 파일이 없습니다.")
        print("먼저 pdf_preprocessor.py를 실행하세요.")
        return
    
    # 2. 벡터 저장소 초기화
    vector_manager = VectorStore()
    
    # 3. 데이터 로드
    pages_data = vector_manager.load_processed_data(json_file)
    if not pages_data:
        return
    
    # 4. 텍스트 청킹
    chunks = vector_manager.process_all_pages(pages_data)
    if not chunks:
        return
    
    # 5. 벡터 DB에 저장
    vector_manager.store_in_vector_db(chunks)
    
    # 6. 저장소 정보 확인
    vector_manager.get_collection_info()
    
    print(f"\n벡터 저장소 구축 완료!")
    print(f"저장 위치: ./chroma_db")
    print(f"컬렉션: {vector_manager.collection_name}")

if __name__ == "__main__":
    main()
