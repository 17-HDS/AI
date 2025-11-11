"""
대화 기억 관리를 위한 벡터 데이터베이스 클래스
ChromaDB를 사용하여 대화 기록을 임베딩으로 저장하고 검색
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

class ConversationMemory:
    """대화 기억 관리 클래스"""
    
    def __init__(self, memory_collection_name: str = "conversation_memory"):
        """
        대화 기억 관리자 초기화
        
        Args:
            memory_collection_name: 기억 저장소 컬렉션 이름
        """
        self.memory_collection_name = memory_collection_name
        self.client = None
        self.collection = None
        
        # BGE-M3 임베딩 함수 초기화
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3"
        )
        
        # ChromaDB 초기화
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """ChromaDB 초기화"""
        try:
            # 기존 클라이언트가 있는지 확인하고 재사용
            try:
                # 기존 클라이언트 재사용 시도
                self.client = chromadb.PersistentClient(path="./chroma_db")
                self.collection = self.client.get_collection(name=self.memory_collection_name)
                print(f"기존 클라이언트로 기억 컬렉션 '{self.memory_collection_name}' 로드됨")
                return
            except:
                pass
            
            # 새 클라이언트 생성
            self.client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 기억 컬렉션 가져오기 또는 생성
            try:
                self.collection = self.client.get_collection(
                    name=self.memory_collection_name,
                    embedding_function=self.embedding_function
                )
                print(f"기존 기억 컬렉션 '{self.memory_collection_name}' 로드됨")
            except:
                self.collection = self.client.create_collection(
                    name=self.memory_collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "대화 기억 저장소", "embedding_model": "BAAI/bge-m3"}
                )
                print(f"새 기억 컬렉션 '{self.memory_collection_name}' 생성됨 (임베딩 모델: BGE-M3)")
                
        except Exception as e:
            print(f"ChromaDB 초기화 오류: {str(e)}")
            # 최후의 수단: 기본 설정으로 시도
            try:
                self.client = chromadb.PersistentClient(path="./chroma_db")
                self.collection = self.client.get_collection(
                    name=self.memory_collection_name,
                    embedding_function=self.embedding_function
                )
                print(f"기본 설정으로 기억 컬렉션 '{self.memory_collection_name}' 로드됨")
            except Exception as e2:
                print(f"모든 초기화 시도 실패: {str(e2)}")
                # 기억 시스템 없이 계속 진행
                self.client = None
                self.collection = None
                print("⚠️ 기억 시스템을 비활성화하고 계속 진행합니다.")
    
    def _get_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩으로 변환 (BGE-M3 사용)"""
        try:
            # BGE-M3 임베딩 함수를 사용하여 임베딩 생성
            embedding = self.embedding_function([text])
            return embedding[0]
        except Exception as e:
            print(f"임베딩 생성 오류: {str(e)}")
            raise
    
    def save_conversation(self, user_query: str, ai_response: str, 
                         sources: List[Dict] = None, 
                         additional_context: str = "") -> str:
        """
        대화를 기억 저장소에 저장
        
        Args:
            user_query: 사용자 질문
            ai_response: AI 응답
            sources: 참고 문서 (선택사항)
            additional_context: 추가 컨텍스트 (선택사항)
            
        Returns:
            저장된 기억의 고유 ID
        """
        if not self.collection:
            print("⚠️ 기억 시스템이 비활성화되어 있습니다.")
            return ""
        
        try:
            # 기억 ID 생성
            memory_id = str(uuid.uuid4())
            
            # 대화 내용을 하나의 텍스트로 결합
            conversation_text = f"""
            사용자: {user_query}
            AI: {ai_response}
            """
            
            if additional_context:
                conversation_text += f"\n컨텍스트: {additional_context}"
            
            if sources:
                sources_text = "참고 문서:\n" + "\n".join([
                    f"- 페이지 {s['page']}: {s['content'][:200]}..." 
                    for s in sources[:3]  # 최대 3개 문서만
                ])
                conversation_text += f"\n{sources_text}"
            
            # 임베딩 생성
            embedding = self._get_embedding(conversation_text)
            
            # 메타데이터 구성
            metadata = {
                "user_query": user_query,
                "ai_response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "memory_type": "conversation",
                "query_length": len(user_query),
                "response_length": len(ai_response)
            }
            
            if sources:
                metadata["source_count"] = len(sources)
                metadata["source_pages"] = [s['page'] for s in sources]
            
            if additional_context:
                metadata["has_context"] = True
            
            # ChromaDB에 저장
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[conversation_text],
                metadatas=[metadata]
            )
            
            print(f"대화 기억 저장 완료: {memory_id}")
            return memory_id
            
        except Exception as e:
            print(f"대화 기억 저장 오류: {str(e)}")
            raise
    
    def search_similar_conversations(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        유사한 대화를 검색
        
        Args:
            query: 검색할 질문
            n_results: 반환할 결과 수
            
        Returns:
            유사한 대화 목록
        """
        if not self.collection:
            print("⚠️ 기억 시스템이 비활성화되어 있습니다.")
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self._get_embedding(query)
            
            # 유사도 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 정리
            similar_conversations = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similar_conversations.append({
                        "id": results['ids'][0][i] if results['ids'] and results['ids'][0] else None,
                        "document": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,  # 거리를 유사도로 변환
                        "user_query": metadata.get("user_query", ""),
                        "ai_response": metadata.get("ai_response", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "source_count": metadata.get("source_count", 0)
                    })
            
            return similar_conversations
            
        except Exception as e:
            print(f"대화 검색 오류: {str(e)}")
            return []
    
    def get_conversation_context(self, current_query: str, max_context_length: int = 2000) -> str:
        """
        현재 질문과 관련된 이전 대화들을 컨텍스트로 반환
        
        Args:
            current_query: 현재 질문
            max_context_length: 최대 컨텍스트 길이
            
        Returns:
            컨텍스트 문자열
        """
        if not self.collection:
            return ""
        
        try:
            # 유사한 대화 검색
            similar_conversations = self.search_similar_conversations(current_query, n_results=3)
            
            if not similar_conversations:
                return ""
            
            # 컨텍스트 구성
            context_parts = []
            current_length = 0
            
            for conv in similar_conversations:
                # 이전 대화 정보 추가
                context_part = f"""
                [이전 대화 - 유사도: {conv['similarity_score']:.2f}]
                질문: {conv['user_query']}
                답변: {conv['ai_response'][:300]}...
                """
                
                if current_length + len(context_part) > max_context_length:
                    break
                
                context_parts.append(context_part)
                current_length += len(context_part)
            
            if context_parts:
                full_context = "=== 관련 이전 대화 ===\n" + "\n".join(context_parts)
                return full_context
            
            return ""
            
        except Exception as e:
            print(f"컨텍스트 생성 오류: {str(e)}")
            return ""
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """기억 저장소 통계 정보 반환"""
        if not self.collection:
            return {"total_memories": 0, "error": "기억 시스템이 비활성화되어 있습니다."}
        
        try:
            count = self.collection.count()
            
            # 최근 기억들 가져오기
            recent_memories = self.collection.get(limit=10)
            
            stats = {
                "total_memories": count,
                "collection_name": self.memory_collection_name,
                "recent_memories": []
            }
            
            if recent_memories and recent_memories['metadatas']:
                for metadata in recent_memories['metadatas'][:5]:
                    stats["recent_memories"].append({
                        "timestamp": metadata.get("timestamp", ""),
                        "user_query": metadata.get("user_query", "")[:50] + "...",
                        "memory_type": metadata.get("memory_type", "")
                    })
            
            return stats
            
        except Exception as e:
            print(f"통계 조회 오류: {str(e)}")
            return {"total_memories": 0, "error": str(e)}
    
    def clear_all_memories(self) -> bool:
        """모든 기억 삭제"""
        try:
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(name=self.memory_collection_name)
            self.collection = self.client.create_collection(
                name=self.memory_collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "대화 기억 저장소", "embedding_model": "BAAI/bge-m3"}
            )
            print("모든 기억이 삭제되었습니다.")
            return True
            
        except Exception as e:
            print(f"기억 삭제 오류: {str(e)}")
            return False
    
    def export_memories(self, output_file: str = None) -> str:
        """기억을 JSON 파일로 내보내기"""
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"conversation_memories_{timestamp}.json"
            
            # 모든 기억 가져오기
            all_memories = self.collection.get(
                include=["documents", "metadatas"]
            )
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_memories": len(all_memories.get('ids', [])),
                "collection_name": self.memory_collection_name,
                "memories": []
            }
            
            if all_memories and all_memories['ids']:
                for i, memory_id in enumerate(all_memories['ids']):
                    memory_data = {
                        "id": memory_id,
                        "document": all_memories['documents'][i],
                        "metadata": all_memories['metadatas'][i]
                    }
                    export_data["memories"].append(memory_data)
            
            # JSON 파일로 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"기억 내보내기 완료: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"기억 내보내기 오류: {str(e)}")
            return ""

# 테스트 함수
def test_memory_system():
    """기억 시스템 테스트"""
    try:
        memory = ConversationMemory()
        
        # 테스트 대화 저장
        memory_id = memory.save_conversation(
            user_query="보험금 지급 조건은 무엇인가요?",
            ai_response="보험금 지급 조건은 다음과 같습니다: 1) 진단서 제출, 2) 치료비 영수증, 3) 보험약관 준수 등이 필요합니다.",
            additional_context="테스트 대화입니다."
        )
        
        print(f"테스트 대화 저장됨: {memory_id}")
        
        # 유사 대화 검색 테스트
        similar = memory.search_similar_conversations("보험금은 어떻게 받나요?")
        print(f"유사 대화 검색 결과: {len(similar)}개")
        
        # 통계 조회
        stats = memory.get_memory_stats()
        print(f"기억 저장소 통계: {stats}")
        
    except Exception as e:
        print(f"테스트 오류: {str(e)}")

if __name__ == "__main__":
    test_memory_system()
