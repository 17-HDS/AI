"""
📘 보험 약관 RAG 챗봇 (VectorStore + 강화 프롬프트 버전)
"""

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from vector_store import VectorStore

load_dotenv()


class RAGChatbot:
    """
    보험 약관 RAG 챗봇

    - VectorStore(ChromaDB)에서 유사 청크 검색
    - 검색된 약관 내용만을 근거로 OpenAI 모델이 답변 생성
    - Streamlit UI에서 chat_streaming() / get_collection_info() 사용
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",   # 현재 VectorStore는 내부에서 ./chroma_db 사용
        collection_name: str = "insurance_terms",
        model: str = "gpt-4o-mini",
        vector_store: Optional[VectorStore] = None,
    ):
        # VectorStore 초기화 (이미 구축된 DB만 사용)
        self.vector_store = vector_store or VectorStore(collection_name=collection_name)

        # OpenAI 클라이언트
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        self.client = OpenAI(api_key=api_key)

        self.model = model
        self.last_sources: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 📊 컬렉션 정보 (Streamlit에서 사용)
    # ------------------------------------------------------------------
    def get_collection_info(self) -> int:
        """
        컬렉션에 저장된 문서(청크) 개수 반환.
        Streamlit UI에서 metric으로 사용.
        """
        try:
            return self.vector_store.get_collection_info()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # 🔍 검색 + 컨텍스트 구성
    # ------------------------------------------------------------------
    def search_similar_docs(self, query: str, top_k: int = 5) -> str:
        """
        VectorStore에서 유사한 청크를 검색하고,
        LLM 프롬프트에 넣을 컨텍스트 문자열을 만든다.

        self.last_sources 에는 Streamlit에서 표시할 출처 정보를 저장한다.
        """
        results = self.vector_store.search_similar(query, top_k=top_k)

        if not results:
            self.last_sources = []
            return ""

        context_blocks = []
        sources: List[Dict[str, Any]] = []

        for item in results:
            # vector_store 버전별 호환 처리
            text = item.get("text") or item.get("content") or ""
            meta = item.get("metadata", {})

            page = item.get("page", meta.get("page", "?"))
            source_name = item.get("source", meta.get("source", "unknown"))

            # 유사도/점수 (optional)
            score = item.get("score", item.get("similarity", None))

            sources.append(
                {
                    "content": text,
                    "page": page,
                    "source": source_name,
                    "score": float(score) if score is not None else None,
                }
            )

            header = f"[page {page} / {source_name}]"
            context_blocks.append(f"{header}\n{text}")

        self.last_sources = sources
        context = "\n\n-----\n\n".join(context_blocks)
        return context

    # ------------------------------------------------------------------
    # 🧠 프롬프트 구성
    # ------------------------------------------------------------------
    def _build_messages(self, query: str, context: str):
        """
        시스템 / 유저 메시지 생성 (약관 전용 강화 프롬프트)
        """
        system_content = (
            "너는 사용자가 가진 '보험 약관 PDF'에서 발췌한 문장만을 근거로 답변하는 어시스턴트이다.\n\n"
            "📌 규칙\n"
            "1. 반드시 한국어로 답한다.\n"
            "2. 제공된 'Context(약관 발췌)' 안에서 직접적인 근거를 찾을 수 있을 때만 답한다.\n"
            "   - 근거가 불분명하거나 전혀 없으면 "
            "     '제공된 약관 범위에서 해당 내용을 찾을 수 없습니다.'라고 말하고 추측하지 않는다.\n"
            "3. 답변은 2~5개의 문단 또는 번호 목록으로 간결하게 정리한다.\n"
            "4. 금액, 지급 여부, 예외 사항을 말할 때는 "
            "   '약관상으로는 ~로 규정되어 있습니다.'처럼 표현하고, "
            "   실제 보상 여부는 보험사 심사와 세부 상품 조건에 따라 달라질 수 있음을 한 문장으로 덧붙인다.\n"
            "5. 가능하면 근거가 된 문장이나 조항에 대해 "
            "   '페이지 X, ○○조(또는 항)'처럼 페이지 정보를 언급한다. "
            "   조항 번호가 보이지 않으면 페이지 정보만 언급한다.\n"
            "6. 법률·세무·투자·의료 등의 일반적인 조언은 하지 말고, "
            "   약관 문구의 의미와 적용 가능성만 설명한다.\n"
            "7. 인터넷이나 일반 상식 등, Context 밖의 외부 지식은 사용하지 않는다."
        )

        user_content = (
            f"[사용자 질문]\n{query}\n\n"
            "[약관 발췌(Context)]\n"
            f"{context}\n\n"
            "위 Context만을 근거로 위 질문에 답변해 주세요.\n"
            "- Context 문장을 그대로 복사하기보다는, 핵심 내용과 조건을 요약해서 설명해 주세요.\n"
            "- 관련 있는 페이지/조항이 있다면 답변 안에서 같이 언급해 주세요.\n"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # 💬 비스트리밍 답변 (옵션)
    # ------------------------------------------------------------------
    def chat(self, query: str, top_k: int = 5, max_tokens: int = 800) -> Dict[str, Any]:
        """
        RAG 기반 단일 응답 (스트리밍 X)
        """
        context = self.search_similar_docs(query, top_k=top_k)

        if not context:
            return {
                "answer": "관련된 약관 내용을 찾을 수 없습니다. 다른 표현으로 질문해 보세요.",
                "sources": [],
            }

        messages = self._build_messages(query, context)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
            )
            answer = completion.choices[0].message.content.strip()
            return {
                "answer": answer,
                "sources": self.last_sources,
            }
        except Exception as e:
            return {
                "answer": f"❌ 오류 발생: {str(e)}",
                "sources": [],
            }

    # ------------------------------------------------------------------
    # 💬 스트리밍 답변 (Streamlit에서 사용)
    # ------------------------------------------------------------------
    def chat_streaming(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 800,
    ):
        """
        RAG 기반 스트리밍 응답 제너레이터.

        Streamlit UI에서:

            for chunk in chatbot.chat_streaming(user_input):
                if not chunk["done"]:
                    # chunk["answer"]로 실시간 출력
                else:
                    # chunk["sources"]로 출처 표시
        """
        context = self.search_similar_docs(query, top_k=top_k)

        if not context:
            yield {
                "answer": "관련된 약관 내용을 찾을 수 없습니다. 다른 표현으로 질문해 보세요.",
                "done": True,
                "sources": [],
            }
            return

        messages = self._build_messages(query, context)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
                stream=True,
            )

            answer = ""

            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta else ""
                if delta:
                    answer += delta
                    # 중간 스트리밍 단계에는 sources 없음
                    yield {
                        "answer": answer,
                        "done": False,
                    }

            # 스트리밍 종료 시 최종 answer + sources 반환
            yield {
                "answer": answer,
                "done": True,
                "sources": self.last_sources,
            }

        except Exception as e:
            yield {
                "answer": f"❌ 오류 발생: {str(e)}",
                "done": True,
                "sources": [],
            }


# 단독 실행 테스트용
if __name__ == "__main__":
    bot = RAGChatbot()
    q = "해지 시 환급금은 어떻게 되나요?"
    res = bot.chat(q)
    print("Q:", q)
    print("A:", res["answer"])
    print("Sources:", res["sources"])
