"""
📘 보험 약관 RAG 챗봇 - Streamlit 웹 인터페이스
"""

import streamlit as st
import os
import time
from dotenv import load_dotenv
from rag_chatbot import RAGChatbot
import json

# 환경변수 로드
load_dotenv()

def initialize_session_state():
    """세션 상태 초기화"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def initialize_chatbot():
    """챗봇 초기화"""
    try:
        with st.spinner("🤖 RAG 챗봇을 초기화하는 중..."):
            chatbot = RAGChatbot()
            count = chatbot.get_collection_info()
            
            if count == 0:
                st.error("❌ 벡터 저장소가 비어있습니다. 먼저 데이터를 구축하세요.")
                return None
            
            st.session_state.chatbot = chatbot
            st.session_state.initialized = True
            st.success(f"✅ 챗봇 초기화 완료! (총 {count}개 문서)")
            
    except Exception as e:
        st.error(f"❌ 챗봇 초기화 오류: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="보험 약관 RAG 챗봇",
        page_icon="📘",
        layout="wide"
    )
    
    # CSS 스타일 추가 (GPT 스타일 채팅 UI)
    st.markdown("""
    <style>
    /* 채팅 영역 고정 높이 및 스크롤 */
    .chat-container {
        height: calc(100vh - 250px);
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    /* 채팅 메시지 스타일 */
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    /* 입력창 하단 고정 */
    .stChatFloatingInputContainer {
        position: sticky;
        bottom: 0;
        z-index: 999;
        background-color: var(--background-color);
        padding: 1rem 0;
    }
    
    /* 메인 레이아웃 조정 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* 헤더 고정 */
    .stApp > header {
        position: fixed;
        top: 0;
        z-index: 1000;
    }
    
    /* 스크롤바 스타일 */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 헤더
    st.title("📘 보험 약관 RAG 챗봇")
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # API 키 확인
        if not os.getenv('OPENAI_API_KEY'):
            st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
            st.info("💡 .env 파일에 OPENAI_API_KEY를 설정하세요.")
            return
        
        # 초기화 버튼
        if st.button("🚀 챗봇 초기화", type="primary"):
            initialize_chatbot()
        
        # 상태 표시
        if st.session_state.initialized:
            st.success("✅ 챗봇 준비 완료")
        else:
            st.warning("⚠️ 챗봇 초기화 필요")
        
        st.markdown("---")
        
        # 사용법
        st.header("📖 사용법")
        st.markdown("""
        1. **초기화**: 챗봇 초기화 버튼 클릭
        2. **질문**: 아래 입력창에 질문 입력
        3. **답변**: AI가 약관을 바탕으로 답변
        4. **출처**: 답변에 페이지 번호 표시
        """)
        
        # 예시 질문
        st.header("💡 예시 질문")
        example_questions = [
            "보험금 지급 사유는 무엇인가요?",
            "보험료는 어떻게 납입하나요?",
            "면책 사항이 있나요?",
            "보험 기간은 얼마나 되나요?",
            "해지 시 환급금은 어떻게 되나요?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                st.session_state.user_input = question
    
    # 메인 채팅 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not st.session_state.initialized:
            st.info("👆 먼저 사이드바에서 챗봇을 초기화하세요.")
        
        # 채팅 영역 (고정 높이, 스크롤 가능)
        chat_area = st.container(height=600)
        
        with chat_area:
            # 채팅 히스토리 표시 (위에서 아래로)
            if st.session_state.chat_history:
                for chat in st.session_state.chat_history:
                    with st.chat_message("user"):
                        st.write(chat["query"])
                    
                    with st.chat_message("assistant"):
                        st.write(chat["answer"])
                        
                        # 출처 정보
                        if chat["sources"]:
                            with st.expander(f"📚 참고 문서 ({len(chat['sources'])}개)"):
                                for j, source in enumerate(chat["sources"], 1):
                                    st.write(f"**문서 {j}** (페이지 {source['page']})")
                                    st.write(source["content"])
                                    st.write("---")
        
        # 사용자 입력 (하단 고정 - Streamlit의 chat_input이 자동으로 하단에 배치됨)
        if st.session_state.initialized:
            user_input = st.chat_input("보험 약관에 대해 질문하세요...", key="chat_input")
            
            if user_input:
                # 사용자 메시지 즉시 표시
                with chat_area:
                    with st.chat_message("user"):
                        st.write(user_input)
                
                # AI 답변 생성 (스트리밍)
                with chat_area:
                    with st.chat_message("assistant"):
                        # 스트리밍 응답을 위한 컨테이너
                        message_placeholder = st.empty()
                        sources_placeholder = st.empty()
                        
                        # 초기 로딩 메시지
                        message_placeholder.markdown("🤔 답변을 생성하는 중...")
                        
                        # 스트리밍 응답 생성
                        full_answer = ""
                        final_sources = []
                        streaming_started = False
                        
                        for chunk in st.session_state.chatbot.chat_streaming(user_input):
                            if not chunk["done"]:
                                # 스트리밍 시작
                                if not streaming_started:
                                    streaming_started = True
                                    message_placeholder.empty()  # 로딩 메시지 제거
                                
                                # 실시간으로 답변 업데이트 (타이핑 효과)
                                full_answer = chunk["answer"]
                                message_placeholder.markdown(full_answer + "▌")
                                time.sleep(0.01)  # 부드러운 스트리밍을 위한 짧은 지연
                            else:
                                # 최종 완성된 응답
                                full_answer = chunk["answer"]
                                final_sources = chunk["sources"]
                                message_placeholder.markdown(full_answer)
                        
                        # 출처 정보 표시
                        if final_sources:
                            with sources_placeholder.expander(f"📚 참고 문서 ({len(final_sources)}개)"):
                                for i, source in enumerate(final_sources, 1):
                                    st.write(f"**문서 {i}** (페이지 {source['page']})")
                                    st.write(source["content"])
                                    st.write("---")
                
                # 채팅 히스토리에 추가
                st.session_state.chat_history.append({
                    "query": user_input,
                    "answer": full_answer,
                    "sources": final_sources
                })
                
                # 페이지 새로고침하여 새 메시지 표시
                st.rerun()
    
    with col2:
        st.header("📊 시스템 정보")
        
        if st.session_state.initialized:
            # 저장소 정보
            st.subheader("📚 벡터 저장소")
            count = st.session_state.chatbot.get_collection_info()
            st.metric("총 문서 수", count)
            
            # 처리된 데이터 정보
            if os.path.exists("processed_data/약관_processed.json"):
                with open("processed_data/약관_processed.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                st.subheader("📄 처리된 PDF")
                st.metric("총 페이지", len(data))
                
                total_tables = sum(page.get("tables_count", 0) for page in data)
                st.metric("총 표", total_tables)
            
            # 채팅 통계
            st.subheader("💬 채팅 통계")
            st.metric("총 대화 수", len(st.session_state.chat_history))
            
        else:
            st.info("챗봇을 초기화하면 시스템 정보가 표시됩니다.")
        
        # 초기화 버튼
        if st.button("🗑️ 채팅 기록 삭제"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()

