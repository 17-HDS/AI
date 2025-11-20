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
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def initialize_chatbot():
    try:
        chatbot = RAGChatbot()
        count = chatbot.get_collection_info()

        if count == 0:
            st.session_state.initialized = False
            st.error("❌ 벡터 저장소가 비어있습니다. 먼저 데이터를 구축하세요.")
            return None

        st.session_state.chatbot = chatbot
        st.session_state.initialized = True

        if 'auto_init_done' not in st.session_state:
            st.session_state.auto_init_done = True
            st.success(f"✅ 챗봇 초기화 완료! (총 {count}개 문서)")
    except Exception as e:
        st.session_state.initialized = False
        st.error(f"❌ 챗봇 초기화 오류: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="보험 약관 RAG 챗봇",
        page_icon="📘",
        layout="wide"
    )

    # ------------------------------------------------------------------
    # 🎨 깔끔한 화이트톤 + 부드러운 말풍선 + 네오모피즘 + 글래스모피즘
    # ------------------------------------------------------------------
    st.markdown("""
    <style>

    /* 전체 배경: 따뜻한 화이트 라이트톤 */
    .stApp {
        background-color: #FAFAFA !important;
    }

    /* 페이지 기본 padding 줄여서 상단 공백 제거 */
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* 채팅창 전체 */
    .chat-container {
        height: calc(100vh - 700px);
        overflow-y: auto;
        padding: 0 0.5rem;
        display: flex;
        flex-direction: column;
    }

    /* 사용자 말풍선 */
    .user-bubble {
        align-self: flex-end;
        background: linear-gradient(135deg, #FFB97A, #FF944D);
        color: white;
        padding: 12px 18px;
        border-radius: 16px;
        margin: 8px 0;
        max-width: 85%;
        box-shadow:
            3px 3px 10px rgba(0,0,0,0.1),
            -2px -2px 8px rgba(255,255,255,0.7);
    }

    /* AI 말풍선: 글래스모피즘 */
    .assistant-bubble {
        align-self: flex-start;
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 12px 18px;
        border-radius: 16px;
        margin: 8px 0;
        max-width: 85%;
        border: 1px solid rgba(230,230,230,0.5);
        box-shadow:
            4px 4px 12px rgba(0,0,0,0.08),
            -3px -3px 10px rgba(255,255,255,0.9);
    }

    /* 입력창 */
    .stTextInput input {
    border-radius: 14px;
    padding: 0.8rem 1rem;
    border: none;
    background: white;
    box-shadow:
        inset 2px 2px 6px rgba(0,0,0,0.07),
        inset -3px -3px 6px rgba(255,255,255,0.8);
    }

    .stTextInput input {
        border-radius: 14px;
        padding: 0.8rem 1rem;
        border: none;
        background: white;
        box-shadow:
            inset 2px 2px 6px rgba(0,0,0,0.07),
            inset -3px -3px 6px rgba(255,255,255,0.8);
    }

    .stTextInput input:focus {
        outline: none;
        box-shadow: 
            inset 1px 1px 4px rgba(0,0,0,0.15),
            inset -1px -1px 4px rgba(255,255,255,0.9),
            0 0 8px rgba(255,140,60,0.35);
    }

    /* 스크롤바 */
    .chat-container::-webkit-scrollbar {
        width: 7px;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: #FF944D;
        border-radius: 4px;
    }
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #E56A00;
    }

    </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------

    initialize_session_state()

    st.title("📘 보험 약관 RAG 챗봇")
    st.markdown("---")

    if not os.getenv('OPENAI_API_KEY'):
        st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
        return

    if not st.session_state.initialized:
        initialize_chatbot()

    # ---------------------- 사이드바 ----------------------
    with st.sidebar:
        st.header("⚙️ 설정")
        if st.button("🔄 챗봇 재초기화"):
            st.session_state.initialized = False
            st.session_state.chatbot = None
            initialize_chatbot()
            st.rerun()

        st.markdown("---")
        st.header("📖 사용법")
        st.markdown("질문하면 약관 기반으로 답변을 드립니다.")

    # ---------------------- 메인 채팅 영역 ----------------------
    col1, col2 = st.columns([2, 1])

    with col1:

        # 채팅 영역
        chat_area = st.container()

        with chat_area:
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

            for chat in st.session_state.chat_history:
                st.markdown(
                    f"<div class='user-bubble'>{chat['query']}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='assistant-bubble'>{chat['answer']}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------- 입력창 + 타이핑 효과 유지 ----------------------
        if st.session_state.initialized:
            with st.container():
                user_input = st.text_input(
                    "질문을 입력하세요...",
                    key="user_input_custom"
                )
                submit = st.button("전송")

                if submit and user_input:
                    # 기존 로직 그대로
                    st.session_state.chat_history.append({
                        "query": user_input,
                        "answer": "",
                        "sources": []
                    })

            if submit and user_input:
                message_placeholder = st.empty()
                full_answer = ""
                final_sources = []

                # 타이핑 스트리밍 유지
                for chunk in st.session_state.chatbot.chat_streaming(user_input):
                    if not chunk["done"]:
                        full_answer = chunk["answer"]
                        message_placeholder.markdown(
                            f"<div class='assistant-bubble'>{full_answer} ▌</div>",
                            unsafe_allow_html=True
                        )
                        time.sleep(0.01)
                    else:
                        full_answer = chunk["answer"]
                        final_sources = chunk["sources"]
                        message_placeholder.markdown(
                            f"<div class='assistant-bubble'>{full_answer}</div>",
                            unsafe_allow_html=True
                        )

                st.session_state.chat_history[-1]["answer"] = full_answer
                st.session_state.chat_history[-1]["sources"] = final_sources

                st.rerun()

    # ---------------------- 시스템 정보 ----------------------
    with col2:
        st.header("📊 시스템 정보")
        if st.session_state.initialized:
            st.metric("문서 수", st.session_state.chatbot.get_collection_info())
            st.metric("대화 수", len(st.session_state.chat_history))

        if st.button("🗑️ 채팅 초기화"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
