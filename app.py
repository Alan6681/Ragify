import streamlit as st
from services.rag_chain import RagChain
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Ragify",
    page_icon= "📕",
    layout="centered"
)

if not "chat_history" in st.session_state:
    st.session_state["chat_history"] = []

if not "vectorstore" in st.session_state:
    st.session_state["vectorstore"] = None

if not "messages" in st.session_state:
    st.session_state["messages"] = [{"role":"assistant", "content": "Hello there ☺️, What are you curious about"}]

st.title("Ragify")

with st.sidebar:
    st.header("Upload File")
    uploaded_files = st.file_uploader(label="Please Upload your pdf", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.markdown("## Select Mode")
        col1, col2 = st.columns(2)
        with col1:
            chat_mode = st.button("Chat Mode", type="secondary")

        with col2:
            quiz_mode = st.button("Quiz Mode", type="secondary")
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1,3,1])

        with col2:
            evaluation = st.button("Evaluation", icon="📉")

        st.markdown("---")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            clear_chat = st.button("Clear chat", type="primary")

if uploaded_files:
    if not "rag_chain" in st.session_state:
        st.session_state["rag_chain"] = RagChain(uploaded_files)


    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if user_query := st.chat_input("Ask about your Pdf"):
    # Display user input
        st.session_state.messages.append({"role" : "user", "content": user_query})
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.write(user_query)

        # Generate AI message
        with st.chat_message("assistant"):
            with st.spinner("thinking..."):
                response = st.session_state.rag_chain.ask(question=user_query, chat_history=st.session_state.chat_history)
                st.write(response)

                st.session_state.chat_history.append(AIMessage(content=response))
                st.session_state.messages.append({"role": "assistant", "content": response})
    
        





# @st.cache_resource
# rag_chain = RagChain()