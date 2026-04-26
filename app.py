import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
 
from dotenv import load_dotenv
load_dotenv()
 
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
st.sidebar.write("Settings")
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload Pdf's and chat with their content")
 
## Input the Groq API Key
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
 
## Clear chat history button
if st.sidebar.button("🗑️ Clear Chat History"):
    if "store" in st.session_state:
        del st.session_state.store
    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore
    if "uploaded_file_names" in st.session_state:
        del st.session_state.uploaded_file_names
    st.rerun()
 
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
 
    session_id = st.text_input("Session ID", value="default_session")
 
    if 'store' not in st.session_state:
        st.session_state.store = {}
 
    uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)
 
    if uploaded_files:
        ## ✅ Check if uploaded files changed — if yes reset vectorstore
        current_file_names = [f.name for f in uploaded_files]
 
        if st.session_state.get("uploaded_file_names") != current_file_names:
            ## New files uploaded — reset vectorstore and chat history
            st.session_state.uploaded_file_names = current_file_names
            st.session_state.store = {}  ## reset chat history too
 
            documents = []
            for uploaded_file in uploaded_files:
                temppdf = f"./temp_{uploaded_file.name}.pdf"
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())
 
                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)
 
            ## Split and create embeddings
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
 
            ## ✅ Save vectorstore in session_state
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.success(f"✅ Loaded {len(current_file_names)} file(s): {', '.join(current_file_names)}")
 
        ## Use vectorstore from session_state
        retriever = st.session_state.vectorstore.as_retriever()
 
        ## History aware retriever
        contextualize_q_system_prompt = (
            "Take the chat history and the user's latest question. "
            "If the question refers to earlier context, rewrite it so it is clear on its own. "
            "Do not answer the question. "
            "If the question is already clear, just return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
 
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
 
        ## Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
 
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
 
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
 
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
 
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
 
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the Groq API Key")
 








