import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
# --- NEW UPDATED IMPORTS FOR 2026 ---
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. SETUP PAGE
st.set_page_config(page_title="PDF AI Chatbot", page_icon="🤖")
st.title("🤖 Your Context-Aware AI")
st.markdown("---")

# 2. LOAD DATA (Connecting to what you built in ingest.py)
@st.cache_resource 
def load_chain():
    # Using the same embeddings as ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to the local vector database
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Mistral-7B via Hugging Face - FIXED WITH TASK PARAMETER
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation", # 🌟 ESSENTIAL: Tells HF what type of model this is
        huggingfacehub_api_token="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", 
        temperature=0.5,
        max_new_tokens=512,
        timeout=300
    )
    
    # Setup Memory for context-aware conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key='answer'
    )
    
    # Create the RAG Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return chain

# 3. INITIALIZE CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the brain
try:
    chain = load_chain()
except Exception as e:
    st.error(f"Error loading the AI: {e}")
    st.stop()

# 4. DISPLAY CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. CHAT INPUT & RESPONSE
if prompt := st.chat_input("Ask me about the document..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI message
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                # The RAG Magic
                result = chain.invoke({"question": prompt})
                answer = result["answer"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Something went wrong while thinking: {e}")