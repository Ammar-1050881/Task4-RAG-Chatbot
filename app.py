import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. SETUP PAGE
st.set_page_config(page_title="PDF AI Chatbot", page_icon="🤖")
st.title("🤖 Your Context-Aware AI")
st.markdown("---")

# 2. LOAD DATA
@st.cache_resource 
def load_chain():
    # Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to Vector DB (Ensure ingest.py was run)
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Secure Token Fetch
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except KeyError:
        st.error("HF_TOKEN not found! Please add it to Streamlit Secrets.")
        st.stop()
    
    # LLM Setup - Switching to Mistral v0.3 (High Support)
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation", 
        huggingfacehub_api_token=hf_token, 
        temperature=0.7,
        max_new_tokens=512,
        timeout=300
    )
    
    # Setup Memory
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

# 3. INITIALIZE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    chain = load_chain()
except Exception as e:
    st.error(f"Error loading the AI: {e}")
    st.stop()

# 4. DISPLAY CHAT
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. USER INPUT & RESPONSE
if prompt := st.chat_input("Ask me about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                result = chain.invoke({"question": prompt})
                answer = result["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"AI Provider Error: {e}")
                st.info("The model is warming up. Please try one more time.")