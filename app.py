import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
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
    
    # Connect to Vector DB
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Secure Token Fetch
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except KeyError:
        st.error("HF_TOKEN not found! Please add it to Streamlit Secrets.")
        st.stop()
    
    # A. Define the Endpoint (The Engine)
    # Using Zephyr-7b as it is more stable for the 'conversational' task
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="conversational", 
        huggingfacehub_api_token=hf_token, 
        temperature=0.5,
        max_new_tokens=512,
        timeout=300
    )
    
    # B. Wrap it for Chat (The Translator)
    # This fixes the "task conversational" error by formatting inputs correctly
    chat_model = ChatHuggingFace(llm=llm_endpoint)
    
    # Setup Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key='answer'
    )
    
    # Create the RAG Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
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
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                # The invoke call now flows through ChatHuggingFace
                result = chain.invoke({"question": prompt})
                answer = result["answer"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"AI Provider Error: {e}")
                st.info("The model provider might be busy. Try asking again in 10 seconds.")