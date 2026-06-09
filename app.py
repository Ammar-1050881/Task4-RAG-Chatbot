import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="PDF AI Chatbot", page_icon="🤖")
st.title("🤖 Your Context-Aware AI")
st.markdown("---")

@st.cache_resource(show_spinner=False)
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    hf_token = st.secrets["HF_TOKEN"]
    
    # 1. Define the base engine
    llm_engine = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="conversational", 
        huggingfacehub_api_token=hf_token, 
        temperature=0.7,
        max_new_tokens=512,
    )
    
    # 2. 🌟 THE FIX: Wrap it in a Chat Interface 🌟
    # This forces LangChain to send a "Conversation" instead of "Text Generation"
    chat_model = ChatHuggingFace(llm=llm_engine)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key='answer'
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return chain

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    chain = load_chain()
except Exception as e:
    st.error(f"Error loading the AI: {e}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Flowing through ChatHuggingFace now
                result = chain.invoke({"question": prompt})
                answer = result["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Provider Error: {e}")