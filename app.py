import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. SETUP PAGE
st.set_page_config(page_title="PDF AI Chatbot", page_icon="🤖")
st.title("🤖 Your Context-Aware AI (Gemini Edition)")
st.markdown("---")

# 2. LOAD DATA
@st.cache_resource(show_spinner=False)
def load_chain():
    # Load Embeddings (Ensure these match your ingest.py script)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to the local vector store
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Define Safety Settings to prevent Gemini from blocking responses
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Initialize Gemini with the API Key from Streamlit Secrets
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        safety_settings=safety_settings,
        temperature=0.4  # Slightly lower for more factual accuracy
    )
    
    # Memory setup using the classic module for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key='answer'
    )
    
    # Create the RAG (Retrieval-Augmented Generation) Chain
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

# 3. INITIALIZE SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the brain
try:
    chain = load_chain()
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.info("Check your GOOGLE_API_KEY in the Secrets dashboard.")
    st.stop()

# 4. DISPLAY CHAT HISTORY
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 5. USER INPUT & RESPONSE LOGIC
if prompt := st.chat_input("Ask about the document..."):
    # Add user message to UI and history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process AI Response
    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            try:
                # The .invoke method is the standard for LangChain in 2026
                response = chain.invoke({"question": prompt})
                
                # Safely get the answer from the dictionary
                answer = response.get("answer", "I couldn't find a clear answer in the document.")
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Execution Error: {e}")
                st.info("Try refreshing the page or checking your internet connection.")