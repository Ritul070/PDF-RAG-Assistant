import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import google.generativeai as genai

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = st.secrets["API_KEY"]
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Conversational RAG Chatbot", page_icon="🤖")

st.title("📄 Conversational PDF RAG Chatbot")

# ==========================================
# SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    st.divider()

    st.subheader("💬 Chat History")

    user_questions = [
        msg["content"]
        for msg in st.session_state.messages
        if msg["role"] == "user"
    ]

    if user_questions:
        for question in user_questions:
            st.write(question)
    else:
        st.caption("No chats yet.")

    st.divider()

    if st.button("🗑 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# CHROMADB (In-Memory)
# ==========================================
client = chromadb.Client()

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function
)


# ==========================================
# PROCESS PDF
# ==========================================
if (
    uploaded_file is not None
    and st.session_state.processed_file != uploaded_file.name
):

    try:
        client.delete_collection("documents")
    except:
        pass

    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_function
    )

    reader = PdfReader(uploaded_file)

    pdf_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pdf_text += text + "\n"


    chunk_size = 500

    chunks = [
        pdf_text[i:i + chunk_size]
        for i in range(0, len(pdf_text), chunk_size)
    ]


    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids
    )

    st.session_state.processed_file = uploaded_file.name

    st.success("PDF processed successfully!")    
# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# USER INPUT
# ==========================================
if prompt := st.chat_input("Ask something..."):

    # Display User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

# ======================================
# ROUTING USING CHROMADB
# ======================================

    if uploaded_file is None:

        response = model.generate_content(prompt)

    else:
        

        results = collection.query(
            query_texts=[prompt],
            n_results=3
        )

        distances = results["distances"][0]
        best_distance = distances[0]

)

        if best_distance < 1:

            retrieved_chunks = "\n\n".join(results["documents"][0])


            history = ""

            for msg in st.session_state.messages[:-1]:
                history += (
                    f"{msg['role'].capitalize()}: "
                    f"{msg['content']}\n"
                )

            final_prompt = f"""
    You are an AI assistant.

    Answer ONLY using the PDF context.

    Context:
    {retrieved_chunks}

    Conversation History:
    {history}

    Question:
    {prompt}
    """

            response = model.generate_content(final_prompt)

        else:

            response = model.generate_content(prompt)

    answer = response.text

    # Display Assistant Message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )