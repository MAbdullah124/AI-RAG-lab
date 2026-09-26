import os
import streamlit as st
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI RAG Lab",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #777;
    margin-bottom: 30px;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 15px;
}

.chunk-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🧠 AI RAG Lab</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Learn Retrieval-Augmented Generation with Groq + FAISS'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# GROQ API KEY
# =========================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:

    st.error(
        "Groq API key not found. "
        "Add GROQ_API_KEY to your Streamlit secrets."
    )

    st.stop()


# =========================================================
# GROQ CLIENT
# =========================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# SESSION STATE
# =========================================================

if "chunks" not in st.session_state:

    st.session_state.chunks = []


if "faiss_index" not in st.session_state:

    st.session_state.faiss_index = None


if "document_name" not in st.session_state:

    st.session_state.document_name = None


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf(file):

    text = ""

    reader = PdfReader(file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# =========================================================
# DOCX TEXT EXTRACTION
# =========================================================

def extract_docx(file):

    document = Document(file)

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            paragraphs.append(
                paragraph.text
            )

    return "\n".join(paragraphs)


# =========================================================
# TXT TEXT EXTRACTION
# =========================================================

def extract_txt(file):

    return file.read().decode(
        "utf-8",
        errors="ignore"
    )


# =========================================================
# GENERAL TEXT EXTRACTION
# =========================================================

def extract_text(file):

    filename = file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf(file)

    elif filename.endswith(".docx"):

        return extract_docx(file)

    elif filename.endswith(".txt"):

        return extract_txt(file)

    return ""


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(
    text,
    chunk_size=700,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += step

    return chunks


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def create_embeddings(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype(
        "float32"
    )


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_faiss_index(chunks):

    embeddings = create_embeddings(
        chunks
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    question,
    top_k=4
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    distances, indices = (
        st.session_state.faiss_index.search(
            question_embedding,
            top_k
        )
    )

    results = []

    scores = []

    for score, index in zip(
        distances[0],
        indices[0]
    ):

        if index != -1:

            results.append(
                st.session_state.chunks[index]
            )

            scores.append(
                float(score)
            )

    return results, scores


# =========================================================
# GENERATE ANSWER WITH GROQ
# =========================================================

def generate_answer(
    question,
    retrieved_chunks
):

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are an AI university study assistant.

Answer the student's question using the
provided document context.

IMPORTANT RULES:

1. Use the provided context as the main source.
2. Do not invent information.
3. If the answer is not present in the
   provided context, say that the information
   was not found in the uploaded document.
4. Explain concepts in simple,
   student-friendly language.
5. Give a direct answer first.
6. Use bullet points when useful.

DOCUMENT CONTEXT:
=================

{context}

=================

STUDENT QUESTION:

{question}

Now provide the answer.
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful university "
                    "study assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=1200
    )

    return response.choices[0].message.content


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 Document")

    uploaded_file = st.file_uploader(
        "Upload your study material",
        type=[
            "pdf",
            "docx",
            "txt"
        ]
    )

    if uploaded_file:

        if st.button(
            "🔎 Process Document",
            use_container_width=True
        ):

            with st.spinner(
                "Processing your document..."
            ):

                try:

                    # Extract text

                    text = extract_text(
                        uploaded_file
                    )

                    if not text.strip():

                        st.error(
                            "No readable text was found "
                            "in this document."
                        )

                    else:

                        # Create chunks

                        chunks = create_chunks(
                            text
                        )

                        if not chunks:

                            st.error(
                                "Could not create document chunks."
                            )

                        else:

                            # Create FAISS index

                            index = create_faiss_index(
                                chunks
                            )

                            # Save everything

                            st.session_state.chunks = (
                                chunks
                            )

                            st.session_state.faiss_index = (
                                index
                            )

                            st.session_state.document_name = (
                                uploaded_file.name
                            )

                            st.success(
                                "Document processed successfully!"
                            )

                            st.info(
                                f"Created {len(chunks)} chunks."
                            )

                except Exception as e:

                    st.error(
                        f"Error processing document: {e}"
                    )

    # -----------------------------------------------------
    # DOCUMENT INFORMATION
    # -----------------------------------------------------

    if st.session_state.document_name:

        st.divider()

        st.subheader(
            "📊 Document Information"
        )

        st.write(
            f"**File:** "
            f"{st.session_state.document_name}"
        )

        st.write(
            f"**Chunks:** "
            f"{len(st.session_state.chunks)}"
        )

        st.write(
            "**Vector Database:** FAISS"
        )

        st.write(
            "**Embedding Model:** "
            "all-MiniLM-L6-v2"
        )

        st.write(
            "**LLM:** Groq"
        )


# =========================================================
# MAIN QUESTION AREA
# =========================================================

st.header(
    "🔍 Ask Questions From Your Document"
)


if st.session_state.faiss_index is None:

    st.info(
        "👈 Upload a PDF, DOCX, or TXT file "
        "from the sidebar and process it first."
    )

else:

    question = st.text_area(
        "Ask a question:",
        placeholder=(
            "Example: What is deadlock?"
        ),
        height=100
    )

    top_k = st.slider(
        "Number of relevant chunks to retrieve",
        min_value=1,
        max_value=8,
        value=4
    )

    if st.button(
        "🤖 Ask AI",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                # -----------------------------------------
                # RETRIEVAL
                # -----------------------------------------

                with st.spinner(
                    "🔎 Searching your document..."
                ):

                    retrieved_chunks, scores = (
                        retrieve_chunks(
                            question,
                            top_k
                        )
                    )

                if not retrieved_chunks:

                    st.warning(
                        "No relevant information was found."
                    )

                else:

                    # -------------------------------------
                    # GENERATION
                    # -------------------------------------

                    with st.spinner(
                        "🤖 Generating answer with Groq..."
                    ):

                        answer = generate_answer(
                            question,
                            retrieved_chunks
                        )

                    # -------------------------------------
                    # ANSWER
                    # -------------------------------------

                    st.subheader(
                        "💡 Answer"
                    )

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        {answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # -------------------------------------
                    # RETRIEVED CONTEXT
                    # -------------------------------------

                    with st.expander(
                        "🔎 View Retrieved Chunks"
                    ):

                        for i, (
                            chunk,
                            score
                        ) in enumerate(
                            zip(
                                retrieved_chunks,
                                scores
                            ),
                            start=1
                        ):

                            st.markdown(
                                f"### Chunk {i}"
                            )

                            st.caption(
                                f"Similarity score: "
                                f"{score:.4f}"
                            )

                            st.markdown(
                                f"""
                                <div class="chunk-box">
                                {chunk}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            except Exception as e:

                st.error(
                    f"AI request failed: {e}"
                )


# =========================================================
# RAG PIPELINE INFORMATION
# =========================================================

st.divider()

st.subheader(
    "🧠 How This RAG System Works"
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.write("📄 **1. Document**")

    st.caption(
        "Upload PDF, DOCX or TXT"
    )

with col2:

    st.write("✂️ **2. Chunking**")

    st.caption(
        "Split the document"
    )

with col3:

    st.write("🔢 **3. Embeddings**")

    st.caption(
        "Convert text to vectors"
    )

with col4:

    st.write("🔍 **4. Retrieval**")

    st.caption(
        "Find relevant chunks"
    )

with col5:

    st.write("🤖 **5. Generation**")

    st.caption(
        "Groq generates the answer"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI RAG Lab • "
    "Document → Chunks → Embeddings → FAISS → "
    "Retrieval → Groq → Answer"
)
