import streamlit as st
import os
import re
import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI RAG Tool Lab",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .tool-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🤖 AI RAG + Tool Calling Lab</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Learning RAG, tool calling, and agent-style decision making'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# API KEY
# =========================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
    )
    st.stop()


client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# =========================================================
# SESSION STATE
# =========================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# DOCUMENT EXTRACTION
# =========================================================

def extract_pdf(file):
    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx(file):
    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def extract_txt(file):
    return file.read().decode("utf-8")


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(
    text,
    chunk_size=700,
    overlap=100
):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# BUILD VECTOR DATABASE
# =========================================================

def build_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings.astype("float32"))

    return index, embeddings


# =========================================================
# RAG SEARCH TOOL
# =========================================================

def rag_search(query, top_k=4):

    if (
        st.session_state.index is None
        or not st.session_state.chunks
    ):
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = st.session_state.index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for score, index in zip(scores[0], indices[0]):

        if index == -1:
            continue

        results.append(
            {
                "chunk": st.session_state.chunks[index],
                "score": float(score),
                "index": int(index)
            }
        )

    return results


# =========================================================
# CALCULATOR TOOL
# =========================================================

def calculator(expression):

    try:

        expression = expression.strip()

        # Allow only mathematical characters
        if not re.fullmatch(
            r"[0-9+\-*/().%\s]+",
            expression
        ):
            return "Invalid mathematical expression."

        # Convert percentage
        expression = expression.replace("%", "/100")

        result = eval(
            expression,
            {"__builtins__": None},
            {}
        )

        return str(result)

    except Exception:
        return "Could not calculate the expression."


# =========================================================
# TOOL DEFINITIONS
# =========================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the uploaded documents for relevant "
                "information. Use this when the user's question "
                "requires information from the uploaded document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query to find relevant "
                            "information in the document."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Perform mathematical calculations. Use this "
                "when the user asks for arithmetic or numerical "
                "calculation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Mathematical expression such as "
                            "25 * 4 or (100 + 50) / 3."
                        )
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# =========================================================
# EXECUTE TOOL
# =========================================================

def execute_tool(tool_name, arguments):

    if tool_name == "rag_search":

        query = arguments.get("query", "")

        results = rag_search(
            query,
            top_k=4
        )

        if not results:
            return "No relevant information was found."

        output = []

        for i, result in enumerate(results, start=1):

            output.append(
                f"Result {i} "
                f"(similarity: {result['score']:.3f}):\n"
                f"{result['chunk']}"
            )

        return "\n\n".join(output)

    elif tool_name == "calculator":

        expression = arguments.get(
            "expression",
            ""
        )

        return calculator(expression)

    return "Unknown tool."


# =========================================================
# AGENT / TOOL CALLING
# =========================================================

def ask_agent(user_question):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent AI assistant. "
                "You have access to two tools: "
                "RAG document search and calculator. "
                "\n\n"
                "Use RAG search when the answer depends on "
                "the uploaded document. "
                "\n\n"
                "Use calculator when mathematical calculation "
                "is required. "
                "\n\n"
                "You may use tools when necessary. "
                "After receiving tool results, provide a clear "
                "final answer to the user."
            )
        }
    ]

    # Add conversation history
    for message in st.session_state.messages[-6:]:

        messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # =====================================================
    # FIRST AI REQUEST
    # =====================================================

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=messages,

        tools=tools,

        tool_choice="auto",

        temperature=0.2,

        max_tokens=1200
    )

    assistant_message = response.choices[0].message

    # =====================================================
    # NO TOOL REQUIRED
    # =====================================================

    if not assistant_message.tool_calls:

        return assistant_message.content, []


    # =====================================================
    # TOOL CALLS
    # =====================================================

    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in assistant_message.tool_calls
            ]
        }
    )

    tool_results = []

    # =====================================================
    # EXECUTE EACH TOOL
    # =====================================================

    for tool_call in assistant_message.tool_calls:

        tool_name = tool_call.function.name

        arguments = tool_call.function.arguments

        import json

        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}

        result = execute_tool(
            tool_name,
            arguments
        )

        tool_results.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            }
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            }
        )

    # =====================================================
    # SECOND AI REQUEST
    # =====================================================

    final_response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=messages,

        temperature=0.2,

        max_tokens=1200
    )

    final_answer = final_response.choices[0].message.content

    return final_answer, tool_results


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ RAG Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=4
    )

    st.divider()

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file:

        st.write(
            f"**File:** {uploaded_file.name}"
        )

        process_button = st.button(
            "🔄 Process Document",
            use_container_width=True
        )

        if process_button:

            with st.spinner(
                "Reading and indexing document..."
            ):

                try:

                    if uploaded_file.name.lower().endswith(
                        ".pdf"
                    ):
                        text = extract_pdf(
                            uploaded_file
                        )

                    elif uploaded_file.name.lower().endswith(
                        ".docx"
                    ):
                        text = extract_docx(
                            uploaded_file
                        )

                    else:
                        text = extract_txt(
                            uploaded_file
                        )

                    text = clean_text(text)

                    if not text:

                        st.error(
                            "No readable text found."
                        )

                    else:

                        chunks = create_chunks(
                            text
                        )

                        index, embeddings = (
                            build_vector_database(
                                chunks
                            )
                        )

                        st.session_state.chunks = chunks

                        st.session_state.embeddings = (
                            embeddings
                        )

                        st.session_state.index = index

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.success(
                            "Document indexed successfully!"
                        )

                except Exception as e:

                    st.error(
                        f"Processing error: {e}"
                    )

    # =====================================================
    # TOOL STATUS
    # =====================================================

    st.divider()

    st.header("🛠️ Available Tools")

    st.success("🔎 RAG Search")

    st.success("🧮 Calculator")

    st.divider()

    if st.session_state.index is not None:

        st.info(
            f"Document: "
            f"{st.session_state.document_name}\n\n"
            f"Chunks: "
            f"{len(st.session_state.chunks)}"
        )

    else:

        st.warning(
            "Upload and process a document first."
        )


# =========================================================
# MAIN TOOL EXPLANATION
# =========================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        ### 🔎 RAG Search

        The AI can search your uploaded document when
        it needs external knowledge.
        """
    )

with col2:

    st.markdown(
        """
        ### 🧮 Calculator

        The AI can call a calculator when a numerical
        calculation is required.
        """
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("tools")
        ):

            with st.expander(
                "🛠️ Tool activity"
            ):

                for tool in message["tools"]:

                    st.write(
                        f"**Tool:** {tool['tool']}"
                    )

                    st.write(
                        f"**Arguments:** "
                        f"{tool['arguments']}"
                    )

                    st.write(
                        f"**Result:** "
                        f"{tool['result']}"
                    )


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask something..."
)


if user_question:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            user_question
        )

    # -----------------------------------------------------
    # AI RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "AI is thinking and selecting tools..."
        ):

            try:

                answer, tool_results = ask_agent(
                    user_question
                )

                st.markdown(
                    answer
                )

                # -----------------------------------------
                # TOOL ACTIVITY
                # -----------------------------------------

                if tool_results:

                    with st.expander(
                        "🛠️ Tool activity"
                    ):

                        for tool in tool_results:

                            st.write(
                                f"**Tool:** "
                                f"{tool['tool']}"
                            )

                            st.write(
                                f"**Arguments:** "
                                f"{tool['arguments']}"
                            )

                            st.write(
                                f"**Result:** "
                                f"{tool['result']}"
                            )

                # -----------------------------------------
                # SAVE ASSISTANT MESSAGE
                # -----------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "tools": tool_results
                    }
                )

            except Exception as e:

                error_message = (
                    f"AI request failed: {e}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "tools": []
                    }
                )
