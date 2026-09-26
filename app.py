import os
import re
import json

import streamlit as st
import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Agentic RAG Lab",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🤖 Agentic RAG Lab")

st.caption(
    "RAG + Tool Calling + Agent Loop"
)


# =========================================================
# GROQ API
# =========================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not found. "
        "Add it to Streamlit Secrets."
    )
    st.stop()


client = Groq(
    api_key=GROQ_API_KEY
)


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "openai/gpt-oss-120b"


# =========================================================
# EMBEDDING MODEL
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

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []


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

        if paragraph.text.strip():

            text += paragraph.text + "\n"

    return text


def extract_txt(file):

    return file.read().decode(
        "utf-8"
    )


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CREATE CHUNKS
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

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += (
            chunk_size - overlap
        )

    return chunks


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def build_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype(
            "float32"
        )
    )

    return index


# =========================================================
# RAG SEARCH
# =========================================================

def rag_search(
    query,
    top_k=4
):

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

    scores, indices = (
        st.session_state.index.search(
            query_embedding.astype(
                "float32"
            ),
            top_k
        )
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        results.append(
            {
                "chunk": st.session_state.chunks[idx],
                "score": float(score),
                "index": int(idx)
            }
        )

    return results


# =========================================================
# CALCULATOR
# =========================================================

def calculator(expression):

    try:

        expression = expression.strip()

        # Only allow safe mathematical characters
        if not re.fullmatch(
            r"[0-9+\-*/().%\s]+",
            expression
        ):

            return (
                "Invalid mathematical expression."
            )

        expression = expression.replace(
            "%",
            "/100"
        )

        result = eval(
            expression,
            {
                "__builtins__": None
            },
            {}
        )

        return str(result)

    except Exception:

        return (
            "Could not calculate "
            "the expression."
        )


# =========================================================
# TOOL DEFINITIONS
# =========================================================

TOOLS = [

    {
        "type": "function",

        "function": {

            "name": "rag_search",

            "description": (
                "Search the uploaded document "
                "for information relevant to "
                "the user's question."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type": "string",

                        "description": (
                            "A focused search query "
                            "for the uploaded document."
                        )
                    }
                },

                "required": [
                    "query"
                ]
            }
        }
    },

    {
        "type": "function",

        "function": {

            "name": "calculator",

            "description": (
                "Perform mathematical calculations."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "expression": {

                        "type": "string",

                        "description": (
                            "A mathematical expression "
                            "such as 25 * 10 or "
                            "(100 + 50) / 3."
                        )
                    }
                },

                "required": [
                    "expression"
                ]
            }
        }
    }
]


# =========================================================
# EXECUTE RAG TOOL
# =========================================================

def execute_rag_tool(
    query,
    top_k=4
):

    results = rag_search(
        query,
        top_k
    )

    if not results:

        return (
            "No relevant information "
            "was found in the document."
        )

    output = []

    for i, result in enumerate(
        results,
        start=1
    ):

        output.append(
            f"Result {i} "
            f"(similarity: "
            f"{result['score']:.3f})\n"
            f"{result['chunk']}"
        )

    return "\n\n".join(
        output
    )


# =========================================================
# EXECUTE ANY TOOL
# =========================================================

def execute_tool(
    tool_name,
    arguments,
    top_k
):

    if tool_name == "rag_search":

        query = arguments.get(
            "query",
            ""
        )

        return execute_rag_tool(
            query,
            top_k
        )

    if tool_name == "calculator":

        expression = arguments.get(
            "expression",
            ""
        )

        return calculator(
            expression
        )

    return "Unknown tool."


# =========================================================
# AGENT
# =========================================================

def run_agent(
    user_question,
    top_k=4,
    max_iterations=5
):

    # ---------------------------------------------
    # AGENT LOG
    # ---------------------------------------------

    agent_logs = []

    # ---------------------------------------------
    # SYSTEM MESSAGE
    # ---------------------------------------------

    system_message = """
You are an intelligent Agentic RAG assistant.

You have access to two tools:

1. rag_search
   - Searches information inside the uploaded document.

2. calculator
   - Performs mathematical calculations.

Your job is to decide what action is required.

Use rag_search when the answer depends on the uploaded
document.

Use calculator when a mathematical calculation is needed.

You may call tools multiple times.

After receiving a tool result, analyze it and decide
whether another tool is necessary.

Do not call tools unnecessarily.

When you have enough information, provide the final
answer directly.

If the uploaded document does not contain the requested
information, clearly say that the information was not
found.

Do not invent information from the document.
"""

    messages = [

        {
            "role": "system",
            "content": system_message
        }

    ]

    # ---------------------------------------------
    # ADD RECENT CONVERSATION
    # ---------------------------------------------

    for message in (
        st.session_state.messages[-6:]
    ):

        messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )

    # ---------------------------------------------
    # CURRENT USER QUESTION
    # ---------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # =============================================
    # AGENT LOOP
    # =============================================

    for iteration in range(
        max_iterations
    ):

        agent_logs.append(
            {
                "type": "thinking",
                "iteration": iteration + 1,
                "message": (
                    f"Agent iteration "
                    f"{iteration + 1}"
                )
            }
        )

        # -----------------------------------------
        # ASK MODEL
        # -----------------------------------------

        response = (
            client.chat.completions.create(

                model=MODEL_NAME,

                messages=messages,

                tools=TOOLS,

                tool_choice="auto",

                temperature=0.2,

                max_tokens=1200
            )
        )

        assistant_message = (
            response.choices[0].message
        )

        # -----------------------------------------
        # NO TOOL CALL
        # -----------------------------------------

        if not assistant_message.tool_calls:

            final_answer = (
                assistant_message.content
            )

            if not final_answer:

                final_answer = (
                    "I could not generate "
                    "a final answer."
                )

            agent_logs.append(
                {
                    "type": "final",
                    "iteration": iteration + 1,
                    "message": "Agent produced final answer."
                }
            )

            return (
                final_answer,
                agent_logs
            )

        # -----------------------------------------
        # SAVE ASSISTANT TOOL CALL
        # -----------------------------------------

        tool_call_data = []

        for tool_call in (
            assistant_message.tool_calls
        ):

            tool_call_data.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": (
                            tool_call.function.name
                        ),
                        "arguments": (
                            tool_call.function.arguments
                        )
                    }
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": (
                    assistant_message.content
                    or ""
                ),
                "tool_calls": tool_call_data
            }
        )

        # -----------------------------------------
        # EXECUTE TOOLS
        # -----------------------------------------

        for tool_call in (
            assistant_message.tool_calls
        ):

            tool_name = (
                tool_call.function.name
            )

            raw_arguments = (
                tool_call.function.arguments
            )

            try:

                arguments = json.loads(
                    raw_arguments
                )

            except Exception:

                arguments = {}

            # -------------------------------------
            # LOG TOOL CALL
            # -------------------------------------

            agent_logs.append(
                {
                    "type": "tool",
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "arguments": arguments
                }
            )

            # -------------------------------------
            # RUN TOOL
            # -------------------------------------

            result = execute_tool(
                tool_name,
                arguments,
                top_k
            )

            # -------------------------------------
            # LOG RESULT
            # -------------------------------------

            agent_logs.append(
                {
                    "type": "result",
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "result": result
                }
            )

            # -------------------------------------
            # SEND RESULT BACK TO MODEL
            # -------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": (
                        tool_call.id
                    ),
                    "content": result
                }
            )

    # =============================================
    # MAX ITERATIONS REACHED
    # =============================================

    agent_logs.append(
        {
            "type": "limit",
            "message": (
                "Maximum agent iterations reached."
            )
        }
    )

    # Ask the model for a final response
    # based on everything collected so far.

    final_response = (
        client.chat.completions.create(

            model=MODEL_NAME,

            messages=messages,

            temperature=0.2,

            max_tokens=1200
        )
    )

    final_answer = (
        final_response.choices[0].message.content
    )

    if not final_answer:

        final_answer = (
            "The agent reached its maximum "
            "number of steps."
        )

    return (
        final_answer,
        agent_logs
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Agent Settings")

    top_k = st.slider(
        "RAG results",
        min_value=1,
        max_value=8,
        value=4
    )

    max_iterations = st.slider(
        "Maximum agent steps",
        min_value=1,
        max_value=8,
        value=5
    )

    st.divider()

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "PDF, DOCX or TXT",
        type=[
            "pdf",
            "docx",
            "txt"
        ]
    )

    if uploaded_file:

        st.write(
            f"**File:** "
            f"{uploaded_file.name}"
        )

        if st.button(
            "🔄 Process Document",
            use_container_width=True
        ):

            with st.spinner(
                "Processing document..."
            ):

                try:

                    filename = (
                        uploaded_file.name.lower()
                    )

                    if filename.endswith(
                        ".pdf"
                    ):

                        text = extract_pdf(
                            uploaded_file
                        )

                    elif filename.endswith(
                        ".docx"
                    ):

                        text = extract_docx(
                            uploaded_file
                        )

                    else:

                        text = extract_txt(
                            uploaded_file
                        )

                    text = clean_text(
                        text
                    )

                    if not text:

                        st.error(
                            "No readable text "
                            "was found."
                        )

                    else:

                        chunks = create_chunks(
                            text
                        )

                        index = (
                            build_vector_database(
                                chunks
                            )
                        )

                        st.session_state.chunks = (
                            chunks
                        )

                        st.session_state.index = (
                            index
                        )

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.success(
                            "Document processed!"
                        )

                except Exception as e:

                    st.error(
                        f"Processing error: {e}"
                    )

    st.divider()

    st.header("🧠 Agent Tools")

    st.success("🔎 RAG Search")

    st.success("🧮 Calculator")

    st.divider()

    if st.session_state.index:

        st.info(
            f"Document: "
            f"{st.session_state.document_name}\n\n"
            f"Chunks: "
            f"{len(st.session_state.chunks)}"
        )

    else:

        st.warning(
            "No document loaded."
        )


# =========================================================
# MAIN INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "RAG",
        "Active"
    )

with col2:

    st.metric(
        "Tools",
        "2"
    )

with col3:

    st.metric(
        "Agent",
        "Active"
    )


st.divider()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("agent_logs")
        ):

            with st.expander(
                "🧠 View Agent Activity"
            ):

                for log in (
                    message["agent_logs"]
                ):

                    if log["type"] == "thinking":

                        st.write(
                            f"🔄 "
                            f"{log['message']}"
                        )

                    elif log["type"] == "tool":

                        st.write(
                            f"🛠️ Tool: "
                            f"{log['tool']}"
                        )

                        st.write(
                            f"Arguments: "
                            f"{log['arguments']}"
                        )

                    elif log["type"] == "result":

                        st.write(
                            f"📋 "
                            f"{log['tool']} result:"
                        )

                        st.code(
                            log["result"]
                        )

                    elif log["type"] == "final":

                        st.write(
                            "✅ Agent completed."
                        )

                    elif log["type"] == "limit":

                        st.warning(
                            log["message"]
                        )


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask the Agent..."
)


if user_question:

    # ---------------------------------------------
    # DISPLAY USER
    # ---------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )

    # ---------------------------------------------
    # RUN AGENT
    # ---------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🧠 Agent is working..."
        ):

            try:

                answer, agent_logs = (
                    run_agent(
                        user_question,
                        top_k,
                        max_iterations
                    )
                )

                st.markdown(
                    answer
                )

                # ---------------------------------
                # SHOW AGENT ACTIVITY
                # ---------------------------------

                with st.expander(
                    "🧠 View Agent Activity"
                ):

                    for log in agent_logs:

                        if log["type"] == "thinking":

                            st.write(
                                f"🔄 "
                                f"{log['message']}"
                            )

                        elif log["type"] == "tool":

                            st.write(
                                f"🛠️ "
                                f"{log['tool']}"
                            )

                            st.write(
                                f"Arguments: "
                                f"{log['arguments']}"
                            )

                        elif log["type"] == "result":

                            st.write(
                                f"📋 "
                                f"{log['tool']} result"
                            )

                            st.code(
                                log["result"]
                            )

                        elif log["type"] == "final":

                            st.success(
                                "Agent completed."
                            )

                        elif log["type"] == "limit":

                            st.warning(
                                log["message"]
                            )

                # ---------------------------------
                # SAVE RESPONSE
                # ---------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "agent_logs": agent_logs
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
                        "agent_logs": []
                    }
                )
