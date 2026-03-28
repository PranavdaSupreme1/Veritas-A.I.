# 1. IMPORTS

import streamlit as st
import os
import time
import threading
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate


# 2. CONFIG

# In case we reach the daily query limit on my account, we could switch to someone else's
os.environ["GOOGLE_API_KEY"] = ""  # Note: don't commit this file with the API KEY in here, apparently bots crawl thru git to find vulnerable API keys to exploit.

st.set_page_config(page_title="DPDP Compliance Bot")

# Make sure to put logo.png in the same folder
st.image("logo.png", width=250)

st.subheader("Ask me anything about the DPDP Act/IT Act/GDPR!")


# 3. CLEAN TEXT

def clean_text(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if (
            len(line) < 30 or
            line.startswith("Subs.") or
            line.startswith("Ins.") or
            "Official Journal" in line or
            "w.e.f" in line
        ):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# 4. INIT VECTOR DB

@st.cache_resource
def init_vector_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load existing DB if available
    if os.path.exists("db"):
        return Chroma(
            persist_directory="db",
            embedding_function=embeddings
        )

    docs = []

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/{file}")
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = file.lower()

            docs.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )

    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="db"
    )

    vectorstore.persist()
    return vectorstore


vector_db = init_vector_db()


# 5. MODEL SETUP

## IN CASE MODEL IS ABOUT TO REACH LIMIT (see https://aistudio.google.com/rate-limit?timeRange=last-28-days&project=gen-lang-client-0927837118)
## CHANGE TO ANY ONE OF THE FOLLOWING: (to be tested still)
## gemini-2.5-flash-lite/gemini-2.5-flash/gemini-3-flash-preview/<more to be added>

llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.1-flash-lite-preview",
    temperature=0
)


# 6. PROMPT (now includes {history} for contextual memory)

prompt = ChatPromptTemplate.from_template("""
You are a legal compliance assistant specializing in DPDP Act 2023, GDPR, and IT Act 2000.

Previous conversation (for context):
{history}

Use the provided context to answer the question.

Context:
{context}

Question: {question}

If relevant context is found:
- Answer clearly and concisely
- Use bullet points where helpful

If only partial information is available:
- Answer based on what is available
- Mention that the answer is partial

If no relevant context is found:
- Say: "I could not find sufficient information in the documents."
- Then provide a general explanation based on your knowledge
""")


# 7. ANIMATED SPINNER

LOADING_MESSAGES = [
    "Looking at what you said...",
    "Scanning my legal databases...",
    "⚖️->💻...",
    "Taking a nap... (even I need sleep)",
    "...",
    "Overthinking...",
    "Writing...",
    "Just give me onee more second...",
]

def run_with_spinner(fn):
    """
    Runs fn() in a background thread while cycling through
    LOADING_MESSAGES in a placeholder. Returns the result of fn().
    """
    placeholder = st.empty()
    result = {}
    exception = {}

    def target():
        try:
            result["value"] = fn()
        except Exception as e:
            exception["error"] = e

    thread = threading.Thread(target=target)
    thread.start()

    i = 0
    while thread.is_alive():
        placeholder.info(LOADING_MESSAGES[i % len(LOADING_MESSAGES)])
        time.sleep(0.9)
        i += 1

    thread.join()
    placeholder.empty()

    if "error" in exception:
        raise exception["error"]

    return result["value"]


# 8. RELEVANCE CHECK (Hybrid)

COMPLIANCE_KEYWORDS = [
    "dpdp", "gdpr", "it act", "data protection", "privacy", "consent",
    "penalty", "fine", "breach", "fiduciary", "principal", "processor",
    "compliance", "regulation", "law", "right", "obligation", "notice",
    "personal data", "sensitive data", "retention", "transfer", "audit",
    "data fiduciary", "data principal", "information technology", "meity",
    "cyber", "grievance", "significant fiduciary", "cross border"
]

def is_compliance_query(query: str) -> bool:
    """
    Hybrid check:
    1. Fast keyword scan first.
    2. If no keywords match, ask the LLM to decide.
    """
    q = query.lower()

    # Step 1: Keyword check (instant, free)
    if any(keyword in q for keyword in COMPLIANCE_KEYWORDS):
        return True

    # Step 2: LLM fallback for ambiguous queries (e.g. "what is a notice?")
    check = llm.invoke(
        f"""You are a classifier. Is the following question related to data protection laws 
such as DPDP Act, GDPR, IT Act, or general legal compliance topics?

Answer only YES or NO. No explanation.

Question: {query}"""
    )
    if isinstance(check.content, list):
        answer = check.content[0].get("text", "")
    else:
        answer = check.content

    answer = answer.strip().upper()
    return answer.startswith("YES")


# 9. SOURCE DETECTION

def detect_sources(query):
    q = query.lower()
    sources = []

    if any(term in q for term in ["gdpr", "eu regulation", "europe data"]):
        sources.append("gdpr")

    if any(term in q for term in ["dpdp", "data protection india", "dpdp act"]):
        sources.append("dpdp")

    if any(term in q for term in [
        "it act", "information technology", "cyber law", "it law", "technology law"
    ]):
        sources.append("it")

    return sources


# 10. QUERY EXPANSION

def expand_query(query: str):
    q = query.lower()
    expanded = [query]

    expanded.extend([
        query + " data protection law",
        query + " dpdp act",
        query + " gdpr",
        query + " information technology act"
    ])

    if "compare" in q:
        expanded.append(query + " comparison differences")

    if "penalty" in q or "fine" in q:
        expanded.append(query + " penalties fines punishment")

    if "consent" in q:
        expanded.append(query + " user consent requirements law")

    return list(set(expanded))  # If empty, system searches through all documents


# 11. RETRIEVAL

def get_relevant_docs(query):
    sources = detect_sources(query)
    expanded_queries = expand_query(query)
    all_docs = []

    for q in expanded_queries:
        docs = vector_db.similarity_search(q, k=6)
        all_docs.extend(docs)

    # Deduplicate
    seen = set()
    unique_docs = []

    for d in all_docs:
        content = d.page_content.strip()
        if content not in seen:
            unique_docs.append(d)
            seen.add(content)

    if len(sources) >= 2:
        return unique_docs[:12]

    if sources:
        filtered = [
            d for d in unique_docs
            if any(src in d.metadata.get("source", "") for src in sources)
        ]
        return filtered[:12] if filtered else unique_docs[:12]

    return unique_docs[:12]


# 12. LAW CITATION MAP
# Maps filename fragments → human-readable law name + emoji

LAW_CITATION_MAP = {
    "dpdp":     ("📘", "Digital Personal Data Protection Act, 2023"),
    "gdpr":     ("🇪🇺", "General Data Protection Regulation (GDPR)"),
    "it_act":   ("💻", "Information Technology Act, 2000"),
    "certin":   ("🛡️", "CERT-In Directions, 2022"),
    "rbi":      ("🏦", "RBI Digital Payment Security Guidelines"),
    "dpdp_rules": ("📗", "DPDP Rules, 2025"),
}

def friendly_citation(filename: str, page) -> str:
    """
    Converts a raw filename + page into a human-readable law citation.
    Falls back to the raw filename if no match is found.
    """
    fname = filename.lower()
    for key, (emoji, law_name) in LAW_CITATION_MAP.items():
        if key in fname:
            return f"{emoji} {law_name} — p. {page}"
    # Fallback: clean up the filename a bit
    display = fname.replace("_", " ").replace(".pdf", "").title()
    return f"📄 {display} — p. {page}"


# 13. CONTEXTUAL MEMORY BUILDER
# Injects the last N Q&A turns into the prompt as plain text

def build_history_string(chat_history: list, n: int = 3) -> str:
    """
    Takes the last n user+assistant pairs from chat_history
    and formats them as a readable conversation block.
    Returns empty string if no history yet.
    """
    # Filter to only chat turns (not audit results)
    chat_turns = [m for m in chat_history if m["role"] in ("user", "assistant")]

    # Take the last n*2 messages (n pairs)
    recent = chat_turns[-(n * 2):]

    if not recent:
        return "No prior conversation."

    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


# 14. SESSION STATE INIT

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ..., "sources": [...]}


# 15. UI (Not the Upendra movie)

# --- Render existing chat history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Re-render sources expander for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for citation in msg["sources"]:
                    st.markdown(f"- {citation}")

# --- Input row ---
col1, col2 = st.columns([5, 2])
with col1:
    user_query = st.text_area(
        "💬 Ask",
        height=175,
        placeholder="Ask a question or upload a policy..."
    )

with col2:
    st.markdown("<div style='padding-top: 6px;'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📄 Audit",
        type="pdf",
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --- Clear history button ---
if st.session_state.chat_history:
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

if user_query or uploaded_file:

    current_history = st.session_state.chat_history.copy()

    # Bundle the entire pipeline into one function so
    # a single spinner covers all three stages seamlessly
    def full_pipeline():

        # 📄 AUDITOR MODE
        if uploaded_file:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            policy_text = "\n".join([d.page_content for d in docs])

            checklist = """
    - User consent is obtained
    - Purpose of data collection is specified
    - Data retention policy is defined
    - User rights are mentioned
    - Grievance officer/contact provided
    - Data security measures described
    """

            audit_prompt = f"""
    You are a DPDP compliance auditor.

    Checklist:
    {checklist}

    Privacy Policy:
    {policy_text}

    For each checklist item:
    - Mark: COMPLIANT / PARTIAL / MISSING
    - Give a short reason

    Then provide:
    - Overall compliance score (0-100)
    - Key risks
    """

            result = llm.invoke(audit_prompt)

            return {
                "mode": "audit",
                "response": result.content
            }

        # 💬 CHAT MODE
        compliant = is_compliance_query(user_query)
        if not compliant:
            return {"mode": "invalid"}

        docs = get_relevant_docs(user_query)
        context = "\n\n".join([d.page_content for d in docs])

        # Build history string for contextual memory
        history = build_history_string(current_history)

        response = llm.invoke(prompt.format(
            context=context,
            question=user_query,
            history=history
        ))

        return {
            "mode": "chat",
            "response": response,
            "docs": docs
        }

    result = run_with_spinner(full_pipeline)

    if result["mode"] == "invalid":
        st.warning("Wait... what does this have to do with compliance?")

    elif result["mode"] == "audit":
        response_text = result["response"]
        if isinstance(response_text, list):
            response_text = response_text[0].get("text", "")

        with st.chat_message("assistant"):
            st.subheader("📊 Compliance Auditor Report")
            st.write(response_text)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"📊 **Compliance Audit Result:**\n\n{response_text}",
            "sources": []
        })

    elif result["mode"] == "chat":
        # Extract answer text
        try:
            answer_text = result["response"].content[0]["text"]
        except:
            answer_text = result["response"].content

        # Build law citations from retrieved docs
        seen_citations = set()
        citations = []
        for d in result["docs"]:
            source = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", "?")
            citation = friendly_citation(source, page)
            if citation not in seen_citations:
                seen_citations.add(citation)
                citations.append(citation)

        # Render new user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Render new assistant message
        with st.chat_message("assistant"):
            st.markdown(answer_text)
            if citations:
                with st.expander("📄 Sources"):
                    for citation in sorted(citations):
                        st.markdown(f"- {citation}")

        # Persist to session state
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
            "sources": []
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "sources": citations
        })
