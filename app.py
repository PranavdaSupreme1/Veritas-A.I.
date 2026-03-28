# 1. IMPORTS
import streamlit as st
import os
import time
import threading
import tempfile
import json
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# 2. CONFIG

os.environ["GOOGLE_API_KEY"] = ""  # Don't commit — bots crawl git for exposed keys.

st.set_page_config(page_title="DPDP Compliance Bot", layout="wide")
st.image("logo.png", width=250)
st.subheader("Ask me anything about the DPDP Act / IT Act / GDPR!")
st.divider()


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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("db"):
        return Chroma(persist_directory="db", embedding_function=embeddings)
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/{file}")
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = file.lower()
            docs.extend(loaded_docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="db")
    vectorstore.persist()
    return vectorstore

vector_db = init_vector_db()


# 5. MODEL SETUP

## Fallback models: gemini-2.5-flash-lite / gemini-2.5-flash / gemini-3-flash-preview
llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite-preview", temperature=0)


# 6. CHAT PROMPT

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
    q = query.lower()
    if any(keyword in q for keyword in COMPLIANCE_KEYWORDS):
        return True
    check = llm.invoke(
        f"""You are a classifier. Is the following question related to data protection laws 
such as DPDP Act, GDPR, IT Act, or general legal compliance topics?
Answer only YES or NO. No explanation.
Question: {query}"""
    )
    answer = check.content if isinstance(check.content, str) else check.content[0].get("text", "")
    return answer.strip().upper().startswith("YES")


# 9. SOURCE DETECTION

def detect_sources(query):
    q = query.lower()
    sources = []
    if any(term in q for term in ["gdpr", "eu regulation", "europe data"]):
        sources.append("gdpr")
    if any(term in q for term in ["dpdp", "data protection india", "dpdp act"]):
        sources.append("dpdp")
    if any(term in q for term in ["it act", "information technology", "cyber law", "it law", "technology law"]):
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
    return list(set(expanded))


# 11. RETRIEVAL

def get_relevant_docs(query):
    sources = detect_sources(query)
    expanded_queries = expand_query(query)
    all_docs = []
    for q in expanded_queries:
        docs = vector_db.similarity_search(q, k=6)
        all_docs.extend(docs)
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
        filtered = [d for d in unique_docs if any(src in d.metadata.get("source", "") for src in sources)]
        return filtered[:12] if filtered else unique_docs[:12]
    return unique_docs[:12]


# 12. LAW CITATION MAP (for chat tab)

LAW_CITATION_MAP = {
    "dpdp_rules": ("📗", "DPDP Rules, 2025"),
    "dpdp":       ("📘", "Digital Personal Data Protection Act, 2023"),
    "gdpr":       ("🇪🇺", "General Data Protection Regulation (GDPR)"),
    "it_act":     ("💻", "Information Technology Act, 2000"),
    "certin":     ("🛡️", "CERT-In Directions, 2022"),
    "rbi":        ("🏦", "RBI Digital Payment Security Guidelines"),
}

def friendly_citation(filename: str, page) -> str:
    fname = filename.lower()
    for key, (emoji, law_name) in LAW_CITATION_MAP.items():
        if key in fname:
            return f"{emoji} {law_name} — p. {page}"
    display = fname.replace("_", " ").replace(".pdf", "").title()
    return f"📄 {display} — p. {page}"


# 13. CLAUSE LINK MAP
# Maps each checklist item key → (section label, official URL)
# URLs point to the live official text so users can read the exact clause.

CLAUSE_LINKS = {
    # ── DPDP Act, 2023 ──────────────────────────────────────────────────────
    "dpdp_consent":        ("Section 6",  "https://www.indiacode.nic.in/bitstream/123456789/19420/1/dpdp_act_2023.pdf#page=8"),
    "dpdp_notice":         ("Section 5",  "https://www.indiacode.nic.in/bitstream/123456789/19420/1/dpdp_act_2023.pdf#page=7"),
    "dpdp_retention":      ("Section 8(7)","https://www.indiacode.nic.in/bitstream/123456789/19420/1/dpdp_act_2023.pdf#page=10"),
    "dpdp_rights":         ("Sections 11-12","https://www.indiacode.nic.in/bitstream/123456789/19420/1/dpdp_act_2023.pdf#page=12"),
    "dpdp_grievance":      ("Section 13", "https://www.indiacode.nic.in/bitstream/123456789/19420/1/dpdp_act_2023.pdf#page=13"),
    "dpdp_security":       ("Section 8(5)","https://www.indiacode.nic.in/bitstream/123456789/19420/1/dpdp_act_2023.pdf#page=10"),
    "dpdp_cross_border":   ("Section 16", "https://www.indiacode.nic.in/bitstream/123456789/19420/1/dpdp_act_2023.pdf#page=15"),
    "dpdp_breach":         ("Section 8(6)","https://www.indiacode.nic.in/bitstream/123456789/19420/1/dpdp_act_2023.pdf#page=10"),

    # ── GDPR ────────────────────────────────────────────────────────────────
    "gdpr_lawful_basis":   ("Article 6",  "https://gdpr-info.eu/art-6-gdpr/"),
    "gdpr_rights":         ("Articles 15-22","https://gdpr-info.eu/art-15-gdpr/"),
    "gdpr_dpo":            ("Article 37", "https://gdpr-info.eu/art-37-gdpr/"),
    "gdpr_cookies":        ("Article 7 + ePrivacy","https://gdpr-info.eu/art-7-gdpr/"),
    "gdpr_retention":      ("Article 5(1)(e)","https://gdpr-info.eu/art-5-gdpr/"),
    "gdpr_third_party":    ("Article 28", "https://gdpr-info.eu/art-28-gdpr/"),
    "gdpr_transfer":       ("Articles 44-49","https://gdpr-info.eu/art-44-gdpr/"),
    "gdpr_privacy_design": ("Article 25", "https://gdpr-info.eu/art-25-gdpr/"),

    # ── IT Act, 2000 ────────────────────────────────────────────────────────
    "it_spdi":             ("Section 43A + Rule 3","https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf#page=28"),
    "it_security":         ("Section 43A + Rule 8","https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf#page=28"),
    "it_consent":          ("Rule 5(1)",  "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf#page=30"),
    "it_disclosure":       ("Rule 6",     "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf#page=31"),
    "it_grievance":        ("Rule 5(9)",  "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf#page=30"),
    "it_purpose":          ("Rule 5(2)",  "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf#page=30"),
}

# Maps each checklist item description → its CLAUSE_LINKS key
# This is what the LLM will reference in its JSON output
CHECKLIST_KEY_MAP = {
    # DPDP
    "User consent is explicitly obtained before data collection":           "dpdp_consent",
    "Purpose of data collection is clearly specified":                      "dpdp_notice",
    "Data retention period / deletion policy is defined":                   "dpdp_retention",
    "User rights (access, correction, erasure, nomination) are mentioned":  "dpdp_rights",
    "Grievance officer name and contact details are provided":              "dpdp_grievance",
    "Data security measures are described":                                 "dpdp_security",
    "Cross-border data transfer restrictions are addressed":               "dpdp_cross_border",
    "Data breach notification procedure is mentioned":                      "dpdp_breach",
    # GDPR
    "Lawful basis for processing is stated":                               "gdpr_lawful_basis",
    "Data subject rights (access, rectification, erasure, portability) are covered": "gdpr_rights",
    "Data Protection Officer (DPO) contact is provided":                   "gdpr_dpo",
    "Cookie consent and tracking disclosure is present":                   "gdpr_cookies",
    "Data retention periods are specified":                                "gdpr_retention",
    "Third-party data sharing and processors are disclosed":               "gdpr_third_party",
    "International data transfer safeguards are described":                "gdpr_transfer",
    "Privacy by design / default principles are mentioned":                "gdpr_privacy_design",
    # IT Act
    "Sensitive personal data or information (SPDI) is identified":         "it_spdi",
    "Reasonable security practices (ISO 27001 or equivalent) are mentioned": "it_security",
    "User consent for SPDI collection is obtained":                        "it_consent",
    "Disclosure of SPDI to third parties is addressed":                    "it_disclosure",
    "Grievance redressal mechanism is provided":                           "it_grievance",
    "Data collection only for lawful purpose is stated":                   "it_purpose",
}


# 14. AUDIT CHECKLISTS

LAW_CHECKLISTS = {
    "DPDP Act, 2023": [
        "User consent is explicitly obtained before data collection",
        "Purpose of data collection is clearly specified",
        "Data retention period / deletion policy is defined",
        "User rights (access, correction, erasure, nomination) are mentioned",
        "Grievance officer name and contact details are provided",
        "Data security measures are described",
        "Cross-border data transfer restrictions are addressed",
        "Data breach notification procedure is mentioned",
    ],
    "GDPR": [
        "Lawful basis for processing is stated",
        "Data subject rights (access, rectification, erasure, portability) are covered",
        "Data Protection Officer (DPO) contact is provided",
        "Cookie consent and tracking disclosure is present",
        "Data retention periods are specified",
        "Third-party data sharing and processors are disclosed",
        "International data transfer safeguards are described",
        "Privacy by design / default principles are mentioned",
    ],
    "IT Act, 2000": [
        "Sensitive personal data or information (SPDI) is identified",
        "Reasonable security practices (ISO 27001 or equivalent) are mentioned",
        "User consent for SPDI collection is obtained",
        "Disclosure of SPDI to third parties is addressed",
        "Grievance redressal mechanism is provided",
        "Data collection only for lawful purpose is stated",
    ],
}


# 15. KNOWN COLLISIONS BETWEEN LAWS
# Each entry: (topic, law_a, law_a_stance, law_b, law_b_stance)
# The LLM will also detect dynamic ones, but these are always shown when both laws are selected.

KNOWN_COLLISIONS = [
    {
        "topic": "Legal basis for data processing",
        "laws": ["DPDP Act, 2023", "GDPR"],
        "dpdp": "Relies primarily on **consent** as the legal basis, with limited 'legitimate uses' (Section 5). No concept of 'legitimate interests'.",
        "gdpr":  "Provides **six** legal bases (Article 6), including legitimate interests, legal obligation, vital interests — consent is just one option.",
        "risk":  "A policy drafted for GDPR that relies on legitimate interests will have **no valid legal basis** under DPDP."
    },
    {
        "topic": "Grievance redressal timeline",
        "laws": ["DPDP Act, 2023", "IT Act, 2000"],
        "dpdp": "Grievances must be resolved within **90 days** (Section 13 + Rules).",
        "it":    "No fixed statutory timeline specified under IT Act Rules.",
        "risk":  "A policy that sets an internal SLA longer than 90 days may comply with IT Act but violate DPDP."
    },
    {
        "topic": "Definition of sensitive personal data",
        "laws": ["GDPR", "IT Act, 2000"],
        "gdpr":  "Lists **special categories** (Art. 9): racial/ethnic origin, health, biometric, genetic, sexual orientation, etc.",
        "it":    "Defines **SPDI** (Rule 3): passwords, financial info, health, sexual orientation, biometric — partial overlap but narrower.",
        "risk":  "Biometric data is sensitive under both, but racial/ethnic origin is sensitive under GDPR only — a single policy may under-protect it for GDPR."
    },
    {
        "topic": "Data localisation / cross-border transfers",
        "laws": ["DPDP Act, 2023", "GDPR"],
        "dpdp": "Transfers allowed to countries notified by Central Government (Section 16). List not yet published — currently permissive.",
        "gdpr":  "Transfers require adequacy decision, SCCs, BCRs, or other Chapter V safeguards (Art. 44-49).",
        "risk":  "A policy with a blanket 'we may transfer data globally' clause may be fine under current DPDP but illegal under GDPR."
    },
]


# 16. AUDIT PROMPT BUILDER (JSON output)

def build_audit_prompt(policy_text: str, selected_laws: list) -> str:
    sections = []
    all_keys = []
    for law in selected_laws:
        items = LAW_CHECKLISTS.get(law, [])
        keys = [CHECKLIST_KEY_MAP.get(item, "") for item in items]
        all_keys.extend(keys)
        items_with_keys = "\n".join(
            f'  - item: "{item}", key: "{CHECKLIST_KEY_MAP.get(item, "")}"'
            for item in items
        )
        sections.append(f"Law: {law}\n{items_with_keys}")

    checklist_block = "\n\n".join(sections)
    laws_str = ", ".join(selected_laws)

    collision_instruction = ""
    if len(selected_laws) >= 2:
        collision_instruction = """
"collisions": [
    // List any topics where the selected laws give CONTRADICTORY requirements for this policy.
    // Only include genuine contradictions found in the policy text, not hypothetical ones.
    // Format each as:
    {
      "topic": "...",
      "description": "One sentence explaining the contradiction found in this policy."
    }
  ],
"""

    return f"""You are a strict compliance auditor. Analyze the privacy policy against: {laws_str}.

CHECKLIST (each item has a key you MUST use exactly):
{checklist_block}

PRIVACY POLICY:
{policy_text}

OUTPUT RULES:
- Respond with ONLY valid JSON. No markdown, no code fences, no explanation outside the JSON.
- Use exactly the key strings provided — do not invent new keys.

OUTPUT FORMAT:
{{
  "violations": [
    {{
      "law": "DPDP Act, 2023",
      "key": "dpdp_consent",
      "item": "User consent is explicitly obtained before data collection",
      "status": "MISSING",
      "reason": "One sentence explaining why."
    }}
    // status must be: MISSING or PARTIAL
    // Only include items that are MISSING or PARTIAL — omit compliant items
  ],
  {collision_instruction if len(selected_laws) >= 2 else '"collisions": [],'}
  "scores": {{
    "DPDP Act, 2023": 72,
    "GDPR": 55,
    "overall": 64
  }},
  "suggestions": [
    "Most critical fix first — one sentence.",
    "Second fix.",
    "Third fix.",
    "Fourth fix (optional).",
    "Fifth fix (optional)."
  ]
}}

Score strictly. Missing a grievance officer = -10. Missing consent = -20. Partial items = -5 each.
Only include laws that were actually selected: {laws_str}.
"""


# 17. CONTEXTUAL MEMORY BUILDER

def build_history_string(chat_history: list, n: int = 3) -> str:
    chat_turns = [m for m in chat_history if m["role"] in ("user", "assistant")]
    recent = chat_turns[-(n * 2):]
    if not recent:
        return "No prior conversation."
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )


# 18. RENDER AUDIT REPORT
# Parses the JSON from the LLM and renders it with red dots + clause links

def render_audit_report(raw: str, selected_laws: list, known_collisions: list):

    # --- Parse JSON safely ---
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)
    except Exception:
        st.error("⚠️ Could not parse audit response as JSON. Raw output below:")
        st.code(raw)
        return

    violations = data.get("violations", [])
    collisions  = data.get("collisions", [])
    scores      = data.get("scores", {})
    suggestions = data.get("suggestions", [])

    # ── Section 1: Violations ────────────────────────────────────────────
    st.markdown("## 🔴 Violations & Weaknesses by Law")

    violations_by_law = {}
    for v in violations:
        violations_by_law.setdefault(v["law"], []).append(v)

    for law in selected_laws:
        law_violations = violations_by_law.get(law, [])
        st.markdown(f"### {law}")
        if not law_violations:
            st.success("✅ No violations found — fully compliant.")
        else:
            for v in law_violations:
                key      = v.get("key", "")
                status   = v.get("status", "MISSING")
                item     = v.get("item", "")
                reason   = v.get("reason", "")
                dot      = "🔴" if status == "MISSING" else "🟡"

                # Look up clause link
                clause_info = CLAUSE_LINKS.get(key)
                if clause_info:
                    section_label, url = clause_info
                    link_html = (
                        f'<a href="{url}" target="_blank" '
                        f'style="font-size:0.75rem; margin-left:8px; '
                        f'color:#888; text-decoration:underline dotted;">'
                        f'↗ {section_label}</a>'
                    )
                else:
                    link_html = ""

                st.markdown(
                    f"{dot} **{status}** — {item}{link_html}<br>"
                    f"<span style='color:grey; font-size:0.85rem; margin-left:1.5rem;'>{reason}</span>",
                    unsafe_allow_html=True
                )
        st.divider()

    # ── Section 2: Collision Warnings ───────────────────────────────────
    all_collisions = []

    # Always-show known collisions for selected law pairs
    for c in known_collisions:
        if all(law in selected_laws for law in c["laws"]):
            all_collisions.append({
                "topic": c["topic"],
                "description": c["risk"],
                "source": "known"
            })

    # LLM-detected collisions (dynamic, policy-specific)
    for c in collisions:
        all_collisions.append({
            "topic": c.get("topic", ""),
            "description": c.get("description", ""),
            "source": "llm"
        })

    if all_collisions:
        st.markdown("## ⚡ Law Collisions Detected")
        st.info(
            "The selected laws give **contradictory requirements** on the following topics. "
            "You may need to choose a stricter standard or add law-specific clauses."
        )
        for c in all_collisions:
            badge = "📌 Known conflict" if c["source"] == "known" else "🤖 Policy-specific conflict"
            with st.expander(f"⚡ {c['topic']} — {badge}"):
                st.markdown(c["description"])
        st.divider()

    # ── Section 3: Scores ───────────────────────────────────────────────
    st.markdown("## 📊 Compliance Scores")

    score_cols = st.columns(len(selected_laws) + 1)
    for i, law in enumerate(selected_laws):
        score = scores.get(law, "N/A")
        color = "#e74c3c" if isinstance(score, int) and score < 60 else \
                "#f39c12" if isinstance(score, int) and score < 80 else "#27ae60"
        score_cols[i].markdown(
            f"<div style='text-align:center; padding:12px; border-radius:8px; "
            f"background:{color}22; border:1px solid {color};'>"
            f"<div style='font-size:2rem; font-weight:bold; color:{color};'>{score}</div>"
            f"<div style='font-size:0.8rem; color:grey;'>{law}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    overall = scores.get("overall", "N/A")
    o_color = "#e74c3c" if isinstance(overall, int) and overall < 60 else \
              "#f39c12" if isinstance(overall, int) and overall < 80 else "#27ae60"
    score_cols[-1].markdown(
        f"<div style='text-align:center; padding:12px; border-radius:8px; "
        f"background:{o_color}22; border:2px solid {o_color};'>"
        f"<div style='font-size:2rem; font-weight:bold; color:{o_color};'>{overall}</div>"
        f"<div style='font-size:0.8rem; color:grey;'>Overall</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.divider()

    # ── Section 4: Suggestions ──────────────────────────────────────────
    st.markdown("## 💡 Suggestions for Improvement")
    for i, s in enumerate(suggestions, 1):
        st.markdown(f"**{i}.** {s}")


# 19. SESSION STATE INIT

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# 20. UI — TABS

tab_chat, tab_audit = st.tabs(["💬 Chat", "📋 Compliance Audit"])


# ─────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────
with tab_chat:

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Sources"):
                    for citation in msg["sources"]:
                        st.markdown(f"- {citation}")

    user_query = st.text_area(
        "💬 Ask a compliance question",
        height=120,
        placeholder="e.g. What are the penalties under DPDP Act for a data breach?"
    )

    _, col_clear = st.columns([4, 1])
    with col_clear:
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_history = []
                st.rerun()

    if user_query:
        current_history = st.session_state.chat_history.copy()

        def chat_pipeline():
            if not is_compliance_query(user_query):
                return {"mode": "invalid"}
            docs = get_relevant_docs(user_query)
            context = "\n\n".join([d.page_content for d in docs])
            history = build_history_string(current_history)
            response = llm.invoke(prompt.format(context=context, question=user_query, history=history))
            return {"mode": "chat", "response": response, "docs": docs}

        result = run_with_spinner(chat_pipeline)

        if result["mode"] == "invalid":
            st.warning("Wait... what does this have to do with compliance?")
        elif result["mode"] == "chat":
            try:
                answer_text = result["response"].content[0]["text"]
            except:
                answer_text = result["response"].content

            seen_citations = set()
            citations = []
            for d in result["docs"]:
                cit = friendly_citation(d.metadata.get("source", "Unknown"), d.metadata.get("page", "?"))
                if cit not in seen_citations:
                    seen_citations.add(cit)
                    citations.append(cit)

            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                st.markdown(answer_text)
                if citations:
                    with st.expander("📄 Sources"):
                        for cit in sorted(citations):
                            st.markdown(f"- {cit}")

            st.session_state.chat_history.append({"role": "user", "content": user_query, "sources": []})
            st.session_state.chat_history.append({"role": "assistant", "content": answer_text, "sources": citations})


# ─────────────────────────────────────────────
# TAB 2: AUDIT
# ─────────────────────────────────────────────
with tab_audit:

    st.markdown("### 📋 Policy Compliance Auditor")
    st.markdown("Upload a privacy policy PDF and select which laws to audit against.")
    st.divider()

    col_upload, col_laws = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader("📄 Upload Privacy Policy (PDF)", type="pdf")

    with col_laws:
        st.markdown("**Audit against:**")
        audit_dpdp = st.checkbox("📘 DPDP Act, 2023", value=True)
        audit_gdpr = st.checkbox("🇪🇺 GDPR")
        audit_it   = st.checkbox("💻 IT Act, 2000")

    selected_laws = []
    if audit_dpdp: selected_laws.append("DPDP Act, 2023")
    if audit_gdpr: selected_laws.append("GDPR")
    if audit_it:   selected_laws.append("IT Act, 2000")

    st.divider()

    if uploaded_file:
        if not selected_laws:
            st.warning("⚠️ Please select at least one law to audit against.")
        else:
            st.info(f"Ready to audit against: **{', '.join(selected_laws)}**")

            if st.button("🔍 Run Audit"):
                file_bytes       = uploaded_file.read()
                laws_for_thread  = selected_laws.copy()

                def audit_pipeline():
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    policy_text = "\n".join([d.page_content for d in docs])
                    audit_prompt_text = build_audit_prompt(policy_text, laws_for_thread)
                    response = llm.invoke(audit_prompt_text)
                    raw = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
                    return {"mode": "audit", "response": raw, "laws": laws_for_thread}

                result = run_with_spinner(audit_pipeline)

                st.divider()
                st.markdown(f"## 📊 Audit Report")
                st.markdown(f"*Laws audited: {', '.join(result['laws'])}*")
                st.divider()

                render_audit_report(result["response"], result["laws"], KNOWN_COLLISIONS)

    else:
        st.markdown(
            "<div style='text-align:center; color:grey; padding:40px 0;'>"
            "⬆️ Upload a PDF above to get started."
            "</div>",
            unsafe_allow_html=True
        )
