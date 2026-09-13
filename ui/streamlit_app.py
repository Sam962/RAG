import sys
import uuid
from pathlib import Path

# Streamlit runs this file as a script; project root is not on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from ui.client import DEFAULT_API_URL, health, history_payload, send_chat

st.set_page_config(page_title="RAG Pro", page_icon="📄", layout="centered")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


def _source_label(citation: dict) -> str:
    source = citation.get("source") or "unknown"
    name = Path(str(source)).name
    page = citation.get("page")
    score = citation.get("rerank_score")
    parts = [name]
    if page is not None:
        parts.append(f"page {page}")
    if score is not None:
        parts.append(f"score {float(score):.2f}")
    return " · ".join(parts)


st.title("RAG Pro")
st.caption("Ask questions about your documents. Answers stay grounded in retrieved chunks.")

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.thread_id, language=None)
    api_ok = health()
    st.write("API:", "up" if api_ok else "down — start uvicorn first")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

if not health():
    st.warning(
        f"Cannot reach the API at `{DEFAULT_API_URL}`. "
        "Run `uv run uvicorn api.main:app --reload` in another terminal."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        citations = message.get("citations") or []
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                for citation in citations:
                    st.markdown(f"- {_source_label(citation)}")
                    st.caption(citation.get("source", ""))

prompt = st.chat_input("Ask a question about your documents")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering…"):
            try:
                result = send_chat(
                    prompt,
                    st.session_state.thread_id,
                    chat_history=history_payload(st.session_state.messages[:-1]),
                )
            except Exception as exc:
                st.error(f"Chat request failed: {exc}")
                st.stop()
        answer = result.get("answer") or ""
        citations = result.get("citations") or []
        st.markdown(answer)
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                for citation in citations:
                    st.markdown(f"- {_source_label(citation)}")
                    st.caption(citation.get("source", ""))
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations}
        )
