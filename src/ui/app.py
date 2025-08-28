import os
import io
import requests
import streamlit as st
from pdfminer.high_level import extract_text

st.set_page_config(page_title="CV ↔ JD RAG Matcher", layout="wide")
st.title("CV ↔ JD RAG Matcher (LLM-Scored)")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Left: CV PDF upload & preview
left, right = st.columns(2)
with left:
    st.subheader("Upload your CV (PDF)")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
    cv_text = ""
    if uploaded_pdf is not None:
        try:
            # Extract text directly from in-memory bytes
            cv_text = extract_text(io.BytesIO(uploaded_pdf.getvalue()))
            st.success("CV text extracted from PDF ✅")
            # Optional preview (collapsible)
            with st.expander("Preview extracted CV text"):
                st.text(cv_text[:5000] if cv_text else "No text extracted.")
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {e}")
            cv_text = ""

    st.caption("Tip: If your PDF is image-only, consider exporting it as searchable PDF or pasting text.")

# --- Right: JD text area
with right:
    st.subheader("Paste Job Description (Text)")
    jd_text = st.text_area(
        "Paste the JD here (Responsibilities, Requirements, Nice-to-haves, etc.)",
        height=420,
        key="jd_text_area",
        placeholder=(
            "Responsibilities:\n- Build and deploy ML services in production (FastAPI)\n"
            "- Implement RAG-based search with vector DB\n\n"
            "Requirements:\n- Strong Python and Docker\n- Cloud (AWS or GCP)\n- Kubernetes is a plus"
        ),
    )

# Tunables
st.markdown("---")
top_k = st.slider("Top-k retrieved evidence per requirement", 1, 7, 3)
run_btn = st.button("Evaluate CV ↔ JD Match")

# --- Call API
if run_btn:
    if not jd_text:
        st.error("Please paste the Job Description text.")
    elif not cv_text:
        st.error("Please upload a CV PDF so I can extract the text.")
    else:
        with st.spinner("Scoring with RAG + LLM..."):
            try:
                r = requests.post(
                    f"{API_URL}/score_llm",
                    json={"cv_text": cv_text, "jd_text": jd_text, "top_k": top_k},
                    timeout=180,
                )
            except requests.exceptions.RequestException as ex:
                st.error(f"Could not reach API at {API_URL}. Error: {ex}")
                st.stop()

        if not r.ok:
            st.error(f"API error: {r.status_code} — {r.text}")
            st.stop()

        data = r.json()

        # --- Summary metrics
        st.success(f"Overall Match: {data.get('overall_score', 0)} / 100")
        c1, c2, c3 = st.columns(3)
        sec = data.get("section_scores", {})
        c1.metric("Hard Skills", sec.get("hard_skills", 0))
        c2.metric("Experience", sec.get("experience", 0))
        c3.metric("Soft Skills", sec.get("soft_skills", 0))

        # --- Good matches
        st.subheader("Good Matches (Evidence-backed)")
        good_matches = data.get("good_matches", [])
        if good_matches:
            for gm in good_matches[:12]:
                st.markdown(f"**Requirement:** {gm.get('requirement', '')}")
                if gm.get("evidence"):
                    st.markdown(f"> **Evidence:** {gm.get('evidence', '')}")
                if gm.get("reason"):
                    st.caption(gm.get("reason", ""))
                st.divider()
        else:
            st.write("No strong matches detected.")

        # --- Missing requirements
        st.subheader("Missing Requirements")
        missing_reqs = data.get("missing_requirements", [])
        if missing_reqs:
            for mr in missing_reqs[:30]:
                st.write(f"- {mr}")
        else:
            st.write("None detected 🎉")

        # --- Missing / weak skills (LLM + heuristic)
        st.subheader("Missing / Weak Skills")
        miss_sk = data.get("missing_skills", [])
        if miss_sk:
            st.write(", ".join(sorted(set(miss_sk))[:40]))
        else:
            st.write("None detected")

        # --- Improvement suggestions
        st.subheader("Improvement Suggestions")
        sugg = data.get("improvement_suggestions", [])
        if sugg:
            for s in sugg[:12]:
                st.write(f"- {s}")
        else:
            st.write("No suggestions provided by the LLM.")
