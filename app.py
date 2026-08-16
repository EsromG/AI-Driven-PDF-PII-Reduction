# app.py
import os
import tempfile
import streamlit as st

from pdf_utils import PDFProcessor

# Page Config

st.set_page_config(
    page_title="AI-Driven PDF PII Redaction System",
    page_icon="🛡️",
    layout="wide"
)

# CSS (will adjust later)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
      .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(59,130,246,0.14), rgba(16,185,129,0.12));
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 1rem;
      }
      .hero h1 { margin: 0; font-size: 2.0rem; }
      .hero p { margin: 0.35rem 0 0; opacity: 0.9; font-size: 1.0rem; }
      .card {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
      }
      .small-note { font-size: 0.9rem; opacity: 0.85; }
      .footer { opacity: 0.65; font-size: 0.85rem; margin-top: 1.5rem; }
      div.stDownloadButton > button {
        border-radius: 12px;
        padding: 0.65rem 1rem;
        font-weight: 600;
      }
      div.stButton > button {
        border-radius: 12px;
        padding: 0.65rem 1rem;
        font-weight: 700;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Cache model load

@st.cache_resource
def get_processor():
    return PDFProcessor()

processor = get_processor()

# Sidebar
with st.sidebar:
    st.markdown("🛡️ AI PII Redaction")
    st.caption("AI-Driven PDF PII Redaction System")

    st.markdown("---")
    st.markdown("### Workflow")
    st.write("1) Upload PDF")
    st.write("2) Redact PII")
    st.write("3) Download redacted PDF")

    

# Hero Header

st.markdown(
    """
    <div class="hero">
      <h1>🧹 AI-Driven PDF PII Redaction System</h1>
      <p>Upload a PDF → automatically detect & remove PII → download a safe redacted copy.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main Layout

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Upload a PDF")
    st.write(
        "We detect and redact common PII such as **emails, phone numbers, SSNs/IDs, addresses, and names** "
        "using a hybrid of **regex rules + transformer model**."
    )

    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )


    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✅ What you’ll get")
    st.write("**Output:** A regenerated PDF where detected PII is removed from the content.")
    st.write("**Why it matters:** Redaction is not just visual. PII should not be copyable/searchable.")
    st.markdown("---")
  
    st.markdown("</div>", unsafe_allow_html=True)


# Processing Section

st.markdown("### 🚀 Run Redaction")

if uploaded_pdf is None:
    st.warning("Upload a PDF to enable redaction.")
    st.stop()

# Show file details
file_col1, file_col2, file_col3 = st.columns(3)
file_col1.metric("File name", uploaded_pdf.name)
file_col2.metric("File type", "PDF")

# Save uploaded file to temp
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
    tmp_in.write(uploaded_pdf.read())
    input_path = tmp_in.name

st.success("File received and staged. Ready to redact.")

# Call-to-action button row
cta1, cta2 = st.columns([0.35, 0.65])
with cta1:
    run = st.button("🧹 Redact PDF", use_container_width=True)
with cta2:
    st.caption("This may take a few seconds depending on PDF size and model load.")

if run:
    with st.spinner("Detecting & redacting PII..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
            output_path = tmp_out.name

        # Run redaction
        processor.redact_pdf(input_path, output_path)

    st.balloons()
    st.success("Redaction completed successfully!")

    # Serve the file to the user and ensure temp files are removed afterwards
    try:
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Redacted PDF",
                data=f,
                file_name=f"redacted_{uploaded_pdf.name}",
                mime="application/pdf",
                use_container_width=True
            )

        st.markdown(
            '<p class="small-note">✅ The output PDF is regenerated from sanitized content. '
            'PII is removed (not merely hidden).</p>',
            unsafe_allow_html=True
        )
    finally:
        # Cleanup temporary files containing potentially sensitive data
        for p in (input_path, output_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                # ignore cleanup errors (could log to a file if needed)
                pass
    

