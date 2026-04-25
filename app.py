import streamlit as st
from backend import analyze_article

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🗞️",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem; border-radius: 20px;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800;
    color: #fff; margin: 0 0 0.5rem;
}
.hero h1 span { color: #6366f1; }
.hero p { color: #94a3b8; font-size: 15px; margin: 0; }

.badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc; font-size: 12px;
    padding: 4px 12px; border-radius: 100px;
    margin-bottom: 1rem;
}

.verdict-fake {
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
    border-radius: 14px; padding: 1.25rem 1.5rem;
    text-align: center; font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800; color: #dc2626;
}
.verdict-real {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    border-radius: 14px; padding: 1.25rem 1.5rem;
    text-align: center; font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800; color: #16a34a;
}
.verdict-uncertain {
    background: #fffbeb;
    border: 1.5px solid #fde68a;
    border-radius: 14px; padding: 1.25rem 1.5rem;
    text-align: center; font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800; color: #d97706;
}

.phrase-tag {
    display: inline-block;
    background: #fef3c7; color: #92400e;
    border: 1px solid #fde68a;
    font-size: 13px; padding: 4px 12px;
    border-radius: 6px; margin: 3px;
}

.section-label {
    font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.06em; color: #6b7280;
    font-weight: 500; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <div class="badge">⚡ AI-Powered Detection</div>
    <h1>Fake News<br><span>Detector</span></h1>
    <p>Hybrid AI combining BERT classification with Gemma3 reasoning</p>
</div>
""", unsafe_allow_html=True)

# Model tags
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🤖 BERT Fine-tuned")
with col2:
    st.info("🦙 Gemma3:4b")
with col3:
    st.info("⚡ 99.8% Accuracy")

st.markdown("---")

# Input
st.markdown("### 📰 Paste Your News Article")
article = st.text_area(
    label="",
    placeholder="Paste any news article or headline here...",
    height=180
)

if st.button("🔍 Analyze Article", use_container_width=True):
    if not article.strip():
        st.warning("⚠️ Please paste a news article first!")
    else:
        with st.spinner("🤖 Analyzing with BERT + Gemma3..."):
            result = analyze_article(article)

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # Verdict box
        if "FAKE" in result['final_verdict']:
            st.markdown(
                f'<div class="verdict-fake">❌ {result["final_verdict"]}</div>',
                unsafe_allow_html=True
            )
        elif "REAL" in result['final_verdict']:
            st.markdown(
                f'<div class="verdict-real">✅ {result["final_verdict"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="verdict-uncertain">⚠️ {result["final_verdict"]}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🤖 BERT Verdict", result['bert_verdict'])
        with col2:
            st.metric("📈 Confidence", result['bert_confidence'])
        with col3:
            agreement_short = "HIGH" if "HIGH" in result['agreement'] else "LOW"
            st.metric("🤝 Agreement", agreement_short)

        st.markdown("---")

        # Explanation
        st.markdown("### 🦙 Gemma3 Explanation")
        st.markdown(result['llm_explanation'])

        st.markdown("---")
        st.success("✅ Analysis Complete!")

# Footer
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:12px; margin-top:2rem;'>"
    "Built with BERT + Gemma3 · For educational purposes only</div>",
    unsafe_allow_html=True
)