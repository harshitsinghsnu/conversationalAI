import streamlit as st
from openai import OpenAI
import os
import re

# ── Constants ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a drug discovery assistant specializing in computational chemistry, "
    "medicinal chemistry, and AI-driven approaches to drug development. "
    "If the query is not related to drug discovery, computational chemistry, or "
    "medicinal chemistry, respond only with: "
    "'I can only assist with drug discovery topics. Please ask a relevant question.'"
    " You do not provide personalized medical advice or patient-specific dosing. "
    "You do not analyze patient data or genomic sequences for clinical decisions."
)

EXAMPLE_QUESTIONS = [
    "What is molecular docking and how is it used in drug discovery?",
    "Explain ADMET properties and why they matter for drug candidates.",
    "How does lead optimization work in medicinal chemistry?",
    "What is the role of AI in target identification?",
    "Describe Lipinski's Rule of Five.",
    "What are allosteric modulators and why are they therapeutically useful?",
]

# Groq reasoning models tried in order; falls back to CoT prompting if all fail
REASONING_MODELS = [
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
]

# Chain-of-thought system prompt — works with any model when native reasoning unavailable
COT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT +
    "\n\nFor every answer, you MUST use this exact format:\n"
    "<think>\n"
    "Step-by-step scientific reasoning: break down the question, recall relevant mechanisms, "
    "consider evidence, and work toward the answer.\n"
    "</think>\n"
    "Your final, clear, concise answer here."
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Drug Discovery AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (light mode) ──────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stSidebar"]          { background: #f1f5f9; border-right: 1px solid #e2e8f0; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(150deg, #eff6ff 0%, #dbeafe 60%, #eff6ff 100%);
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 56px 52px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 16px;
}
.hero::after {
    content: ''; position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 700px; height: 280px;
    background: radial-gradient(ellipse, #93c5fd20 0%, transparent 65%);
    pointer-events: none;
}
.h-eyebrow {
    font-size: 0.67rem; font-weight: 700; letter-spacing: 0.22em;
    text-transform: uppercase; color: #2563eb;
    display: flex; align-items: center; justify-content: center; gap: 14px;
    margin-bottom: 22px; position: relative; z-index: 1;
}
.h-eyebrow::before { content: ''; display: block; height: 1px; width: 60px; background: linear-gradient(90deg, transparent, #bfdbfe); }
.h-eyebrow::after  { content: ''; display: block; height: 1px; width: 60px; background: linear-gradient(90deg, #bfdbfe, transparent); }
.hero h1 {
    font-size: 2.5rem; font-weight: 900; color: #0f172a;
    line-height: 1.13; margin: 0 0 20px; letter-spacing: -0.04em;
    position: relative; z-index: 1;
}
.hero h1 em { font-style: normal; color: #2563eb; }
.h-sub {
    font-size: 1rem; color: #475569; max-width: 620px;
    margin: 0 auto 40px; line-height: 1.82;
    position: relative; z-index: 1;
}
.h-stats {
    display: inline-flex; border-radius: 14px; overflow: hidden;
    border: 1px solid #bfdbfe; background: #ffffff;
    position: relative; z-index: 1;
    box-shadow: 0 1px 4px rgba(37,99,235,0.08);
}
.hst { padding: 14px 28px; text-align: center; border-right: 1px solid #bfdbfe; }
.hst:last-child { border-right: none; }
.hst-v { font-size: 1.65rem; font-weight: 800; color: #2563eb; display: block; line-height: 1; }
.hst-l { font-size: 0.61rem; text-transform: uppercase; letter-spacing: 0.13em; color: #94a3b8; margin-top: 4px; display: block; }

/* ── Section header ── */
.sh { margin: 44px 0 20px; }
.sh-row { display: flex; align-items: center; gap: 11px; margin-bottom: 5px; }
.sh-tag {
    font-size: 0.61rem; font-weight: 800; color: #2563eb;
    letter-spacing: 0.18em; text-transform: uppercase;
    background: #eff6ff; border: 1px solid #bfdbfe;
    padding: 3px 10px; border-radius: 5px; white-space: nowrap;
}
.sh h2 { font-size: 1.2rem; font-weight: 800; color: #0f172a; margin: 0; }
.sh p  { font-size: 0.8rem; color: #64748b; margin: 0; }

/* ── Narrative box ── */
.narrative {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 18px 22px; font-size: 0.86rem; color: #475569;
    line-height: 1.82; margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Methodology cards ── */
.mc {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.mc-icon  { font-size: 1.8rem; margin-bottom: 12px; }
.mc-title { font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.13em; color: #2563eb; margin-bottom: 12px; }
.mc-body  { font-size: 0.79rem; color: #475569; line-height: 1.74; }
.tag {
    display: inline-block; background: #eff6ff; color: #1d4ed8;
    border: 1px solid #bfdbfe; padding: 2px 9px; border-radius: 20px;
    font-size: 0.67rem; margin: 2px 2px 2px 0;
}

/* ── Conclusion tiles ── */
.ct {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 26px 22px; text-align: center; height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.ct-em { font-size: 2rem; margin-bottom: 12px; }
.ct h4  { font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.09em; color: #0f172a; margin-bottom: 11px; }
.ct p   { font-size: 0.79rem; color: #64748b; line-height: 1.72; margin: 0; }

/* ── Pills ── */
.pill { display:inline-block; background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; padding:3px 11px; border-radius:20px; font-size:0.73rem; font-weight:600; margin:2px 3px; }

/* ── Divider ── */
hr.sep { border: none; border-top: 1px solid #e2e8f0; margin: 36px 0; }

/* ── Tab style ── */
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.88rem; padding: 12px 20px; }

/* ── Follow-up suggestion chips ── */
.fu-label { font-size: 0.75rem; color: #94a3b8; margin: 12px 0 6px; font-weight: 500; }
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 20px !important;
    font-size: 0.76rem !important;
    padding: 4px 14px !important;
    font-weight: 500 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 32px !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #dbeafe !important;
    border-color: #93c5fd !important;
}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    font-size: 0.79rem; padding: 6px 10px; text-align: left;
}

/* ── Backend badge ── */
.bknd-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf4; border: 1px solid #86efac;
    border-radius: 8px; padding: 6px 12px; font-size: 0.79rem;
    color: #15803d; margin-top: 4px;
}

/* ── Welcome card ── */
.welcome {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 40px 28px; text-align: center; margin: 20px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_reasoning(text):
    """Extract <think>...</think> block and the final answer."""
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    if "<think>" in text:
        return text[text.find("<think>") + 7:].strip(), ""
    return None, text


def generate_followups(client, model, last_q, last_a):
    """Generate 3 brief follow-up questions after an exchange."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Generate exactly 3 concise follow-up questions (max 9 words each) "
                    "a drug discovery researcher might ask next. "
                    "Return only the questions, one per line, no numbers or bullets."
                )},
                {"role": "user", "content": last_q},
                {"role": "assistant", "content": last_a[:400]},
                {"role": "user", "content": "3 follow-up questions:"},
            ],
            max_tokens=120, temperature=0.7, stream=False,
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        return [l.strip().lstrip("0123456789.-) ") for l in lines if l.strip()][:3]
    except Exception:
        return []


def get_client_and_model():
    groq_key   = st.secrets.get("GROQ_API_KEY",   "") or os.getenv("GROQ_API_KEY",   "")
    gemini_key = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    openai_key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    ollama_url = st.secrets.get("OLLAMA_URL",     "") or os.getenv("OLLAMA_URL",     "http://localhost:11434")
    if groq_key:
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"), "llama-3.1-8b-instant", "Groq", "free"
    if gemini_key:
        return OpenAI(api_key=gemini_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"), "gemini-2.0-flash", "Gemini", "free"
    if openai_key:
        return OpenAI(api_key=openai_key), "gpt-4o-mini", "OpenAI", "paid"
    return OpenAI(base_url=f"{ollama_url}/v1", api_key="ollama"), "llama3.1:8b", "Ollama", "local"

# ── Session state ─────────────────────────────────────────────────────────────

if "messages"        not in st.session_state: st.session_state.messages        = []
if "pending_example" not in st.session_state: st.session_state.pending_example = None
if "follow_ups"      not in st.session_state: st.session_state.follow_ups      = []

client, model_id, backend_name, backend_tier = get_client_and_model()
tier_icon  = {"free": "🟢", "paid": "🟡", "local": "🔵"}.get(backend_tier, "🟢")
can_reason = backend_name == "Groq"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧬 Drug Discovery AI")
    st.caption("Scoped to computational & medicinal chemistry")
    st.divider()

    st.markdown(f"""<div class="bknd-pill">
    {tier_icon} <b>{backend_name}</b> &nbsp;&middot;&nbsp; <code>{model_id}</code>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    if can_reason:
        use_reasoning = st.toggle(
            "Enable Reasoning",
            value=False,
            help="Switches to DeepSeek R1 — shows step-by-step scientific reasoning before each answer.",
        )
        if use_reasoning:
            st.caption(f"Tries: {', '.join(f'`{m}`' for m in REASONING_MODELS)} → CoT fallback")
    else:
        use_reasoning = False

    st.divider()
    st.markdown("#### 💡 Try an example")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        label = q if len(q) <= 52 else q[:49] + "..."
        if st.button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_example = q
            st.session_state.follow_ups = []

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.follow_ups = []
        st.rerun()

    st.divider()
    st.markdown("""
**In scope ✓**
Computational chemistry · Medicinal chemistry · AI/ML in drug discovery ·
ADMET profiling · Molecular docking · Lead optimization · Target ID

**Out of scope ✗**
Medical advice · Clinical dosing · Patient genomics
""")

# ── Tabs — Live Demo first ────────────────────────────────────────────────────

tab_demo, tab_report = st.tabs(["🔬  Live Demo", "📊  Evaluation Report"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 -- LIVE DEMO (primary)
# ══════════════════════════════════════════════════════════════════════════════

with tab_demo:

    # Header
    st.markdown(
        '<p style="font-size:1.7rem;font-weight:900;color:#0f172a;margin-bottom:4px;letter-spacing:-0.04em">'
        '&#128300; Drug Discovery Assistant</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<span class="pill">ADMET</span>'
        '<span class="pill">Molecular Docking</span>'
        '<span class="pill">Lead Optimization</span>'
        '<span class="pill">Target ID</span>'
        '<span class="pill">Cheminformatics</span>',
        unsafe_allow_html=True)

    if use_reasoning:
        st.markdown(
            '<p style="font-size:0.78rem;color:#7c3aed;margin-top:6px">'
            '&#128161; Reasoning mode active &mdash; DeepSeek R1 will think step-by-step before answering.</p>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<p style="font-size:0.78rem;color:#94a3b8;margin-top:6px">'
            f'{tier_icon} {backend_name} &middot; <code style="color:#2563eb">{model_id}</code></p>',
            unsafe_allow_html=True)

    st.markdown('<hr class="sep" style="margin:12px 0 0">', unsafe_allow_html=True)

    # Welcome screen
    if not st.session_state.messages:
        st.markdown("""
<div class="welcome">
  <div style="font-size:2.2rem;margin-bottom:12px">🧬</div>
  <h3 style="color:#0f172a;margin:0 0 10px;font-size:1.1rem;font-weight:800">
    Welcome to Drug Discovery AI
  </h3>
  <p style="color:#64748b;font-size:0.86rem;margin:0;line-height:1.75">
    Ask me anything about drug discovery &mdash; molecular docking, ADMET properties,
    lead optimization, target identification, and more.<br>
    <span style="color:#94a3b8;font-size:0.8rem">
      Use the <strong>example questions</strong> in the sidebar, or type your own below.
    </span>
  </p>
</div>
""", unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("thinking"):
                with st.expander("💭 Reasoning", expanded=False):
                    st.markdown(msg["thinking"])
            st.markdown(msg["content"])

    # Follow-up suggestions (shown after last assistant reply)
    if st.session_state.follow_ups and st.session_state.messages:
        st.markdown('<p class="fu-label">💡 Suggested follow-ups</p>', unsafe_allow_html=True)
        fu_cols = st.columns(len(st.session_state.follow_ups))
        for i, (col, q) in enumerate(zip(fu_cols, st.session_state.follow_ups)):
            with col:
                if st.button(q, key=f"fu_{i}", use_container_width=True):
                    st.session_state.pending_example = q
                    st.session_state.follow_ups = []
                    st.rerun()

    # Resolve prompt
    prompt: str | None = None
    if st.session_state.pending_example:
        prompt = st.session_state.pending_example
        st.session_state.pending_example = None

    typed = st.chat_input("Ask about ADMET, molecular docking, lead optimization...")
    if typed:
        prompt = typed
        st.session_state.follow_ups = []

    # Handle prompt
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        sys_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        with st.chat_message("assistant"):
            thinking_text = None
            full_response = ""

            if use_reasoning and can_reason:
                # ── Reasoning mode ────────────────────────────────────────────
                # Try each Groq reasoning model in order, then fall back to
                # chain-of-thought prompting with the regular model.
                status_ph = st.empty()
                answer_ph = st.empty()

                def _stream_reasoning(model_name, messages):
                    """Stream from model_name; return raw text."""
                    s = client.chat.completions.create(
                        model=model_name, messages=messages,
                        max_tokens=2048, stream=True,
                    )
                    raw = ""
                    got_end = False
                    status_ph.markdown(f"*💭 Reasoning with `{model_name}`...*")
                    for chunk in s:
                        delta = chunk.choices[0].delta.content or ""
                        raw += delta
                        if not got_end:
                            end_idx = raw.find("</think>")
                            if end_idx >= 0:
                                got_end = True
                                ans = raw[end_idx + 8:].strip()
                                if ans:
                                    status_ph.markdown("*💭 Reasoning complete — writing answer...*")
                                    answer_ph.markdown(ans + "▌")
                        else:
                            ans = raw[raw.find("</think>") + 8:].strip()
                            answer_ph.markdown(ans + "▌")
                    return raw

                raw = ""
                used_model = None

                # 1. Try dedicated reasoning models on Groq
                for rm in REASONING_MODELS:
                    try:
                        raw = _stream_reasoning(rm, sys_msgs)
                        used_model = rm
                        break
                    except Exception:
                        continue

                # 2. Chain-of-thought fallback — works with any model
                if not used_model:
                    status_ph.markdown(f"*💭 Thinking step-by-step with `{model_id}`...*")
                    cot_msgs = [{"role": "system", "content": COT_SYSTEM_PROMPT}] + history
                    try:
                        s = client.chat.completions.create(
                            model=model_id, messages=cot_msgs,
                            max_tokens=1536, stream=True,
                        )
                        got_end = False
                        for chunk in s:
                            delta = chunk.choices[0].delta.content or ""
                            raw += delta
                            if not got_end:
                                end_idx = raw.find("</think>")
                                if end_idx >= 0:
                                    got_end = True
                                    ans = raw[end_idx + 8:].strip()
                                    if ans:
                                        status_ph.markdown("*💭 Reasoning complete — writing answer...*")
                                        answer_ph.markdown(ans + "▌")
                            else:
                                ans = raw[raw.find("</think>") + 8:].strip()
                                answer_ph.markdown(ans + "▌")
                        used_model = f"{model_id} (CoT)"
                    except Exception as e:
                        raw = f"Connection error: {e}"

                # Final render
                thinking_text, full_response = parse_reasoning(raw)
                status_ph.empty()
                answer_ph.empty()
                if not full_response:
                    full_response = raw

                if thinking_text:
                    with st.expander("💭 Reasoning", expanded=False):
                        st.markdown(thinking_text)
                st.markdown(full_response)
                if used_model:
                    st.caption(f"Reasoned with `{used_model}`")

            else:
                # ── Standard streaming mode ────────────────────────────────────
                placeholder = st.empty()
                try:
                    stream = client.chat.completions.create(
                        model=model_id, messages=sys_msgs, max_tokens=1024, stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        full_response += delta
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"Connection error ({backend_name}): {e}"
                    placeholder.error(full_response)

        # Save message with optional thinking
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "thinking": thinking_text,
        })

        # Generate follow-up suggestions asynchronously
        if full_response and len(full_response) > 30:
            st.session_state.follow_ups = generate_followups(
                client, model_id, prompt, full_response
            )

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 -- EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_report:

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero">
  <div class="h-eyebrow">CeRAI AIEvaluationTool v2.0 &nbsp;&middot;&nbsp; Gates Foundation AI Fellowship India 2026</div>
  <h1>Safety &amp; Reliability Evaluation<br>of a <em>Drug Discovery</em> LLM</h1>
  <p class="h-sub">
    A systematic evaluation of an LLM-powered drug discovery assistant &mdash;
    examining truthfulness, hallucination resistance, content filtering, and scope
    enforcement across 13 hand-crafted test cases and two independent inference backends.
  </p>
  <div class="h-stats">
    <div class="hst"><span class="hst-v">6</span><span class="hst-l">Metrics</span></div>
    <div class="hst"><span class="hst-v">19</span><span class="hst-l">Test Cases</span></div>
    <div class="hst"><span class="hst-v">3</span><span class="hst-l">Eval Plans</span></div>
    <div class="hst"><span class="hst-v">2</span><span class="hst-l">Backends</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 01 Executive Summary ───────────────────────────────────────────────────
    st.markdown("""
<div class="sh">
  <div class="sh-row"><span class="sh-tag">01</span><h2>Executive Summary</h2></div>
  <p>Research context and key findings narrative.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="narrative">
<strong style="color:#0f172a">Llama 3.1 8B</strong>, deployed as a domain-scoped drug discovery assistant,
was evaluated across 13 test cases and 6 safety &amp; quality metrics using the CeRAI AIEvaluationTool v2.0.
The model demonstrates <strong style="color:#16a34a">strong safety guardrails</strong> &mdash; it consistently
refuses jailbreak attempts, harmful synthesis requests, and privacy-violating queries with zero false negatives.
Factual accuracy is <strong style="color:#d97706">partially satisfactory at 75%</strong>, with the gap likely
attributable to exact-match scoring penalising scientifically valid paraphrases rather than genuine factual errors.
The most critical finding is a <strong style="color:#dc2626">scope enforcement failure</strong>: despite an explicit
system-prompt instruction, the model answered an out-of-domain query (biryani recipe), exposing a known
instruction-following limitation in sub-10B open models.
Crucially, evaluating on both Ollama (local) and Groq (cloud) confirmed that all safety properties are
<strong style="color:#0f172a">infrastructure-independent</strong> &mdash; the production deployment maintains
the same safety profile as the evaluated model.
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ── 02 Methodology ─────────────────────────────────────────────────────────
    st.markdown("""
<div class="sh">
  <div class="sh-row"><span class="sh-tag">02</span><h2>Methodology</h2></div>
  <p>Target system, evaluation framework, and test suite design.</p>
</div>
""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
<div class="mc">
  <div class="mc-icon">&#127919;</div>
  <div class="mc-title">Target System</div>
  <div class="mc-body">
    <strong style="color:#0f172a">Llama 3.1 8B</strong> deployed as a domain-scoped
    drug discovery assistant &mdash; restricted to computational chemistry, medicinal chemistry,
    and AI-driven drug development. Explicit system-prompt refusal instruction for
    out-of-scope queries.<br><br>
    <span class="tag">llama-3.1-8b-instant</span>
    <span class="tag">Groq API</span>
    <span class="tag">Ollama local</span>
  </div>
</div>
""", unsafe_allow_html=True)

    with m2:
        st.markdown("""
<div class="mc">
  <div class="mc-icon">&#9881;&#65039;</div>
  <div class="mc-title">CeRAI Framework</div>
  <div class="mc-body">
    Open-source evaluation suite with SQLite backend, FastAPI interface manager,
    and CLI-driven test execution across 3 structured evaluation plans.<br><br>
    <span class="tag">Responsible AI</span>
    <span class="tag">Conv. Quality</span>
    <span class="tag">Guardrails &amp; Safety</span><br><br>
    Strategies: <em>truthfulness_squad</em> &middot; <em>hallucination_haluqa</em> &middot;
    <em>safety_strategy</em> &middot; <em>llm_judge_positive</em>
  </div>
</div>
""", unsafe_allow_html=True)

    with m3:
        st.markdown("""
<div class="mc">
  <div class="mc-icon">&#129514;</div>
  <div class="mc-title">Test Suite Design</div>
  <div class="mc-body">
    13 hand-crafted test cases spanning 5 risk categories &mdash; factual recall,
    adversarial jailbreaks, boundary probing, fictitious compound detection,
    and scope enforcement.<br><br>
    <span class="tag">Jailbreak attempts</span>
    <span class="tag">Synthesis requests</span>
    <span class="tag">Factual recall</span>
    <span class="tag">Fictitious compounds</span>
    <span class="tag">Scope probing</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ── 03 Limitations ──────────────────────────────────────────────────────────
    st.markdown("""
<div class="sh">
  <div class="sh-row"><span class="sh-tag">03</span><h2>Limitations</h2></div>
  <p>Honest constraints on what this study can and cannot claim.</p>
</div>
""", unsafe_allow_html=True)

    lim1, lim2 = st.columns(2)
    with lim1:
        with st.expander("Statistical & design constraints", expanded=True):
            st.markdown("""
- **13 test cases** is a directional signal, not a statistically robust benchmark. No confidence intervals are reported.
- **Single-temperature, single-draw** — adversarial robustness requires pass@k across multiple samples.
- **No multi-turn evaluation.** All prompts are single-turn; multi-step context manipulation was not tested.
- **BLEU for scope testing is semantically inappropriate.** A correct refusal phrased differently from the reference phrase scores ~0 even when accurate. NLI or cosine similarity would be more appropriate.
- **No expert validation** of reference answers by domain specialists in medicinal chemistry.
""")
        with st.expander("Generalisability caveat"):
            st.markdown("""
Results characterise a **small, general-purpose, prompted model**. These findings are **not** representative of:
- Fine-tuned drug discovery models (ChemBERTa, Galactica, BioMedLM)
- RAG-augmented systems backed by real drug databases (ChEMBL, DrugBank)
- Larger models (GPT-4o, Claude 3, Gemini Ultra)
- Production pharmaceutical AI with dedicated safety-layer pipelines
""")

    with lim2:
        with st.expander("CeRAI tool bugs found & fixed", expanded=True):
            st.markdown("""
| Bug | Resolution |
|-----|------------|
| `--config` flag silently ignored | Fixed path resolution in CLI |
| `.env` not loaded by server — API keys missing | Key relay via `api_context.extra` |
| `base_url_local` ignored — hardcoded `localhost:8000` | Fixed to read from config |
| `llm_judge_negative` inverted DD018/DD019 scores | DB-patched via fix script |
| `weasyprint` unconditional import crashed on Windows | Wrapped in `try/except` |
| `similarity_match` raised on unknown metric names | Added BLEU fallback |
| Non-ASCII apostrophe broke refusal keyword check | Added Unicode normalisation |
""")
        with st.expander("LLM judge sensitivity"):
            st.markdown("""
DD015–DD019 use LLM-as-judge (Ollama `llama3.1:8b`). These scores are:
- **Stochastic** — not deterministic across runs
- **Model-sensitive** — change when the judge model changes
- **Prompt-sensitive** — slight rewordings shift scores

Do not compare LLM-judge scores across runs with different judge models or temperatures.
""")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ── 04 Conclusions ──────────────────────────────────────────────────────────
    st.markdown("""
<div class="sh">
  <div class="sh-row"><span class="sh-tag">04</span><h2>Conclusions &amp; Recommendations</h2></div>
  <p>Three targeted actions to move from research evaluation to production-ready deployment.</p>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class="ct">
  <div class="ct-em">&#128274;</div>
  <h4>Fix Scope Enforcement</h4>
  <p>Deploy a zero-shot NLI topic classifier as a pre-model gate. Never rely on
  system-prompt instruction-following for hard domain restrictions &mdash; a known
  limitation of sub-10B instruction-tuned models. Few-shot refusal examples
  may help as a secondary measure.</p>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="ct">
  <div class="ct-em">&#128207;</div>
  <h4>Improve Eval Metrics</h4>
  <p>Replace exact-match truthfulness scoring with semantic similarity (BERTScore
  or cosine embedding distance). Redesign the scope enforcement metric to be
  directionally intuitive &mdash; higher score should mean better enforcement,
  preventing misinterpretation of near-zero results.</p>
</div>
""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class="ct">
  <div class="ct-em">&#129514;</div>
  <h4>Scale the Test Suite</h4>
  <p>Expand to 50&ndash;100 expert-validated test cases per metric. Include multi-turn
  adversarial scenarios and rare drug discovery subdomains (PROTACs, covalent
  inhibitors, FBDD) to stress-test knowledge boundaries beyond common
  textbook topics.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ── 05 Reproduce ────────────────────────────────────────────────────────────
    with st.expander("&#9881;&#65039;  How to Reproduce — 3 terminals, ~20 min"):
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**1. Setup**")
            st.code("""git clone https://github.com/harshitsinghsnu/conversationalAI
cd EvaluationTool/AIEvaluationTool
pip install -r requirements.txt
pip install -r src/app/interface_manager/requirements.txt
echo 'GROQ_API_KEY=gsk_...' > .env
python setup_drug_discovery.py
python fix_dd018_dd019_strategy.py""", language="bash")
        with r2:
            st.markdown("**2. Run (3 terminals)**")
            st.code("""# T1 -- Interface Manager
cd src/app/interface_manager && python main.py

# T2 -- Execute all 3 plans
for pid in 1 2 3; do
  python src/app/testcase_executor/main.py \\
    --config config_sqlite.json \\
    --testplan-id $pid --execute
done

# T3 -- Analyse results
python src/app/response_analyzer/analyze.py \\
  --config config_sqlite.json --run-name <run>""", language="bash")
        st.info("**Live demo:** https://2gcrangcc67rwsa2cbpopr.streamlit.app  |  **Repo:** github.com/harshitsinghsnu/conversationalAI")
