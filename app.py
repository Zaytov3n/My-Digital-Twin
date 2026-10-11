import streamlit as st

from utils.llm import API_KEY, MODEL_NAME, build_system_instruction, get_client
from utils.rag import format_snippets, retrieve
from utils.tools import (
    ACTIVITY_OPTIONS,
    DEFAULT_ACTIVITIES,
    build_twin_context,
)

st.set_page_config(page_title="My Digital Twin", page_icon="🤖", layout="wide")

if not API_KEY:
    st.error(
        "GEMINI_API_KEY not found. Add it to your local `.env` file, or to "
        "the app's **Secrets** in Streamlit Community Cloud."
    )
    st.stop()

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "activities" not in st.session_state:
    st.session_state.activities = list(DEFAULT_ACTIVITIES)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    user_name = st.text_input("Your name", value="Seth")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)

    st.divider()
    st.subheader("📝 Log an activity")
    new_activity = st.selectbox("Activity", ACTIVITY_OPTIONS)
    if st.button("Add to history", use_container_width=True):
        st.session_state.activities.append(new_activity)
        st.rerun()

    if st.button("Reset activity history", use_container_width=True):
        st.session_state.activities = list(DEFAULT_ACTIVITIES)
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Model: `{MODEL_NAME}`")

# ---------------- Main layout ----------------
st.title("🤖 My Digital Twin")
st.caption("Ask about your activity patterns or what to do next.")

context = build_twin_context(st.session_state.activities)
chat_tab, insights_tab = st.tabs(["💬 Chat", "📊 Insights"])

with insights_tab:
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", context["number_of_activity_records"])
    c2.metric("Current activity", context["current_recorded_activity"] or "—")
    c3.metric("Likely next", context["most_likely_next_activity"] or "—")

    st.subheader("Activity frequencies")
    if context["activity_frequencies"]:
        st.bar_chart(context["activity_frequencies"])

    st.subheader("Next-activity probabilities")
    st.json(context["next_activity_probabilities"])

    st.subheader("History")
    st.write(" → ".join(st.session_state.activities))

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask your digital twin...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    knowledge = format_snippets(retrieve(prompt, k=3))
                    recent = st.session_state.messages[-20:]
                    answer = get_client().chat(
                        messages=recent,
                        system_instruction=build_system_instruction(
                            user_name, context, knowledge
                        ),
                        temperature=temperature,
                    )
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error(f"Request failed: {e}")