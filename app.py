import streamlit as st
from core.agent_setup import get_math_agent, get_wikipedia_agent, get_reasoning_agent

# --- App Config ---
st.set_page_config(page_title="GemMath AI - Solve Math with Gemma 2", page_icon="🧮", layout="wide")

st.title("🧠 GemMath AI")
st.markdown("Your personal assistant for solving **numeric** and **symbolic** math problems with **Gemma 2**, plus Wikipedia search and logic reasoning.")

# --- Sidebar for API Key ---
with st.sidebar:
    st.title("⚙️ Settings")
    groq_api_key = st.text_input("🔑 Groq API Key", type="password")
    show_steps = st.toggle("🧠 Show steps / explanation")

    if not groq_api_key:
        st.info("Please enter your Groq API Key to use the app.")
        st.stop()

# --- Tabs ---
tabs = st.tabs(["🔢 Solve a Math Problem", "📚 Search Wikipedia", "🤖 Logical Reasoning", "ℹ️ About the Assistant"])

# --- Math Solver Tab ---
with tabs[0]:
    st.subheader("Enter your math question")
    math_question = st.text_area("Example: 'differentiate x**2 * sin(x)' or '12 + 5 * 3'", placeholder="Enter your math question here...")
    if st.button("🧮 Solve", key="solve"):
        with st.spinner("Thinking with Gemma 2..."):
            agent = get_math_agent(groq_api_key)
            result = agent.run(math_question)

            # LaTeX result rendering
            if result.startswith("$$") and result.endswith("$$"):
                st.latex(result.strip("$").strip())
            else:
                st.success(result)

            # Optional: Show LLM explanation steps
            if show_steps:
                with st.spinner("Explaining how the answer was derived..."):
                    reasoning_agent = get_reasoning_agent(groq_api_key)
                    explanation_prompt = f"Explain step-by-step how to solve: {math_question}"
                    steps = reasoning_agent.run(explanation_prompt)
                    st.markdown("---")
                    st.markdown("### 🧠 Explanation")
                    st.info(steps)

# --- Wikipedia Tab ---
with tabs[1]:
    st.subheader("Search Wikipedia")
    wiki_query = st.text_input("What do you want to know about?", "Black holes")
    if st.button("🔍 Search", key="wiki"):
        with st.spinner("Searching Wikipedia..."):
            agent = get_wikipedia_agent(groq_api_key)
            result = agent.run(wiki_query)
            st.success(result)

# --- Reasoning Tab ---
with tabs[2]:
    st.subheader("Enter a logical reasoning question")
    logic_question = st.text_area("Example: 'If A is taller than B and B is taller than C...'", placeholder="Type your logical question here...")
    if st.button("🧠 Analyze", key="logic"):
        with st.spinner("Processing with Gemma 2..."):
            agent = get_reasoning_agent(groq_api_key)
            result = agent.run(logic_question)
            st.success(result)

# --- About Tab ---
with tabs[3]:
    st.markdown("""
        ### 🤖 About GemMath AI
        Welcome to **GemMath AI**, a chatbot powered by **Google Gemma 2 via Groq**, designed to:

        - 🧮 Solve numeric math problems (e.g., `12 + 5 * 3`)
        - ∫ Perform symbolic math with SymPy (e.g., `differentiate x**2 * sin(x)`)
        - 📚 Fetch relevant information from Wikipedia
        - 🧠 Answer logical reasoning questions
        - 💡 Provide step-by-step reasoning if enabled

        Built using [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io), and [Groq](https://groq.com).

        > This is a personal educational project. Please verify outputs independently.
    """)
