import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults

# ---------------- LOAD ENV FIRST ----------------
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY not found. Check your .env file.")
    st.stop()

# ---------------- TOOLS ----------------
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
)

search = TavilySearchResults(
    name="Search",
    tavily_api_key=TAVILY_API_KEY   # ✅ EXPLICIT (IMPORTANT)
)

tools = [search, arxiv, wiki]

# ---------------- UI ----------------
st.title("🔎 LangChain Search Agent (Groq + Tavily)")

st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 I can search the web, Arxiv, and Wikipedia."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- CHAT ----------------
if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        streaming=True
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(prompt, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
