import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent

# Import your existing database setup and ML tool!
from agent import setup_database_connection, predict_failure_probability, send_alert_email, send_bulk_alert_emails

# --- Setup the Agent ---
@st.cache_resource # This keeps the agent loaded so it doesn't reconnect every time you type
def initialize_agent():
    # Load environment variables (like GOOGLE_API_KEY) from .env file
    load_dotenv()
    
    db = setup_database_connection()
    
    # Stop the app gracefully if the database fails to connect
    if db is None:
        st.error("❌ Failed to connect to the PostgreSQL database.")
        st.info("Ensure your database is running and `DB_PASSWORD` is set in your environment variables or Streamlit Secrets.")
        st.stop()
        
    # Safely fetch the API key and handle missing keys gracefully
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Google API Key is missing.")
        st.info("Please add `GOOGLE_API_KEY` to your Streamlit Secrets in the deployment dashboard.")
        st.stop()
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=5, google_api_key=api_key)
    
    # Create a strict identity for Veda AI so it handles out-of-domain questions gracefully
    custom_prefix = """You are Veda AI, an elite EV Predictive Maintenance Assistant.
    Your ONLY job is to analyze vehicle telemetry data from the database and predict battery/motor failures.
    You also have the ability to send emergency email alerts to individual customers or bulk emails to multiple customers if their vehicles are at high risk.
    If the user asks questions unrelated to EV telemetry, vehicle health, or your database (like geography, routing, history, or personal questions), politely decline and state: "I am Veda AI, an EV Engineering Assistant. I can only assist with vehicle telemetry and health diagnostics." Do NOT attempt to use SQL tools for these questions.
    """

    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        extra_tools=[predict_failure_probability, send_alert_email, send_bulk_alert_emails],
        prefix=custom_prefix,
        verbose=True, # We keep this True so you can still see the thought process in your terminal
        max_iterations=40 # Increased so the AI doesn't give up on complex, multi-step questions
    )
    return agent_executor

agent = initialize_agent()

# --- Streamlit UI Design ---
st.set_page_config(page_title="Veda AI", page_icon="🚗")
st.title("🚗 Veda AI: EV Engineering Assistant")
st.markdown("Ask me to analyze vehicle telemetry, predict failures, or calculate fleet averages.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history in Streamlit's memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input text box
if prompt := st.chat_input("E.g., What is the average motor temp? Is the vehicle at risk?"):
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing telemetry and predicting health..."):
            try:
                response = agent.invoke({"input": prompt})
                
                # Clean up Gemini's raw output format
                raw_output = response['output']
                if isinstance(raw_output, list):
                    clean_output = "".join([item.get('text', '') if isinstance(item, dict) else str(item) for item in raw_output])
                else:
                    clean_output = raw_output
                    
                st.markdown(clean_output)
                st.session_state.messages.append({"role": "assistant", "content": clean_output})
            except Exception as e:
                st.error(f"An error occurred: {e}")