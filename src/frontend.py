import streamlit as st
import requests

# Page configuration
st.set_page_config(page_title="LangGraph Agent AI", layout="centered")
st.title("AI ChatBot Agents")
st.write("Create and interact with AI agents!")

# Input fields
system_prompt = st.text_area(
    "Define your AI Agent: ",
    height=70,
    placeholder="Type your system prompt here...",
)

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider_model = st.radio("Select Provider:", ("Groq", "OpenAI"))
API_URL = "http://127.0.0.1:9999/chat"

# Model selection based on provider
if provider_model == "Groq":
    selected_model = st.selectbox("Select Groq Model", MODEL_NAMES_GROQ)
elif provider_model == "OpenAI":
    selected_model = st.selectbox("Select the OpenAI Model", MODEL_NAMES_OPENAI)

# Web search allowance checkbox
allowed_websearch = st.checkbox("Allow Web Search")

# User query input
user_query = st.text_area(
    "Enter Your Query", height=180, placeholder="Ask your question here..."
)

# Submit button
if st.button("Ask the Agent"):
    if user_query.strip() and system_prompt.strip():  # Ensure system prompt is not empty
        with st.spinner("Processing your query..."):
            # Prepare payload
            payload = {
                "model_name": selected_model,
                "model_provider": provider_model,
                "system_promt": system_prompt,  # Ensure this field is included
                "message": [user_query],
                "allow_search": allowed_websearch,
            }

            try:
                # Send POST request
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    try:
                        # Try to parse the response as JSON
                        response_data = response.json()

                        # Check if the response contains an error message
                        if isinstance(response_data, dict) and "error" in response_data:
                            st.error(response_data["error"])
                        elif isinstance(response_data, dict) and "response" in response_data:
                            st.subheader("Agent Response")
                            st.markdown(f"**Final Response:** {response_data['response']}")
                        else:
                            # If response doesn't have expected fields, display the whole response
                            st.subheader("Agent Response")
                            st.markdown(f"**Response:** {response_data}")
                    except ValueError:
                        # If response is not JSON, handle it as a plain string
                        st.error(f"Invalid response format: {response.text}")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter both the system prompt and query before submitting.")
