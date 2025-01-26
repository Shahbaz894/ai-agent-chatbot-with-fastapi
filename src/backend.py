from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException
from agent import get_response_from_ai_agent  # Ensure this function is properly implemented and imported

# Request schema definition
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_promt: str
    message: List[str]
    allow_search: bool

# FastAPI app initialization
app = FastAPI(title="LangGraph AI Agent")

# List of allowed model names
ALLOWED_MODEL_NAMES = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
    "gpt-4o-mini"
]

@app.post('/chat')
def chat_endpoint(request: RequestState):
    """
    Chat endpoint for interacting with AI agents.
    Validates the model name and forwards the request to the AI agent handler.
    """
    # Validate model name
    if request.model_name not in ALLOWED_MODEL_NAMES:
        raise HTTPException(status_code=400, detail="Invalid model name. Kindly select a valid AI model.")
    
    try:
        # Extract request parameters
        llm_id = request.model_name
        query = request.message
        allowed_search = request.allow_search
        system_promt = request.system_promt
        provider = request.model_provider

        # Get response from AI agent
        response = get_response_from_ai_agent(
            llm_id=llm_id,
            query=query,
            allow_search=allowed_search,
            system_prompt=system_promt,
            provider=provider
        )
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
