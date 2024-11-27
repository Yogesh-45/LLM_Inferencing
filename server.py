from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn

# Define the FastAPI app
app = FastAPI()

def llama_generate(input_data, history= None):
    # Path to the GGUF model file
    GGUF_MODEL_PATH = "D:/yogi/LLM/simplismart/model/vicuna-7B_Q4_K_M.gguf"

    # Load the GGUF model
    try:
        llama = Llama(model_path=GGUF_MODEL_PATH, n_ctx=512, n_threads=4)
    except Exception as e:
        raise RuntimeError(f"Failed to load GGUF model: {e}")

    response = llama(
            prompt=input_data.prompt,
            max_tokens= input_data.max_tokens,
            temperature=input_data.temperature,
            stop=["\n"]
        )
    
    return response

# Input schema for API
class QueryInput(BaseModel):
    prompt: str
    max_tokens: int = 128  # Default number of tokens to generate
    temperature: float = 0.8  # Default temperature for generation


# API endpoint to generate text
@app.post("/generate")
async def generate_text(input_data: QueryInput):
    try:
        # Generate text with the Llama model
        response = llama_generate(
            input_data=input_data,
        )
        return {"prompt": input_data.prompt, "response": response["choices"][0]["text"]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    return {"message": "GGUF model server is running!"}


if __name__ == "__main__":
    uvicorn.run(
        app= "server:app",
        host= '127.0.0.1',
        port= 7860,
        reload= True
    )
