from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sys

sys.stdout.reconfigure(line_buffering=True)  # Enable line buffering

# Load the GPT-2 medium model for better quality
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

class GenerationResponse(BaseModel):
    response: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: PromptRequest):
    print(f"Received prompt: {request.prompt}")
    
    # Add a more detailed system prompt to guide the model
    system_prompt = "You are a helpful AI assistant. Respond to questions or comments directly and accurately. If you're unsure about something, admit it. Don't make assumptions or hallucinate information."
    full_prompt = f"{system_prompt}\n\nHuman: {request.prompt}\nAI:"
    
    inputs = tokenizer.encode_plus(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + 50,  # Limit new tokens
            num_return_sequences=1,
            temperature=0.6,  # Slightly reduce temperature for more focused responses
            top_k=50,
            top_p=0.92,  # Slightly reduce top_p for more focused responses
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # Increase to further reduce repetition
            repetition_penalty=1.3   # Increase repetition penalty
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_response = response.split("AI:")[-1].strip()
    
    # Ensure the response is not empty and is reasonably long
    if len(ai_response) < 20:
        ai_response = "I apologize, but I don't have a good response to that. Could you please rephrase your question or provide more context?"
    
    # Add a post-processing step to improve coherence
    ai_response = ai_response.split('.')[0] + '.'  # Keep only the first sentence
    
    print(f"Generated response: {ai_response}")
    return GenerationResponse(response=ai_response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)