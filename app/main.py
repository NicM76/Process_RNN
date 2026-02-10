import torch
import torch.nn.functional as F
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager

# Import your model definition
from src.model import ProcessLSTM

# --- CONFIGURATION (Must match training!) ---
# In a real company, these would be in a config.yaml or .env file
MODEL_PATH = "models/process_lstm.pth"
MAPPING_PATH = "data/activity_map.json"
NUM_CLASSES = 25
EMBED_DIM = 16
HIDDEN_DIM = 64
WINDOW_SIZE = 5

# Global variables to hold artifacts
artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model and mappings ONCE when the server starts.
    """
    # 1. Load the Activity Mapping
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Mapping not found at {MAPPING_PATH}")
    
    with open(MAPPING_PATH, "r") as f:
        activity_to_int = json.load(f)
        # Create Reverse Mapping (Int -> String)
        int_to_activity = {v: k for k, v in activity_to_int.items()}
    
    # 2. Load the Model Architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProcessLSTM(NUM_CLASSES, EMBED_DIM, HIDDEN_DIM)
    
    # 3. Load Weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode
    
    # Store in global dictionary
    artifacts["model"] = model
    artifacts["act_to_int"] = activity_to_int
    artifacts["int_to_act"] = int_to_activity
    artifacts["device"] = device
    
    print(f"✅ System Online. Model loaded on {device}.")
    yield
    # Clean up (if needed) when app shuts down
    artifacts.clear()

app = FastAPI(title="Process Predictor API", lifespan=lifespan)

# --- REQUEST SCHEMA ---
class TraceRequest(BaseModel):
    activities: List[str]

# --- RESPONSE SCHEMA ---
class PredictionResponse(BaseModel):
    next_activity: str
    confidence: float
    input_trace_length: int

@app.get("/")
def health_check():
    return {"status": "running", "model_version": "v1.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict_next_event(payload: TraceRequest):
    # Unpack artifacts
    model = artifacts["model"]
    act_to_int = artifacts["act_to_int"]
    int_to_act = artifacts["int_to_act"]
    device = artifacts["device"]
    
    # 1. Preprocess: Convert strings to integers
    # Use 0 (PAD) for unknown activities
    indices = [act_to_int.get(act, 0) for act in payload.activities]
    
    if not indices:
        raise HTTPException(status_code=400, detail="Trace cannot be empty")

    # 2. Sliding Window Logic (Same as training!)
    # Take only the last WINDOW_SIZE elements
    input_seq = indices[-WINDOW_SIZE:]
    
    # Pre-pad with 0 if too short
    if len(input_seq) < WINDOW_SIZE:
        input_seq = [0] * (WINDOW_SIZE - len(input_seq)) + input_seq
        
    # 3. Convert to Tensor
    # Shape: (1, WINDOW_SIZE)
    input_tensor = torch.LongTensor([input_seq]).to(device)
    
    # 4. Inference
    with torch.no_grad():
        logits = model(input_tensor) # Shape: (1, 25)
        probabilities = F.softmax(logits, dim=1)
        
        # Get the top prediction
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_idx = predicted_idx.item()
    confidence_score = confidence.item()
    
    # 5. Decode
    prediction_name = int_to_act.get(predicted_idx, "UNKNOWN")
    
    return {
        "next_activity": prediction_name,
        "confidence": round(confidence_score, 4),
        "input_trace_length": len(payload.activities)
    }