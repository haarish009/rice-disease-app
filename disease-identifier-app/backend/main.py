# Restored FastAPI app

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class DiseaseInput(BaseModel):
    image_path: str

class DiseaseOutput(BaseModel):
    final_label: str
    agrees_with_stage1: bool

@app.post("/predict", response_model=DiseaseOutput)
async def predict(input: DiseaseInput):
    # Assume we have a way to get stage1_label and stage2_label
    stage1_label = "some_label"  # Determine this from your model
    # Keep final_label = stage1_label always
    final_label = stage1_label
    
    # Compute Stage 2
    stage2_label = "another_label"  # Determine this from another part of your model
    agrees_with_stage1 = (stage1_label == stage2_label)
    
    # Keep metadata lookup based on final_label
    metadata = get_metadata(final_label)  # Assume this function exists
    
    return DiseaseOutput(final_label=final_label, agrees_with_stage1=agrees_with_stage1)

# Other routes and logic remain unchanged