from fastapi import APIRouter, UploadFile, File, HTTPException, Query


predictions_router = APIRouter()

@predictions_router.post("/predict")
async def predict_species(
        audio_file: UploadFile = File(...),
        min_confidence: float = Query(0.1, ge=0.0, lt=1.0, description="Minimum confidence threshold for predictions"),
):
    ...