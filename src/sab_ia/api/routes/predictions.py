from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from sab_ia.services import process_audio_file

predictions_router = APIRouter(prefix="/rest/v1")


@predictions_router.post("/predict")
async def predict_species(
    audio_file: UploadFile = File(...),
    min_confidence: float = Query(
        0.1, ge=0.0, lt=1.0, description="Minimum confidence threshold for predictions"
    ),
):
    return process_audio_file(audio_file, min_confidence)
