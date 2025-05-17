import tempfile
from pathlib import Path
from typing import Dict, Any

from birdnet import predict_species_within_audio_file, SpeciesPredictions
from fastapi import UploadFile


async def process_audio_file(
    audio_file: UploadFile, min_confidence: float = 0.1
) -> Dict[str, Any]:
    """
    Process an uploaded audio file to predict bird species.

    Args:
        audio_file: The uploaded audio file
        min_confidence: Minimum confidence threshold for predictions

    Returns:
        Dictionary with processed species predictions
    """

    temp_path = await create_temp_file(audio_file)

    try:
        predictions = run_birdnet_prediction(temp_path, min_confidence)
        species_data = aggregate_species_data(predictions)
        return format_result(species_data)

    finally:
        remove_temp_file(temp_path)


async def create_temp_file(audio_file: UploadFile) -> Path:

    temp_path = Path(tempfile.mktemp(suffix=".wav"))
    content = await audio_file.read()
    temp_path.write_bytes(content)
    return temp_path


def remove_temp_file(temp_path: Path) -> None:
    if temp_path.exists():
        temp_path.unlink()


def run_birdnet_prediction(
    audio_path: Path, min_confidence: float
) -> SpeciesPredictions:
    """
    Run BirdNET prediction on an audio file.

    Args:
        audio_path: Path to the audio file
        min_confidence: Minimum confidence threshold for predictions

    Returns:
        SpeciesPredictions object with raw prediction results
    """
    predictions = SpeciesPredictions(
        predict_species_within_audio_file(
            audio_path,
            min_confidence=min_confidence,
            batch_size=100,
            use_bandpass=True,
            silent=True,
        )
    )
    return predictions
