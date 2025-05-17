from typing import Dict, Any

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


