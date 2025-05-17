import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple


from fastapi import UploadFile
from birdnet import predict_species_within_audio_file, SpeciesPredictions


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


def parse_species_name(species_name: str) -> Tuple[str, str]:
    """
    Parse a species name string into scientific and common names.

    Args:
        species_name: The full species name (e.g., "Carduelis carduelis_European Goldfinch")

    Returns:
        Tuple of (scientific_name, common_name)
    """
    parts = species_name.split("_", 1)
    scientific_name = parts[0]
    common_name = parts[1] if len(parts) > 1 else ""
    return scientific_name, common_name


def aggregate_species_data(predictions: SpeciesPredictions) -> List[Dict[str, Any]]:
    """
    Aggregate prediction data across time intervals.

    Args:
        predictions: Raw SpeciesPredictions from BirdNET

    Returns:
        List of dictionaries with aggregated species data
    """

    species_data = defaultdict(
        lambda: {
            "scientific_name": "",
            "common_name": "",
            "confidence": 0.0,
            "occurrences": 0,
        }
    )

    # Process each time interval prediction
    for time_interval, species_predictions in predictions.items():
        for species_name, confidence in species_predictions.items():

            scientific_name, common_name = parse_species_name(species_name)

            if (
                scientific_name not in species_data
                or confidence > species_data[scientific_name]["confidence"]
            ):
                species_data[scientific_name]["scientific_name"] = scientific_name
                species_data[scientific_name]["common_name"] = common_name
                species_data[scientific_name]["confidence"] = confidence

            # Increment occurrence count
            species_data[scientific_name]["occurrences"] += 1

    # Convert to list and sort by confidence (highest first)
    species_list = list(species_data.values())
    species_list.sort(key=lambda x: x["confidence"], reverse=True)

    return species_list


def format_result(species_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"species_count": len(species_list), "species": species_list}
