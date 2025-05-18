from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from sab_ia.services.prediction_service import (
    process_audio_file,
    create_temp_file,
    run_birdnet_prediction,
    parse_species_name,
    aggregate_species_data,
    format_result,
    ensure_mono_audio,
)


def test_parse_species_name():

    scientific, common = parse_species_name("Carduelis carduelis_European Goldfinch")
    assert scientific == "Carduelis carduelis"
    assert common == "European Goldfinch"

    # Test with only scientific name
    scientific, common = parse_species_name("Parus major")
    assert scientific == "Parus major"
    assert common == ""


def test_format_result():

    species_list = [
        {
            "scientific_name": "Carduelis carduelis",
            "common_name": "European Goldfinch",
            "confidence": 0.85,
            "occurrences": 4,
        },
        {
            "scientific_name": "Parus major",
            "common_name": "Great Tit",
            "confidence": 0.76,
            "occurrences": 2,
        },
    ]

    result = format_result(species_list)

    assert result["species_count"] == 2
    assert result["species"] == species_list


def test_aggregate_species_data():

    mock_predictions = {
        (0.0, 3.0): {
            "Carduelis carduelis_European Goldfinch": 0.85,
            "Parus major_Great Tit": 0.76,
        },
        (3.0, 6.0): {
            "Carduelis carduelis_European Goldfinch": 0.65,
            "Turdus merula_Common Blackbird": 0.62,
        },
        (6.0, 9.0): {"Carduelis carduelis_European Goldfinch": 0.75},
    }

    aggregated = aggregate_species_data(mock_predictions)

    # Check if the species are sorted by confidence
    assert len(aggregated) == 3
    assert aggregated[0]["scientific_name"] == "Carduelis carduelis"
    assert aggregated[0]["confidence"] == 0.85  # Should use the highest confidence
    assert aggregated[0]["occurrences"] == 3  # Should count all occurrences

    # Check the second and third species
    assert aggregated[1]["scientific_name"] == "Parus major"
    assert aggregated[1]["occurrences"] == 1

    assert aggregated[2]["scientific_name"] == "Turdus merula"
    assert aggregated[2]["occurrences"] == 1


@patch("sab_ia.services.prediction_service.predict_species_within_audio_file")
def test_run_birdnet_prediction(mock_predict):

    mock_predictions = {(0.0, 3.0): {"Species1_Common1": 0.9}}
    mock_predict.return_value = mock_predictions

    result = run_birdnet_prediction(Path("test.wav"), 0.1)

    mock_predict.assert_called_once_with(
        Path("test.wav"),
        min_confidence=0.1,
        batch_size=100,
        use_bandpass=True,
        silent=True,
    )
    assert result == mock_predictions


@patch("sab_ia.services.prediction_service.AudioSegment")
def test_ensure_mono_audio_stereo(mock_audio_segment):

    mock_audio = MagicMock()
    mock_audio.channels = 2  # Stereo
    mock_audio_segment.from_file.return_value = mock_audio

    ensure_mono_audio(Path("test.wav"))

    mock_audio.set_channels.assert_called_once_with(1)
    mock_audio.set_channels.return_value.export.assert_called_once()


@patch("sab_ia.services.prediction_service.AudioSegment")
def test_ensure_mono_audio_already_mono(mock_audio_segment):

    mock_audio = MagicMock()
    mock_audio.channels = 1  # Already mono
    mock_audio_segment.from_file.return_value = mock_audio

    ensure_mono_audio(Path("test.wav"))

    mock_audio.set_channels.assert_not_called()


@patch("sab_ia.services.prediction_service.AudioSegment")
def test_ensure_mono_audio_exception(mock_audio_segment):

    mock_audio_segment.from_file.side_effect = Exception("Audio processing error")

    # Assert exception is raised with the expected message
    with pytest.raises(ValueError, match="Falha ao processar arquivo de áudio:.*"):
        ensure_mono_audio(Path("test.wav"))


@patch("sab_ia.services.prediction_service.tempfile.mktemp")
async def test_create_temp_file(mock_mktemp):

    mock_mktemp.return_value = "/tmp/test_12345.wav"

    mock_upload = AsyncMock()
    mock_upload.read.return_value = b"audio data"

    with patch("sab_ia.services.prediction_service.Path") as mock_path:
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        result = await create_temp_file(mock_upload)

        mock_upload.read.assert_called_once()
        mock_path_instance.write_bytes.assert_called_once_with(b"audio data")
        assert result == mock_path_instance


@pytest.mark.asyncio
@patch("sab_ia.services.prediction_service.create_temp_file")
@patch("sab_ia.services.prediction_service.ensure_mono_audio")
@patch("sab_ia.services.prediction_service.run_birdnet_prediction")
@patch("sab_ia.services.prediction_service.aggregate_species_data")
@patch("sab_ia.services.prediction_service.format_result")
@patch("sab_ia.services.prediction_service.remove_temp_file")
async def test_process_audio_file(
    mock_remove, mock_format, mock_aggregate, mock_run, mock_ensure, mock_create
):

    temp_path = Path("/tmp/test.wav")
    mock_create.return_value = temp_path

    mock_predictions = MagicMock()
    mock_run.return_value = mock_predictions

    mock_species_data = [{"scientific_name": "Test", "confidence": 0.9}]
    mock_aggregate.return_value = mock_species_data

    expected_result = {"species_count": 1, "species": mock_species_data}
    mock_format.return_value = expected_result

    mock_upload = MagicMock()

    result = await process_audio_file(mock_upload, 0.1)

    mock_create.assert_called_once_with(mock_upload)
    mock_ensure.assert_called_once_with(temp_path)
    mock_run.assert_called_once_with(temp_path, 0.1)
    mock_aggregate.assert_called_once_with(mock_predictions)
    mock_format.assert_called_once_with(mock_species_data)
    mock_remove.assert_called_once_with(temp_path)

    assert result == expected_result
