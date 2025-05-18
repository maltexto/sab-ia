# 🦜 Sab_ia: a BirdNET Wrapper

An API that identifies bird species from audio recordings using the BirdNET machine learning model.

## 📁 Project Structure

```
.
├── src/
│   └── sab_ia/
│       ├── main.py              # FastAPI app entry point
│       ├── api/
│       │   └── routes/          # API route definitions
│       │       └── predictions.py
│       └── services/            # Business logic
│           └── prediction_service.py
├── Dockerfile                   # Docker configuration
└── pyproject.toml              # Project dependencies
```

## 🚀 Running with Docker

### Prerequisites

- 🐳 Docker installed on your system

### Steps to Run

1. **Clone the repository (if not already done)**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Build the Docker image**

   ```bash
   docker buildx build -t sab_ia .
   ```

3. **Run the container**

   ```bash
   docker run -p 8000:8000 sab_ia
   ```

4. **Access the API**
   
   The API will be available at http://localhost:8000

## 🔍 API Usage

### Predict Bird Species Endpoint

**Endpoint:** `POST /rest/v1/predict`

This endpoint accepts an audio file and returns a list of predicted bird species with confidence scores.

#### 📡 Using curl

```bash
curl -X POST http://localhost:8000/rest/v1/predict \
     -H "accept: application/json" \
     -F "audio_file=@/path/to/your/bird_recording.wav"
```

#### 🔄 Response Format

```json
{
  "species_count": 3,
  "species": [
    {
      "scientific_name": "Carduelis carduelis",
      "common_name": "European Goldfinch",
      "confidence": 0.85,
      "occurrences": 4
    },
    {
      "scientific_name": "Parus major",
      "common_name": "Great Tit",
      "confidence": 0.76,
      "occurrences": 2
    },
    {
      "scientific_name": "Turdus merula",
      "common_name": "Common Blackbird",
      "confidence": 0.62,
      "occurrences": 1
    }
  ]
}
```

## 🧠 How It Works

1. The API receives an audio file through the `/rest/v1/predict` endpoint
2. The audio is converted to mono format if necessary
3. BirdNET processes the audio file to detect bird species
4. The predictions are aggregated across time intervals
5. The results are formatted and returned as JSON

## 🛠️ Technical Details

- The API is built with FastAPI
- Bird species identification is powered by BirdNET
- Audio processing uses pydub and ffmpeg
- The API automatically handles stereo-to-mono conversion
- Multisample prediction increases accuracy

## 📊 Response Fields

- `species_count`: Total number of unique species detected
- `species`: List of detected species, each containing:
  - `scientific_name`: Latin/scientific name of the species
  - `common_name`: Common name of the species
  - `confidence`: Highest confidence score (0-1) for this species
  - `occurrences`: Number of time segments where this species was detected

## ⚠️ Notes

- Audio files should be clear recordings of bird sounds
- Longer recordings typically yield better results
- The API works best with WAV files, but can process other formats
- Processing time varies based on the length of the audio file
