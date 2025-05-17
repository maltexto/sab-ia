from fastapi import FastAPI
from sab_ia.api.routes import predictions_router as prediction

app = FastAPI(title="SAB_IA")
app.include_router(prediction)

@app.get("/")
async def root():
    return {"status": "piu"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)