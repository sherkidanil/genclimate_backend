from fastapi import FastAPI
from routers.forecast import router as forecast_router
# from routers.weather_plots import router as plots_router
from core.logger import configure_logging

configure_logging()

app = FastAPI(
    title="GenClimate Forecast Service",
    version="1.0.0",
    description="Point forecasts from AIFS medium‑range and S2S models",
)

app.include_router(forecast_router, prefix="/v1")
# app.include_router(plots_router, prefix="/v1")


@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
