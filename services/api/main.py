from fastapi import FastAPI

from services.api.routes.health import router as health_router
from services.api.routes.label import router as label_router
from shared.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="AILS AI Endpoint Platform", version="0.1.0")
    app.state.settings = settings

    app.include_router(health_router)
    app.include_router(label_router)

    return app


app = create_app()
