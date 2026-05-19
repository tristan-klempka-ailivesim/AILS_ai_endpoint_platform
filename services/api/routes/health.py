from fastapi import APIRouter, HTTPException, Request

from services.inference.openai_client import OpenAIModelClient

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    client = OpenAIModelClient(settings)
    model_ready = False

    try:
        models = await client.list_models()
        model_ready = settings.model_server_model in models
    except HTTPException:
        model_ready = False
    except Exception:  # health should report degraded state, not fail the API
        model_ready = False

    return {
        "status": "ok",
        "model_ready": model_ready,
        "metadata_ready": False,
        "label_ready": model_ready,
    }
