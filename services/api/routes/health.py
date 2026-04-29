from fastapi import APIRouter, HTTPException, Request

from services.inference.openai_client import OpenAIModelClient

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    client = OpenAIModelClient(settings)
    reachable = False
    model_match = False
    warning = None

    try:
        models = await client.list_models()
        reachable = True
        model_match = settings.model_server_model in models
        if not model_match:
            warning = "configured MODEL_SERVER_MODEL was not found in /models response"
    except HTTPException as exc:
        warning = str(exc.detail)
    except Exception as exc:  # health should report degraded state, not fail the API
        warning = f"model server readiness check failed: {exc}"

    body: dict[str, object] = {
        "status": "ok",
        "model": settings.model_server_model,
        "prompt_version": settings.prompt_version,
        "model_server_base_url": settings.model_server_base_url,
        "model_server_reachable": reachable,
        "model_server_model_match": model_match,
    }
    if warning:
        body["warning"] = warning
    return body
