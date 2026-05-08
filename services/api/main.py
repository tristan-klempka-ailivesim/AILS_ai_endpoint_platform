import json
import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from starlette.concurrency import iterate_in_threadpool

from services.api.routes.health import router as health_router
from services.api.routes.label import router as label_router
from shared.settings import get_settings

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


def _failure_reason_from_body(body: bytes) -> str | None:
    if not body:
        return None

    try:
        data = json.loads(body)
    except ValueError:
        return body[:500].decode("utf-8", errors="replace")

    if isinstance(data, dict) and data.get("detail"):
        return str(data["detail"])
    return None


async def log_request_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "api request failed request_id=%s method=%s path=%s status=%s latency_ms=%.2f "
            "failure_reason=%s",
            request_id,
            request.method,
            request.url.path,
            500,
            latency_ms,
            "unhandled exception",
        )
        raise

    failure_reason = None
    if response.status_code >= 400:
        chunks = [chunk async for chunk in response.body_iterator]
        failure_reason = _failure_reason_from_body(b"".join(chunks))
        response.body_iterator = iterate_in_threadpool(iter(chunks))

    response.headers["X-Request-ID"] = request_id
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "api request completed request_id=%s method=%s path=%s status=%s latency_ms=%.2f "
        "failure_reason=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        failure_reason or "",
    )
    return response


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="AILS AI Endpoint Platform", version="0.1.0")
    app.state.settings = settings
    app.middleware("http")(log_request_middleware)

    app.include_router(health_router)
    app.include_router(label_router)

    return app


app = create_app()
