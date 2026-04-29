FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

COPY services ./services
COPY shared ./shared
COPY prompts ./prompts

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
