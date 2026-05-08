import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, Response

from services.inference.openai_client import OpenAIModelClient
from shared.schemas import LabelOutput, LabelRequest, validate_image_data_url

router = APIRouter()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an expert AI classifying 3D scene segments and predicting materials. "
    "You meticulously analyze images and format responses exactly as specified."
)


def build_label_prompt(label_request: LabelRequest, prompt_template: str) -> str:
    segment_lines = []
    for segment in label_request.segments:
        rgb = ",".join(str(value) for value in segment.rgb)
        segment_lines.append(f"ID {segment.id} (shown in {segment.color_name}, RGB={rgb})")
    segment_text = "; ".join(segment_lines)

    return prompt_template.format(
        segment_count=len(label_request.segments),
        segments=segment_text,
    )


def extract_json_array(raw_content: str, request_id: str | None = None) -> object:
    decoder = json.JSONDecoder()
    stripped = raw_content.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for index, character in enumerate(raw_content):
        if character != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed

    preview = raw_content[:500].replace("\n", "\\n")
    logger.warning(
        "model output was not valid JSON array request_id=%s preview=%s",
        request_id,
        preview,
    )
    raise HTTPException(status_code=502, detail="model output was not valid JSON")


def parse_label_output(
    raw_content: str,
    requested_ids: list[int],
    request_id: str | None = None,
) -> list[LabelOutput]:
    parsed = extract_json_array(raw_content, request_id=request_id)

    if not isinstance(parsed, list):
        raise HTTPException(status_code=502, detail="model output must be a JSON array")

    try:
        labels = [LabelOutput.model_validate(item) for item in parsed]
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="model output did not match label schema",
        ) from exc

    returned_ids = [label.id for label in labels]
    if len(returned_ids) != len(set(returned_ids)):
        raise HTTPException(
            status_code=502,
            detail="model output contained duplicate segment IDs",
        )

    if set(returned_ids) != set(requested_ids):
        raise HTTPException(
            status_code=502,
            detail="model output segment IDs did not match request",
        )

    if len(labels) != len(requested_ids):
        raise HTTPException(
            status_code=502,
            detail="model output did not include one label per segment",
        )

    by_id = {label.id: label for label in labels}
    return [by_id[segment_id] for segment_id in requested_ids]


async def label_asset_parts(
    label_request: LabelRequest,
    request: Request,
    response: Response,
) -> list[LabelOutput]:
    settings = request.app.state.settings
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Prompt-Version"] = settings.prompt_version

    validate_image_data_url(
        label_request.image,
        max_decoded_bytes=settings.max_decoded_image_bytes,
        max_pixels=settings.max_image_pixels,
    )

    client = OpenAIModelClient(settings)
    prompt_text = build_label_prompt(label_request, settings.label_prompt_template)
    raw_content = await client.chat_completion(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt_text,
        image_data_url=label_request.image,
        request_id=request_id,
    )
    requested_ids = [segment.id for segment in label_request.segments]
    return parse_label_output(raw_content, requested_ids, request_id=request_id)


router.add_api_route(
    "/asset-parts/label",
    label_asset_parts,
    methods=["POST"],
    response_model=list[LabelOutput],
)
router.add_api_route(
    "/label",
    label_asset_parts,
    methods=["POST"],
    response_model=list[LabelOutput],
    deprecated=True,
)
