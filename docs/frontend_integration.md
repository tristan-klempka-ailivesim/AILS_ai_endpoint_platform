# AI Label API Frontend Integration

Base URL:

```text
http://192.168.1.12:8080
```

Readiness check:

```http
GET /health
```

Asset parts label endpoint:

```http
POST /asset-parts/label
```

Headers:

```http
Content-Type: application/json
X-Request-ID: optional client-generated request ID
```

Request body:

```json
{
  "image": "data:image/png;base64,...",
  "segments": [
    {
      "id": 0,
      "color_name": "red",
      "rgb": [216, 38, 38]
    }
  ]
}
```

Rules:

- Send the image as a data URL, not multipart/form-data.
- Supported image media types: `image/png`, `image/jpeg`, `image/webp`.
- Do not send a prompt. The server owns the prompt.
- `segments` must contain at least one item.
- Segment `id` values must be unique integers.
- `rgb` must contain exactly three integers from `0` to `255`.
- The response order follows the request segment order.

Success response:

```json
[
  {
    "id": 0,
    "name": "main_hull",
    "material": "fiberglass"
  }
]
```

Success response headers:

```http
X-Request-ID: <request_id>
X-Prompt-Version: label_v1
```

Error response:

```json
{
  "detail": "error message"
}
```

Status codes:

- `400`: invalid image data or image size limits.
- `422`: request schema validation failed.
- `502`: model returned invalid output or non-2xx response.
- `503`: model server unavailable.
- `504`: model request timed out.

Health response:

```json
{
  "status": "ok",
  "model_ready": true,
  "metadata_ready": false,
  "label_ready": true
}
```

`metadata_ready` is currently `false`; metadata generation is planned for `POST /asset/metadata`.
