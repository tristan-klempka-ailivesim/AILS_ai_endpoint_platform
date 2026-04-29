#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import httpx

SEGMENTS = [
    {"id": 0, "color_name": "red", "rgb": [216, 38, 38]},
    {"id": 1, "color_name": "green", "rgb": [48, 232, 101]},
    {"id": 2, "color_name": "purple", "rgb": [168, 61, 244]},
    {"id": 3, "color_name": "gold/yellow", "rgb": [228, 212, 103]},
    {"id": 4, "color_name": "sky blue", "rgb": [25, 195, 229]},
    {"id": 5, "color_name": "hot pink", "rgb": [243, 36, 148]},
]


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


async def run_one(
    client: httpx.AsyncClient,
    *,
    api_url: str,
    payload: dict,
    index: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    request_id = f"load-test-{index:04d}"
    start = time.perf_counter()
    async with semaphore:
        try:
            response = await client.post(
                f"{api_url.rstrip('/')}/asset-parts/label",
                json=payload,
                headers={"X-Request-ID": request_id},
            )
            latency = time.perf_counter() - start
            body = response.json()
            ok = response.status_code == 200 and isinstance(body, list)
            return {
                "index": index,
                "ok": ok,
                "status": response.status_code,
                "latency": latency,
                "request_id": response.headers.get("X-Request-ID"),
                "prompt_version": response.headers.get("X-Prompt-Version"),
                "body": body,
            }
        except Exception as exc:
            latency = time.perf_counter() - start
            return {
                "index": index,
                "ok": False,
                "status": None,
                "latency": latency,
                "request_id": request_id,
                "prompt_version": None,
                "body": {"error": str(exc)},
            }


async def run_load(args: argparse.Namespace) -> int:
    payload = {"image": image_to_data_url(Path(args.image)), "segments": SEGMENTS}
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            run_one(
                client,
                api_url=args.api_url,
                payload=payload,
                index=index,
                semaphore=semaphore,
            )
            for index in range(args.requests)
        ]
        results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started

    for result in results:
        status = "ok" if result["ok"] else "fail"
        print(
            f"{result['index']:04d} {status} status={result['status']} "
            f"latency={result['latency']:.3f}s request_id={result['request_id']}",
        )
        if not result["ok"]:
            print(json.dumps(result["body"], indent=2))

    latencies = [result["latency"] for result in results]
    successes = [result for result in results if result["ok"]]
    success_rate = len(successes) / len(results)
    p50_latency = statistics.median(latencies)
    max_latency = max(latencies)
    print()
    print(f"requests={len(results)} concurrency={args.concurrency} successes={len(successes)}")
    print(f"success_rate={success_rate:.3f}")
    print(f"total_elapsed={elapsed:.3f}s")
    print(f"latency_min={min(latencies):.3f}s")
    print(f"latency_p50={p50_latency:.3f}s")
    print(f"latency_max={max_latency:.3f}s")

    failures = []
    if success_rate < args.min_success_rate:
        failures.append(
            f"success_rate {success_rate:.3f} < threshold {args.min_success_rate:.3f}"
        )
    if p50_latency > args.max_p50_latency:
        failures.append(
            f"latency_p50 {p50_latency:.3f}s > threshold {args.max_p50_latency:.3f}s"
        )
    if max_latency > args.max_latency:
        failures.append(f"latency_max {max_latency:.3f}s > threshold {args.max_latency:.3f}s")

    if failures:
        print()
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print()
    print("PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concurrent load test for the /asset-parts/label endpoint."
    )
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument("--image", default="tmp/combined_prompt.png")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--max-p50-latency", type=float, default=10.0)
    parser.add_argument("--max-latency", type=float, default=20.0)
    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.min_success_rate < 0 or args.min_success_rate > 1:
        parser.error("--min-success-rate must be between 0 and 1")
    if args.max_p50_latency <= 0:
        parser.error("--max-p50-latency must be greater than 0")
    if args.max_latency <= 0:
        parser.error("--max-latency must be greater than 0")

    return asyncio.run(run_load(args))


if __name__ == "__main__":
    raise SystemExit(main())
