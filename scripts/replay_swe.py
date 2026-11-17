"""Replay prepared SWE token histories against one already running, owned server."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from prepare_swe_replay import digest, external_directory


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


async def replay(args):
    import aiohttp

    plan = json.loads(args.plan.read_text())
    # Verify all inputs before sending anything; use the same plan for both arms.
    for phase in plan["phases"]:
        for item in phase["requests"]:
            prompt = json.loads(Path(item["prompt_file"]).read_text())
            if (
                digest(prompt) != item["prompt_sha256"]
                or len(prompt) != item["prompt_tokens"]
            ):
                raise ValueError("Prepared input mismatch")
    root = external_directory(args.output)
    write(
        root / "invocation.json",
        {
            "base_url": args.base_url,
            "model": args.model,
            "plan_sha256": digest(plan),
            "timeout_seconds": 1800,
        },
    )
    records, phases = [], []
    whole_start = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=1800, sock_connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def metrics(name):
            async with session.get(args.base_url + "/metrics") as response:
                response.raise_for_status()
                (root / name).write_text(await response.text())

        for phase in plan["phases"]:
            name = phase["name"]
            await metrics(name + "-before.prom")
            semaphore = asyncio.Semaphore(phase["concurrency"])
            phase_records = []
            start = time.monotonic()

            async def request(index, item):
                async with semaphore:
                    key = f"{name}-{index:04d}"
                    prompt = json.loads(Path(item["prompt_file"]).read_text())
                    body = {
                        "model": args.model,
                        "prompt": prompt,
                        "max_tokens": plan["output_len"],
                        "temperature": 0,
                        "ignore_eos": True,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    first, usage, finish = None, None, None
                    done, text, status, error = 0, "", None, None
                    raw = bytearray()
                    began = time.monotonic()
                    try:
                        async with session.post(
                            args.base_url + "/v1/completions", json=body
                        ) as response:
                            status = response.status
                            response.raise_for_status()
                            async for line in response.content:
                                raw.extend(line)
                                if not line.startswith(b"data:"):
                                    continue
                                data = line[5:].strip()
                                if data == b"[DONE]":
                                    done += 1
                                    continue
                                event = json.loads(data)
                                if event.get("error"):
                                    raise RuntimeError(str(event["error"]))
                                usage = event.get("usage") or usage
                                for choice in event.get("choices", []):
                                    if choice.get("text"):
                                        if first is None:
                                            first = time.monotonic()
                                        text += choice["text"]
                                    finish = choice.get("finish_reason") or finish
                    except Exception as exc:
                        error = repr(exc)
                    elapsed = time.monotonic() - began
                    valid = bool(
                        error is None
                        and status == 200
                        and done == 1
                        and finish == "length"
                        and first is not None
                        and usage
                        and usage["prompt_tokens"] == item["prompt_tokens"]
                        and usage["completion_tokens"] == plan["output_len"]
                        and usage["total_tokens"]
                        == usage["prompt_tokens"] + usage["completion_tokens"]
                    )
                    record = {
                        "key": key,
                        **item,
                        "valid": valid,
                        "error": error,
                        "http_status": status,
                        "usage": usage,
                        "wall_s": elapsed,
                        "ttft_s": None if first is None else first - began,
                        "tpot_s": None
                        if first is None or not valid
                        else (elapsed - (first - began))
                        / max(1, usage["completion_tokens"] - 1),
                        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                    (root / (key + ".sse")).write_bytes(raw)
                    with (root / "requests.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    phase_records.append(record)
                    records.append(record)

            await asyncio.gather(
                *(request(i, item) for i, item in enumerate(phase["requests"]))
            )
            wall = time.monotonic() - start
            await asyncio.sleep(1)
            await metrics(name + "-after.prom")
            summary = {
                "name": name,
                "wall_s": wall,
                "requests": len(phase_records),
                "failed": sum(not r["valid"] for r in phase_records),
                "output_tokens": sum(
                    r["usage"]["completion_tokens"] for r in phase_records if r["valid"]
                ),
            }
            phases.append(summary)
            write(root / (name + "-summary.json"), summary)
            if summary["failed"]:
                write(
                    root / "incomplete.json",
                    {"phase": name, "failed": summary["failed"]},
                )
                raise RuntimeError(
                    "Incomplete arm; preserve failures and do not report a complete run"
                )
    write(
        root / "complete.json",
        {
            "requests": len(records),
            "whole_client_wall_s": time.monotonic() - whole_start,
            "phase_wall_s": sum(p["wall_s"] for p in phases),
            "output_tokens": sum(p["output_tokens"] for p in phases),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18236")
    parser.add_argument("--model", default="qwen3.5-35b-a3b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    asyncio.run(replay(args))


if __name__ == "__main__":
    main()
