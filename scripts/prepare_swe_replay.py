"""Prepare reusable published SWE histories; write all data outside this repository."""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
from pathlib import Path
from urllib.request import urlopen

INPUT_HASHES = {
    "protocol": "eacf3628cda8d7accc2e39b20fce687b1deb1cb0b0f470e8b11c8dd4517f71f8",
    "tools": "95c0546bbc92c4a6a9e6780ae825d78e05e2c436211603790502303b7f74fcf8",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def external_directory(value):
    path = Path(value).resolve()
    if path.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError(
            "Place generated inputs and results outside the plugin repository"
        )
    path.mkdir(parents=True, exist_ok=False)
    return path


def convert(query):
    if not isinstance(query, list) or len(query) < 2:
        raise ValueError("incomplete_published_history")
    if query[0].get("role") != "system" or query[1].get("role") != "user":
        raise ValueError("missing_original_initial_history")
    messages, awaiting = [], set()
    for source in query:
        role, content = source.get("role"), source.get("content")
        if role not in ("system", "user", "assistant", "tool") or not isinstance(
            content, str
        ):
            raise ValueError("invalid_role_or_missing_content")
        message = {"role": role, "content": content}
        calls = source.get("tool_calls", [])
        if calls:
            if role != "assistant":
                raise ValueError("tool_call_outside_assistant")
            message["tool_calls"] = []
            for call in calls:
                function = call["function"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("tool_arguments_not_object")
                if not call.get("id") or call["id"] in awaiting:
                    raise ValueError("invalid_tool_call_identity")
                awaiting.add(call["id"])
                message["tool_calls"].append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": function["name"], "arguments": arguments},
                    }
                )
        if role == "tool":
            ids = source.get("tool_call_ids", [])
            if len(ids) != 1 or ids[0] not in awaiting:
                raise ValueError("ambiguous_or_unmatched_tool_response")
            message["tool_call_id"] = ids[0]
            awaiting.remove(ids[0])
        if role == "assistant" and not content and not calls:
            raise ValueError("empty_published_assistant")
        messages.append(message)
    if awaiting:
        raise ValueError("unanswered_published_tool_call")
    return messages


def select_sources(protocol):
    swe = protocol["datasets"]["swe"]
    sources = list(swe["sources"])
    known = {source["task_id"] for source in sources}
    extra = sorted(
        set(swe["available_published_task_ids"]) - known,
        key=lambda task: hashlib.sha256(
            ("kv-tiering-pressure-pool-v1\0" + task).encode()
        ).hexdigest(),
    )[:96]
    for task in extra:
        sources.append(
            {
                "task_id": task,
                "cache_key": "swe/" + task + ".traj",
                "url": "https://scaleapi-results.s3.amazonaws.com/swe-bench-pro/"
                "claude-sonnet-4-paper/traj/" + task + "/" + task + ".traj",
            }
        )
    if len(known) != 32 or len(sources) != 128:
        raise ValueError(
            "Unexpected frozen task pool; do not silently change selection"
        )
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-cache", type=Path)
    args = parser.parse_args()
    for name, expected in INPUT_HASHES.items():
        if hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned benchmark input mismatch: " + name)
    protocol = json.loads(args.protocol.read_text())
    tools = json.loads(args.tools.read_text())
    for name, expected in protocol["tokenizer_files"].items():
        if hashlib.sha256((args.model / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Tokenizer/model configuration mismatch: " + name)
    for name in ("transformers", "tokenizers"):
        if (
            importlib.metadata.version(name)
            != protocol["tokenizer_implementation"][name]
        ):
            raise ValueError("Tokenizer implementation mismatch: " + name)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    root = external_directory(args.output)
    sources_dir, prompts_dir = root / "sources", root / "prompts"
    sources_dir.mkdir()
    prompts_dir.mkdir()
    sources = select_sources(protocol)
    (root / "selection.json").write_text(json.dumps(sources, indent=2) + "\n")

    def fetch(source):
        name = Path(source["cache_key"]).name
        cached = args.source_cache / name if args.source_cache else None
        if cached and cached.is_file():
            payload = cached.read_bytes()
        else:
            url = source["url"].replace(
                "s3.amazonaws.com", "s3.us-west-1.amazonaws.com"
            )
            with urlopen(url, timeout=60) as response:
                payload = response.read()
        sha = hashlib.sha256(payload).hexdigest()
        if source.get("sha256") and (
            sha != source["sha256"] or len(payload) != source["size"]
        ):
            raise ValueError("Frozen source mismatch: " + source["task_id"])
        (sources_dir / name).write_bytes(payload)
        return dict(source, sha256=sha, size=len(payload))

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        acquired = list(pool.map(fetch, sources))
    (root / "sources-lock.json").write_text(json.dumps(acquired, indent=2) + "\n")
    reference = {
        (r["task_id"], r["source_index"]): r
        for r in protocol["datasets"]["swe"]["requests"]
    }
    tasks, exclusions, excluded_tasks = [], [], []
    for task_index, source in enumerate(acquired):
        trace = json.loads((sources_dir / Path(source["cache_key"]).name).read_bytes())
        requests, over_context = [], False
        for index, step in enumerate(trace["trajectory"]):
            try:
                messages = convert(step.get("query"))
            except (ValueError, KeyError, TypeError) as exc:
                exclusions.append(
                    {"task_id": source["task_id"], "index": index, "reason": str(exc)}
                )
                continue
            tokens = tokenizer.apply_chat_template(
                messages,
                tools=tools["tools"],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
                return_dict=False,
            )
            if not tokens or not isinstance(tokens[0], int):
                raise ValueError("Unexpected tokenization output")
            over_context |= len(tokens) + 32 > 261120
            expected = reference.get((source["task_id"], index))
            if expected and len(tokens) != expected["prompt_tokens"]:
                raise ValueError("Frozen prompt-length checkpoint mismatch")
            path = prompts_dir / f"task-{task_index:02d}-step-{index:03d}.json"
            path.write_text(json.dumps(tokens) + "\n")
            requests.append(
                {
                    "task_id": source["task_id"],
                    "source_index": index,
                    "prompt_file": str(path),
                    "prompt_tokens": len(tokens),
                    "prompt_sha256": digest(tokens),
                }
            )
        if over_context:
            excluded_tasks.append(source["task_id"])
        elif requests:
            tasks.append(requests)
        else:
            raise ValueError("Selected task has no complete published history")
    if not tasks:
        raise ValueError("No eligible tasks")
    phases = [
        {
            "name": f"swe-turn-{turn:03d}",
            "concurrency": len(tasks),
            "requests": [task[turn] for task in tasks if len(task) > turn],
        }
        for turn in range(max(map(len, tasks)))
    ]
    plan = {"output_len": 32, "phases": phases}
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (root / "preparation.json").write_text(
        json.dumps(
            {
                "protocol_sha256": hashlib.sha256(
                    args.protocol.read_bytes()
                ).hexdigest(),
                "tools_sha256": hashlib.sha256(args.tools.read_bytes()).hexdigest(),
                "eligible_tasks": len(tasks),
                "excluded_tasks": excluded_tasks,
                "excluded_steps": exclusions,
                "plan_sha256": digest(plan),
            },
            indent=2,
        )
        + "\n"
    )
    print(root / "plan.json")


if __name__ == "__main__":
    main()
