#!/usr/bin/env python3

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LIST_URL = "https://api.leetgpu.com/api/v1/challenges"
DETAIL_URL_TEMPLATE = "https://api.leetgpu.com/api/v1/challenges/{challenge_id}"
DEFAULT_OUTPUT = Path("web/challenges.generated.js")


def fetch_json(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))

    curl_output = subprocess.check_output(
        [
            "curl",
            "-sS",
            "-A",
            "Mozilla/5.0",
            "--max-time",
            "120",
            url,
        ],
        text=True,
    )
    if not curl_output.strip():
        raise RuntimeError(f"Empty response while fetching {url}") from last_error
    return json.loads(curl_output)


def title_case(value: str) -> str:
    if not value:
        return "Unknown"
    return value[:1].upper() + value[1:]


def zero_pad(challenge_id: int) -> str:
    return f"{challenge_id:03d}"


def build_placeholder_entry(challenge_id: int, reason: str) -> dict:
    return {
        "id": zero_pad(challenge_id),
        "numericId": challenge_id,
        "title": "Unavailable",
        "difficulty": "Unknown",
        "accessTier": "unknown",
        "available": False,
        "listTag": "Missing API",
        "kind": "metadata-only",
        "supportsRun": False,
        "supportsCudaCompile": False,
        "unsupportedReason": reason,
        "specHtml": (
            "<p>This challenge ID is not currently available from "
            "<code>api.leetgpu.com</code>.</p>"
        ),
        "starterCode": "",
        "signatureHint": "",
        "examples": [],
        "judgeContract": reason,
        "submissionHint": reason,
        "testcaseHint": "This challenge cannot be run in the browser judge right now.",
        "testcaseHelp": reason,
        "resultTitle": "Judge Status",
        "resultSubtitle": reason,
    }


@dataclass
class ChallengeDetail:
    challenge_id: int
    payload: Optional[dict]
    error: Optional[str] = None


def fetch_detail(challenge_id: int) -> ChallengeDetail:
    url = DETAIL_URL_TEMPLATE.format(challenge_id=challenge_id)
    try:
        payload = fetch_json(url)
        if isinstance(payload, dict) and payload.get("error") == "challenge not found":
            return ChallengeDetail(challenge_id=challenge_id, payload=None, error="Challenge not found in API.")
        return ChallengeDetail(challenge_id=challenge_id, payload=payload)
    except HTTPError as exc:
        if exc.code == 404:
            return ChallengeDetail(challenge_id=challenge_id, payload=None, error="Challenge not found in API.")
        return ChallengeDetail(challenge_id=challenge_id, payload=None, error=f"HTTP {exc.code} while fetching challenge.")
    except URLError as exc:
        return ChallengeDetail(challenge_id=challenge_id, payload=None, error=f"Network error: {exc.reason}")


def extract_cuda_starter(payload: dict) -> Optional[dict]:
    for starter in payload.get("starterCode", []):
        if starter.get("language") == "cuda":
            return starter
    return None


def extract_signature_hint(starter_code: str) -> str:
    if not starter_code:
        return ""

    lines = [line.strip() for line in starter_code.splitlines() if line.strip()]
    if not lines:
        return ""

    fragments = []
    started = False
    for line in lines:
        if line.startswith("extern \"C\"") or line.startswith("void solve("):
            started = True
        if started:
            fragments.append(line)
            if line.endswith("{") or line.endswith("}") or line.endswith(");"):
                break
            if len(" ".join(fragments)) > 180:
                break
    return " ".join(fragments).strip()


def build_entry(challenge_id: int, payload: dict) -> dict:
    cuda_starter = extract_cuda_starter(payload)
    has_cuda_starter = cuda_starter is not None
    unsupported_reason = (
        "The browser judge is not implemented for this challenge yet."
        if has_cuda_starter
        else "This challenge does not provide a CUDA starter in the API."
    )

    return {
        "id": zero_pad(challenge_id),
        "numericId": challenge_id,
        "title": payload.get("title", f"Challenge {challenge_id}"),
        "difficulty": title_case(payload.get("difficultyLevel", "unknown")),
        "accessTier": payload.get("accessTier", "unknown"),
        "available": True,
        "listTag": f"LeetGPU #{challenge_id}",
        "kind": "metadata-only",
        "supportsRun": False,
        "supportsCudaCompile": has_cuda_starter,
        "unsupportedReason": unsupported_reason,
        "specHtml": payload.get("spec", ""),
        "starterCode": (cuda_starter or {}).get("fileContent", ""),
        "signatureHint": extract_signature_hint((cuda_starter or {}).get("fileContent", "")),
        "examples": [],
        "judgeContract": unsupported_reason,
        "submissionHint": (
            "You can inspect the official CUDA starter and compile your own .cu file in-browser."
            if has_cuda_starter
            else "This challenge currently has no CUDA starter, so the browser CUDA flow is disabled."
        ),
        "testcaseHint": "A browser-side judge is not available for this challenge yet.",
        "testcaseHelp": unsupported_reason,
        "resultTitle": "Judge Status",
        "resultSubtitle": unsupported_reason,
    }


def generate_catalog(start_id: int, end_id: int) -> dict:
    published_list = fetch_json(LIST_URL)
    published_ids = {
        int(item["id"])
        for item in published_list.get("challenges", [])
        if isinstance(item, dict) and "id" in item
    }

    challenges = []
    missing_ids = []
    for challenge_id in range(start_id, end_id + 1):
        detail = fetch_detail(challenge_id)
        if detail.payload is None:
            missing_ids.append(challenge_id)
            reason = detail.error or "Challenge not found in API."
            if challenge_id in published_ids:
                reason = f"Challenge metadata exists in the list API but detail fetch failed: {reason}"
            challenges.append(build_placeholder_entry(challenge_id, reason))
            continue

        challenges.append(build_entry(challenge_id, detail.payload))

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceListUrl": LIST_URL,
        "sourceDetailUrlTemplate": DETAIL_URL_TEMPLATE,
        "range": {"start": start_id, "end": end_id},
        "missingIds": missing_ids,
        "challengeCount": len(challenges),
        "challenges": challenges,
    }


def write_js_catalog(catalog: dict, output_path: Path) -> None:
    payload = json.dumps(catalog, ensure_ascii=False, indent=2)
    js = (
        "// Generated by tools/generate_leetgpu_catalog.py\n"
        "window.LEETGPU_CATALOG = "
        + payload
        + ";\n"
        "window.LEETGPU_GENERATED_PROBLEMS = window.LEETGPU_CATALOG.challenges;\n"
    )
    output_path.write_text(js, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local LeetGPU challenge catalog.")
    parser.add_argument("--start", type=int, default=1, help="First challenge ID to fetch.")
    parser.add_argument("--end", type=int, default=74, help="Last challenge ID to fetch.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JS file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = generate_catalog(args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_js_catalog(catalog, args.output)
    print(
        f"Wrote {catalog['challengeCount']} challenge entries to {args.output} "
        f"(missing: {catalog['missingIds']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
