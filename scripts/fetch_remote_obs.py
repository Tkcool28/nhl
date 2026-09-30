"""Fetch market observation files synced by the frontend app.

The app commits data/market_log/obs_*.jsonl directly to remote main; the
local morning pipeline never sees them, so settle_logs() cannot settle them.
Run this before daily_slate.py: it downloads any obs files present on
remote but missing locally (remote is authoritative; local copies are
read-only snapshots for settlement).

Usage: python scripts/fetch_remote_obs.py
"""
import base64
import json
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import add_surrogate_to_request, read_response_body

BASE = "https://api.github.com/repos/Tkcool28/nhl"
REPO = Path("/home/hatch/workspace/nhl")


def api(path):
    req = urllib.request.Request(
        BASE + path,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "muse/1.0"})
    add_surrogate_to_request(req, "custom.github",
                             allowed_hosts=["api.github.com"])
    with urllib.request.urlopen(req) as r:
        return json.loads(read_response_body(r).decode())


def main():
    log_dir = REPO / "data" / "market_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        entries = api("/contents/data/market_log")
    except Exception as e:
        print(f"fetch_remote_obs: could not list remote dir: {e}")
        return
    n = 0
    for e in entries:
        name = e.get("name", "")
        if not name.startswith("obs_") or not name.endswith(".jsonl"):
            continue
        dest = log_dir / name
        blob = api(f"/contents/data/market_log/{name}")
        content = base64.b64decode(blob["content"])
        if dest.exists() and dest.read_bytes() == content:
            continue  # already have this exact snapshot
        dest.write_bytes(content)
        print(f"fetched {name} ({blob.get('size', '?')} bytes)")
        n += 1
    if not n:
        print("fetch_remote_obs: nothing new")


if __name__ == "__main__":
    main()
