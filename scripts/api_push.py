"""Commit local changes and push to GitHub via the git-database API.

The workspace has no git credential helper, so pushes go through blobs ->
tree -> commit -> ref update (see AGENTS.md). The frontend app syncs market
observations directly to remote main, so the remote tree routinely differs
from our local parent tree. Instead of aborting on any mismatch, we walk the
remote ancestry for a commit whose tree matches a local commit, then replay
our diff on top of the remote head. Files under data/market_log/obs_* are
app-owned: the remote copy always wins and is never overwritten by us.

Fails closed (aborts, local changes kept staged) when no common tree is
found or the remote moves during the push.

Usage: python scripts/api_push.py "commit message" [pathspec...]
"""
import base64
import json
import subprocess
import sys
import urllib.request

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import add_surrogate_to_request, read_response_body

BASE = "https://api.github.com/repos/Tkcool28/nhl"
REPO = "/home/hatch/workspace/nhl"

# App-owned paths: the frontend sync writes these directly to remote main.
# Never overwrite the remote copy with ours.
APP_OWNED_PREFIXES = ("data/market_log/obs_",)


def api(path, method="GET", payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=body, method=method,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "muse/1.0"})
    if body:
        req.add_header("Content-Type", "application/json")
    add_surrogate_to_request(req, "custom.github",
                             allowed_hosts=["api.github.com"])
    with urllib.request.urlopen(req) as r:
        return json.loads(read_response_body(r).decode())


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def find_common_base(remote):
    """Walk remote ancestry for a tree that exists in local history."""
    local_trees = {}
    for c in git("log", "--format=%H").split("\n"):
        local_trees[git("rev-parse", f"{c}^{{tree}}")] = c
    node = remote
    for _ in range(15):
        cm = api(f"/git/commits/{node}")
        t = cm["tree"]["sha"]
        if t in local_trees:
            return local_trees[t]
        parents = cm.get("parents") or []
        if not parents:
            break
        node = parents[0]["sha"]
    return None


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else "update"
    paths = sys.argv[2:] or ["-A"]
    git("add", *paths)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"],
                            cwd=REPO).returncode != 0
    if not staged:
        print("nothing to push")
        return
    git("commit", "-q", "-m", msg)
    head = git("rev-parse", "HEAD")

    remote = api("/git/ref/heads/main")["object"]["sha"]
    rtree = api(f"/git/commits/{remote}")["tree"]["sha"]

    base = find_common_base(remote)
    if base is None:
        subprocess.run(["git", "reset", "--soft", f"{head}~1"], cwd=REPO,
                       check=True)
        sys.exit("ABORT: no common tree with remote main; reset to "
                 "pre-commit state")
    if base != head:
        print(f"remote moved; replaying {base[:8]}..{head[:8]} "
              f"on top of {remote[:8]}", flush=True)

    entries = []
    for line in git("diff-tree", "--no-commit-id", "-r",
                    base, head).split("\n"):
        if not line.strip():
            continue
        meta, path = line.split("\t")
        mode = meta.split()[1]
        status = meta.split()[0]
        if path.startswith(APP_OWNED_PREFIXES):
            continue  # app-owned; remote copy is authoritative
        if status == "D":
            entries.append({"path": path, "mode": mode, "type": "blob",
                            "sha": None})
            continue
        with open(f"{REPO}/{path}", "rb") as f:
            content = f.read()
        blob = api("/git/blobs", "POST",
                   {"content": base64.b64encode(content).decode(),
                    "encoding": "base64"})
        entries.append({"path": path, "mode": mode, "type": "blob",
                        "sha": blob["sha"]})
    if not entries:
        print("nothing new to push (only app-owned files changed)")
        return
    tree = api("/git/trees", "POST",
               {"base_tree": rtree, "tree": entries})["sha"]
    commit = api("/git/commits", "POST",
                 {"message": git("log", "-1", "--format=%B", head),
                  "tree": tree, "parents": [remote],
                  "author": {"name": "Muse", "email": "muse@local"}})["sha"]
    # final race check
    now = api("/git/ref/heads/main")["object"]["sha"]
    if now != remote:
        subprocess.run(["git", "reset", "--soft", f"{head}~1"], cwd=REPO,
                       check=True)
        sys.exit("ABORT: remote main moved during push; reset")
    api("/git/refs/heads/main", "PATCH", {"sha": commit})
    print(f"pushed {commit[:12]}")


if __name__ == "__main__":
    main()
