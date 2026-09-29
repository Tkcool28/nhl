"""Commit local changes and push to GitHub via the git-database API.

The workspace has no git credential helper, so pushes go through blobs ->
tree -> commit -> ref update (see AGENTS.md). Aborts if the remote ref
moved between read and write (frontend market-log syncs could race us).

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


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else "update"
    paths = sys.argv[2:] or ["-A"]
    git("add", *paths)
    if not git("status", "--porcelain"):
        print("nothing to push")
        return
    git("commit", "-q", "-m", msg)
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD~1")

    remote = api("/git/ref/heads/main")["object"]["sha"]
    rtree = api(f"/git/commits/{remote}")["tree"]["sha"]
    if rtree != git("rev-parse", f"{parent}^{{tree}}"):
        # remote moved and trees differ: do not clobber
        subprocess.run(["git", "reset", "--soft", parent], cwd=REPO,
                       check=True)
        sys.exit("ABORT: remote main moved; reset to pre-commit state")

    entries = []
    for line in git("diff-tree", "--no-commit-id", "-r", parent, head).split("\n"):
        if not line.strip():
            continue
        meta, path = line.split("\t")
        mode = meta.split()[1]
        with open(f"{REPO}/{path}", "rb") as f:
            content = f.read()
        blob = api("/git/blobs", "POST",
                   {"content": base64.b64encode(content).decode(),
                    "encoding": "base64"})
        entries.append({"path": path, "mode": mode, "type": "blob",
                        "sha": blob["sha"]})
    tree = api("/git/trees", "POST",
               {"base_tree": rtree, "tree": entries})["sha"]
    commit = api("/git/commits", "POST",
                 {"message": git("log", "-1", "--format=%B", head),
                  "tree": tree, "parents": [remote],
                  "author": {"name": "Muse", "email": "muse@local"}})["sha"]
    # final race check
    now = api("/git/ref/heads/main")["object"]["sha"]
    if now != remote:
        subprocess.run(["git", "reset", "--soft", parent], cwd=REPO,
                       check=True)
        sys.exit("ABORT: remote main moved during push; reset")
    api("/git/refs/heads/main", "PATCH", {"sha": commit})
    print(f"pushed {commit[:12]}")


if __name__ == "__main__":
    main()
