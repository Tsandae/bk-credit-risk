"""Create sprint milestones, labels and issues from tickets.csv using the GitHub CLI.
Prerequisites: install GitHub CLI (https://cli.github.com), run `gh auth login`, then run this
script from inside your cloned repository. Create a GitHub Project (Board layout) in the web UI
and add the new issues to it (Projects -> Add item, or enable the auto-add workflow).
Not tested against your account: check the output and adjust if gh reports errors.
"""
import csv, subprocess

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("WARN:", " ".join(cmd[:4]), r.stderr.strip()[:200])
    return r

rows = list(csv.DictReader(open("project/tickets.csv", encoding="utf-8")))
for s in sorted({r["sprint"] for r in rows}):
    run(["gh", "api", "repos/{owner}/{repo}/milestones", "-f", f"title={s}"])
for l in sorted({r["label"] for r in rows}):
    run(["gh", "label", "create", l, "--force"])
for r in rows:
    body = f"Epic: {r['epic']}\n\nAcceptance criteria: {r['acceptance_criteria']}\n\nRepo path: {r['repo_path']}"
    run(["gh", "issue", "create", "--title", f"{r['id']} {r['title']}", "--body", body, "--label", r["label"], "--milestone", r["sprint"]])
print("Done. Now add the issues to your GitHub Project board and take the screenshot.")
