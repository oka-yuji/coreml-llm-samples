#!/usr/bin/env python3
"""Grade the frozen A-P7b task set from the raw E2E logs.

Conservative by construction: anything the log does not positively show is a FAIL.
Usage: python3 grade.py <data-dir>   ->  grades.json + a markdown table on stdout
"""
import json, os, re, sys, glob

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
TASKS = {t["id"]: t for t in json.load(open(os.path.join(DATA, "tasks.json")))["tasks"]}
URL_RE = re.compile(r"https?://[^\s\)\]\>」』、,。]+")

# Task-specific content checks. Frozen with tasks.json; keep in lockstep with its "rubric" field.
CONTENT = {
    "S1": re.compile(r"core ?ml", re.I),
    "S2": re.compile(r"neural engine|ニューラルエンジン|ANE", re.I),
    "S3": re.compile(r"MLX"),
    "S4": re.compile(r"async|await|非同期", re.I),
    "U1": re.compile(r"core ?ml", re.I),
    "U2": re.compile(r"create ?ml", re.I),
    "U3": re.compile(r"apple ?silicon|アップルシリコン", re.I),
    "N1": re.compile(r"晴|曇|くもり|雨|雪|℃|度"),
    "N2": re.compile(r"M4"),
    "N3": re.compile(r"neural engine|ANE", re.I),
    "D1": re.compile(r"晴|曇|くもり|雨|雪|℃|度"),
    "D2": None,  # graded by the >=2 URL rule below
}
FETCH_HOST = {"U1": "developer.apple.com/documentation/coreml",
              "U2": "developer.apple.com/documentation/createml",
              "U3": "ja.wikipedia.org"}
NOTE_TASKS = {"N1", "N2", "N3"}


def blocks(text):
    """The driver writes '--- <kind> <title>' then the entry body until the next marker."""
    out, kind, title, buf = [], None, None, []
    for line in text.split("\n"):
        if line.startswith("--- ") or line.startswith("[e2e] ") or line.startswith("AGENT E2E "):
            if kind:
                out.append((kind, title, "\n".join(buf).strip()))
            kind, title, buf = None, None, []
            if line.startswith("--- "):
                parts = line[4:].split(" ", 1)
                kind, title = parts[0], (parts[1] if len(parts) > 1 else "")
        elif kind:
            buf.append(line)
    if kind:
        out.append((kind, title, "\n".join(buf).strip()))
    return out


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    r = {"log": os.path.basename(path), "raw_len": len(text)}
    r["completed"] = "AGENT E2E COMPLETED" in text
    r["diverges"] = text.count("[agent] re-render diverges")
    r["rewind"] = text.count("cannot rewind")
    r["deadline"] = "[e2e] deadline of" in text
    r["steplimit"] = "Step limit reached" in text
    m = re.search(r"^\[e2e\] loaded .* in ([\d.]+)s", text, re.M)
    r["load_s"] = float(m.group(1)) if m else None
    m = re.search(r"^\[e2e\] ([\d.]+)s  toolCalls=(\d+)  entries=(\d+)  failures=(\d+)", text, re.M)
    if m:
        r["elapsed_s"], r["tool_calls"], r["failures"] = float(m.group(1)), int(m.group(2)), int(m.group(4))
    else:
        r["elapsed_s"], r["tool_calls"], r["failures"] = None, None, None
    m = re.search(r"^\[e2e\] status: step (\d+)\s+\|(.*)$", text, re.M)
    r["steps"] = int(m.group(1)) if m else None
    r["finish"] = m.group(2).split("|")[-1].strip() if m else None
    # 2026-09-10: the driver prints '[e2e] outcome: <word>' between the answer and the verdict line,
    # so stop the answer there; the old form (answer straight to AGENT E2E) is still accepted.
    m = re.search(r"^\[e2e\] final answer \((\d+) chars\):\n(.*?)\n(?:\[e2e\] outcome: |AGENT E2E )",
                  text, re.M | re.S)
    r["answer"] = m.group(2) if m else ""
    r["answer_chars"] = int(m.group(1)) if m else 0
    m = re.search(r"^\[e2e\] outcome: (\S+)", text, re.M)
    r["outcome"] = m.group(1) if m else None
    r["urls"] = URL_RE.findall(r["answer"])
    bl = blocks(text)
    r["tool_call_names"] = [t for k, t, _ in bl if k == "toolCall"]
    r["fetch_urls"] = [b.split("url:", 1)[1].strip() for k, t, b in bl
                       if k == "toolCall" and t == "fetch_page" and "url:" in b]
    r["notes"] = []
    for k, t, b in bl:
        if k == "toolCall" and t == "write_note" and "content:" in b:
            r["notes"].append(b.split("content:", 1)[1].strip())
    r["note_paths"] = [b for k, t, b in bl if k == "toolResult" and t == "write_note result"]
    r["tool_errors"] = [b[:120] for k, t, b in bl if k == "toolResult" and b.startswith("error:")]
    return r


def grade(task_id, r, note_files):
    fails = []
    if not r["completed"]:
        fails.append("incomplete")
    if r["diverges"]:
        fails.append("re-render-diverges")
    if r["rewind"]:
        fails.append("cannot-rewind")
    if r["deadline"]:
        fails.append("deadline")
    if r["steplimit"]:
        fails.append("step-limit")
    if r["finish"] == "cap":
        fails.append("cap")
    if not r["urls"]:
        fails.append("no-source-url")
    rx = CONTENT.get(task_id)
    if rx is not None and not rx.search(r["answer"]):
        fails.append("content-regex")
    if task_id == "D2" and len(r["urls"]) < 2:
        fails.append("fewer-than-2-urls")
    if task_id in FETCH_HOST and not any(FETCH_HOST[task_id] in u for u in r["fetch_urls"]):
        fails.append("no-fetch-of-target-url")
    if task_id in NOTE_TASKS:
        under = [p for p in r["note_paths"] if "agent-notes/" in p and p.endswith(".md")]
        if not under:
            fails.append("no-note-written")
        bodies = r["notes"] + [open(f, encoding="utf-8", errors="replace").read() for f in note_files]
        if not bodies or max((len(b.encode()) for b in bodies), default=0) < 200:
            fails.append("note-under-200-bytes")
        if any("</think>" in b for b in bodies):
            fails.append("note-has-thinking")
    return fails


# Only runs the driver finished (one ledger row each) are graded; a log still being written
# would otherwise be scored as an incomplete run.
DONE = set()
for led in glob.glob(os.path.join(DATA, "round_*.tsv")):
    for line in open(led, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) == 6:
            DONE.add(f"{f[0]}_{f[1]}_r{f[2]}")

rows = []
for path in sorted(glob.glob(os.path.join(DATA, "runs", "*.log"))):
    name = os.path.basename(path)[:-4]
    if name not in DONE:
        continue
    task_id, cond, rnd = name.split("_")
    r = parse(path)
    note_files = sorted(glob.glob(os.path.join(DATA, "runs", name + ".note*.md")))
    fails = grade(task_id, r, note_files)
    rows.append({"id": task_id, "cond": cond, "round": int(rnd[1:]), "pass": not fails,
                 "fails": fails, "judge": "auto", **r})

json.dump(rows, open(os.path.join(DATA, "grades.json"), "w"), ensure_ascii=False, indent=1)

by = {}
for r in rows:
    by[(r["id"], r["cond"], r["round"])] = r
ids = list(TASKS)
print(f"| task | low r1 | low r2 | low r3 | pass^3 | off r1 | lowfo r1 |")
print("|---|---|---|---|---|---|---|")
for i in ids:
    c = []
    for rnd in (1, 2, 3):
        r = by.get((i, "low", rnd))
        c.append("PASS" if r and r["pass"] else ("**FAIL**" if r else "—"))
    got = [by.get((i, "low", n)) for n in (1, 2, 3)]
    p3 = "PASS" if all(g and g["pass"] for g in got) else ("**FAIL**" if all(got) else "—")
    def mark(r):
        return "PASS" if r and r["pass"] else ("**FAIL**" if r else "—")
    print(f"| {i} | {c[0]} | {c[1]} | {c[2]} | {p3} | "
          + mark(by.get((i, "off", 1))) + " | " + mark(by.get((i, "lowfo", 1))) + " |")


def rate(sel):
    s = [r for r in rows if sel(r)]
    return f"{sum(r['pass'] for r in s)}/{len(s)}" if s else "—"


print()
print("pass@1 low (r1) :", rate(lambda r: r["cond"] == "low" and r["round"] == 1))
print("pass@1 off      :", rate(lambda r: r["cond"] == "off"))
n3 = [i for i in ids if all(by.get((i, "low", n)) for n in (1, 2, 3))]
print("pass^3 low      :", f"{sum(all(by[(i,'low',n)]['pass'] for n in (1,2,3)) for i in n3)}/{len(n3)}")
print("all low runs    :", rate(lambda r: r["cond"] == "low"))
print("all 48 (low+off):", rate(lambda r: r["cond"] in ("low", "off")))
print("pass@1 lowfo    :", rate(lambda r: r["cond"] == "lowfo"))
print("total runs      :", len(rows))
oc = {}
for r in rows:
    oc[r.get("outcome")] = oc.get(r.get("outcome"), 0) + 1
print("outcomes        :", dict(sorted(oc.items(), key=lambda kv: -kv[1])))
fc = {}
for r in rows:
    for f in r["fails"]:
        fc[f] = fc.get(f, 0) + 1
print("failure tags    :", dict(sorted(fc.items(), key=lambda kv: -kv[1])))
