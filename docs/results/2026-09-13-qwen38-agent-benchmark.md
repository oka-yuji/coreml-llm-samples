# Qwen3.8 27B agent benchmark (2026-09-13)

The number on the model card — 48 of 48 runs — comes from this measurement. This page records what
was run, how it was graded, and what the machine looked like, so the number can be reproduced or
disputed.

Machine: Apple M4 Max (Mac16,6), 128 GB unified memory, macOS 26.6.2. Runtime: this repository at
tag `qwen38-27b-agent-v1` (the runtime code is what the campaign ran; the commits between the
campaign and the tag touch only the catalog entry, the documentation and the GUI test driver),
`DemoApp` Release build for `arm64`, model bundle `okayuji/Qwen3.8-27B-coreml-agent` (int8, context
16,384). One process at a time; a memory watchdog and a quiet gate (memory pressure normal for 60 s)
ran between rounds and never fired.

## Tasks

Twelve tasks, frozen by SHA-256 in [`tasks.json`](2026-09-13-qwen38-agent-benchmark/tasks.json):

| id | group | task |
|---|---|---|
| S1 | search-summary | Core ML の最新情報(2026 年)を検索して 3 行で要約して |
| S2 | search-summary | Apple の Neural Engine について調べて、3 行で要約して |
| S3 | search-summary | Search for what Apple's MLX framework is and summarise it in three lines. |
| S4 | search-summary | Swift の async/await とは何かを調べて、3 行で要約して |
| U1 | url-reading | https://developer.apple.com/documentation/coreml を読んで要点を 3 つ挙げて |
| U2 | url-reading | Read https://developer.apple.com/documentation/createml and list three key points. |
| U3 | url-reading | https://ja.wikipedia.org/wiki/Apple_Silicon を読んで要点を 3 つ挙げて |
| N1 | note | 東京の明日の天気を調べて、要約をメモに保存して |
| N2 | note | Apple の M4 チップについて調べて、要点をメモに保存して |
| N3 | note | Look up what Apple's Neural Engine is and save a short summary as a note. |
| D1 | date-dependent | 大阪の今日の天気を教えて |
| D2 | date-dependent | What are this week's top AI news stories? List three of them with their source URLs. |

## Conditions

Each task ran through the headless driver (`DemoApp --agent-e2e --model <bundle> --task "<text>"
--effort <low|off> --deadline 900`) with the app defaults: 8 steps, 1,536 tokens per step, page
budget 10,000 characters, DuckDuckGo search, file operations off. Rounds: `low` three times and
`off` once, 48 runs in total; the notes folder was emptied before the campaign. A separate round
with file operations enabled (`--file-ops`, `low`, 12 runs) was measured on 2026-09-13 with the
runtime as it was for campaign D; it passed 12 of 12 and the agent never called `move_note`, which
none of the tasks asks for.

## Grading

[`grade.py`](2026-09-13-qwen38-agent-benchmark/grade.py) reads each run's log and fails a run on any
of: the driver did not report `AGENT E2E COMPLETED`; the final answer is empty or misses the content
the task asks for (a regular expression per task); a task that asked for search or reading has no
`Sources:` section with a URL; a note task wrote no note; a tool returned an error; the runtime had
to re-prefill the whole conversation (`cannot rewind` in the log). The script is the same one used
for the earlier campaigns; the only change since the first campaign is parsing of the driver's
`[e2e] outcome:` line.

## Result

| Condition | Runs | Passed |
|---|---|---|
| `low`, round 1 | 12 | 12 |
| `low`, round 2 | 12 | 12 |
| `low`, round 3 | 12 | 12 |
| `off`, round 1 | 12 | 12 |
| All | 48 | 48 |

Every task passed all three `low` rounds, so the pass^3 figure is 12 of 12. All 48 runs ended with
`outcome: answered`; no run hit the step limit or the token cap.

Per generation (148 generations): decode median 16.65 tok/s (12.4–19.7); time to first token median
6.88 s (0.68–46.4; the maximum is the first prefill of a session); peak footprint median 55,662 MB
(max 55,716 MB); speculation acceptance median 0.86; every generation ended at EOS. KV reuse: the
first step of every run prefilled from scratch (`reuse=none`) and every later step continued by
text prefix (`reuse=text`, 100 of 100); the log line `cache reuse unavailable` did not occur. Bundle
load median 47.4 s (46.6–49.9 s). Files: [`summary-tables.md`](2026-09-13-qwen38-agent-benchmark/summary-tables.md),
[`metrics-summary.txt`](2026-09-13-qwen38-agent-benchmark/metrics-summary.txt),
[`grades-table.md`](2026-09-13-qwen38-agent-benchmark/grades-table.md),
[`frozen-sha256.txt`](2026-09-13-qwen38-agent-benchmark/frozen-sha256.txt).

## History

The same tasks were measured four times in September 2026 while the runtime changed
(`summary-tables.md` has all four columns):

| Campaign | All 48 | pass^3 | What changed before it |
|---|---|---|---|
| C, 2026-09-02 | 47/48 | 12/12 | page-level chunk selection, parallel tools, citation check |
| D, 2026-09-13 | 45/48 | 10/12 | source-date instruction, model identity, `move_note` (off) |
| E, 2026-09-13 | 43/48 | 10/12 | text-prefix KV reuse (not yet armed after speculative EOS), numbered note names |
| F, 2026-09-13 | 48/48 | 12/12 | text-prefix KV reuse armed after every generation |

The failures in C, D and E were, with two exceptions, runs whose answer was correct but whose log
contained `cannot rewind`: the hybrid model's recurrent state cannot be rolled back, so a
re-tokenisation drift of the conversation forced a full re-prefill. Reusing the cache by text prefix
removed the cause; F has no such run. The two exceptions in E were a `low` run that spent its token
budget on thinking and an `off` run that hit the step limit.

## What this benchmark does not show

Twelve tasks written by the author on one machine. It checks that the agent completes typical
search, read and note tasks with real sources; it does not measure factual accuracy beyond the
per-task content check, and it does not compare against other runtimes or models.
