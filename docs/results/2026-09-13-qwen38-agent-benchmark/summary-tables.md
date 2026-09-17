### C/D/E/F headline

| metric | C (2026-09-02) | D (2026-09-13) | E (2026-09-13, d2aa75f) | F (2026-09-13, 934aab1) | F - E | F - C |
|---|---|---|---|---|---|---|
| pass@1 low (r1) | 12/12 (100%) | 11/12 (92%) | 10/12 (83%) | 12/12 (100%) | +17 pt | +0 pt |
| pass@1 off | 11/12 (92%) | 12/12 (100%) | 11/12 (92%) | 12/12 (100%) | +8 pt | +8 pt |
| all low runs (36) | 36/36 (100%) | 33/36 (92%) | 32/36 (89%) | 36/36 (100%) | +11 pt | +0 pt |
| all runs (48) | 47/48 (98%) | 45/48 (94%) | 43/48 (90%) | 48/48 (100%) | +10 pt | +2 pt |
| pass^3 low (12 tasks) | 12/12 (100%) | 10/12 (83%) | 10/12 (83%) | 12/12 (100%) | +17 pt | +0 pt |
| pass@1 lowfo (--file-ops, 12) | (not run) | 12/12 (100%) | (not re-run) | (not re-run) | - | - |

### task x side (low r1/r2/r3 / pass^3 / off)

| id | group | C | D | E | F |
|---|---|---|---|---|---|
| S1 | search-summary | ooo O o | ooo O o | ooo O o | ooo O o |
| S2 | search-summary | ooo O o | ooo O o | ooo O o | ooo O o |
| S3 | search-summary | ooo O o | ooo O o | ooo O o | ooo O o |
| S4 | search-summary | ooo O o | ooo O o | ooo O o | ooo O o |
| U1 | url-reading | ooo O o | ooo O o | ooo O o | ooo O o |
| U2 | url-reading | ooo O o | ooo O o | ooo O o | ooo O o |
| U3 | url-reading | ooo O o | ooo O o | ooo O o | ooo O o |
| N1 | note | ooo O o | oxx X o | xxx X o | ooo O o |
| N2 | note | ooo O x | xoo X o | ooo O o | ooo O o |
| N3 | note | ooo O o | ooo O o | ooo O o | ooo O o |
| D1 | date-dependent | ooo O o | ooo O o | ooo O x | ooo O o |
| D2 | date-dependent | ooo O o | ooo O o | xoo X o | ooo O o |

(o=PASS x=FAIL, 4th char = pass^3, 5th = off)

### per group

| group | n | side | pass@1 low | pass^3 | off | all low (n=3x) |
|---|---|---|---|---|---|---|
| search-summary | 4 | C | 4/4 | 4/4 | 4/4 | 12/12 |
|  |  | D | 4/4 | 4/4 | 4/4 | 12/12 |
|  |  | E | 4/4 | 4/4 | 4/4 | 12/12 |
|  |  | F | 4/4 | 4/4 | 4/4 | 12/12 |
| url-reading | 3 | C | 3/3 | 3/3 | 3/3 | 9/9 |
|  |  | D | 3/3 | 3/3 | 3/3 | 9/9 |
|  |  | E | 3/3 | 3/3 | 3/3 | 9/9 |
|  |  | F | 3/3 | 3/3 | 3/3 | 9/9 |
| note | 3 | C | 3/3 | 3/3 | 2/3 | 9/9 |
|  |  | D | 2/3 | 1/3 | 3/3 | 6/9 |
|  |  | E | 2/3 | 2/3 | 3/3 | 6/9 |
|  |  | F | 3/3 | 3/3 | 3/3 | 9/9 |
| date-dependent | 2 | C | 2/2 | 2/2 | 2/2 | 6/6 |
|  |  | D | 2/2 | 2/2 | 2/2 | 6/6 |
|  |  | E | 1/2 | 1/2 | 1/2 | 5/6 |
|  |  | F | 2/2 | 2/2 | 2/2 | 6/6 |

### failure tags

| tag | C | D | E | F |
|---|---|---|---|---|
| cannot-rewind | 1 | 3 | 3 | 0 |
| incomplete | 0 | 0 | 2 | 0 |
| step-limit | 0 | 0 | 1 | 0 |
| no-source-url | 0 | 0 | 1 | 0 |
| content-regex | 0 | 0 | 1 | 0 |
| re-render-diverges | 0 | 0 | 1 | 0 |
| cap | 0 | 0 | 1 | 0 |
| fewer-than-2-urls | 0 | 0 | 1 | 0 |

- **C failing runs (1)**: N2_off_r1 (outcome=None; cannot-rewind)
- **D failing runs (3)**: N1_low_r2 (outcome=answered; cannot-rewind), N1_low_r3 (outcome=answered; cannot-rewind), N2_low_r1 (outcome=answered; cannot-rewind)
- **E failing runs (5)**: D1_off_r1 (outcome=step-limit; incomplete+step-limit+no-source-url+content-regex), D2_low_r1 (outcome=cap; incomplete+re-render-diverges+cap+fewer-than-2-urls), N1_low_r1 (outcome=answered; cannot-rewind), N1_low_r2 (outcome=answered; cannot-rewind), N1_low_r3 (outcome=answered; cannot-rewind)
- **F failing runs (0)**: none

### timing (seconds per run, from the raw logs)

| side | cond | runs | load median | elapsed median | elapsed mean | elapsed max | steps median | toolCalls median |
|---|---|---|---|---|---|---|---|---|
| C | low | 36 | 25.1 | 61.2 | 76.1 | 168.3 | 3 | 2 |
| C | off | 12 | 25.1 | 56.2 | 64.9 | 149.3 | 3 | 2 |
| D | low | 36 | 47.4 | 80.8 | 91.9 | 201.9 | 3 | 3 |
| D | off | 12 | 47.8 | 71.2 | 65.3 | 143.6 | 2 | 2 |
| D | lowfo | 12 | 47.8 | 84.1 | 93.5 | 247.6 | 3 | 3 |
| E | low | 36 | 47.4 | 71.5 | 86.7 | 204.3 | 3 | 3 |
| E | off | 12 | 47.7 | 64.7 | 63.2 | 129.4 | 3 | 2 |
| F | low | 36 | 47.4 | 85.8 | 87.6 | 184.5 | 3 | 3 |
| F | off | 12 | 47.5 | 65.5 | 67.0 | 151.5 | 3 | 2 |

### answers: Sources section and citations

- **C** (48 runs): Sources-style heading 48 (100%), >=1 URL 48 (100%), mean URLs 2.42
- **D** (60 runs): Sources-style heading 60 (100%), >=1 URL 60 (100%), mean URLs 2.25
- **E** (48 runs): Sources-style heading 46 (96%), >=1 URL 47 (98%), mean URLs 2.17
- **F** (48 runs): Sources-style heading 48 (100%), >=1 URL 48 (100%), mean URLs 2.40
