import argparse
import collections
import json
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="origin/main")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
def git(*parts):
    return subprocess.check_output(["git", "-C", str(root), *parts]).decode().split("\0")
base = subprocess.check_output(["git", "-C", str(root), "merge-base", "HEAD", args.base]).decode().strip()
files = set(git("diff", "--name-only", "-z", base)) | set(git("ls-files", "--others", "--exclude-standard", "-z"))
files = sorted(f for f in files if f.endswith(".swift") and (root / f).is_file())
if not files:
    print("No changed Swift files")
    sys.exit(0)
allowed = json.loads(Path(__file__).with_name("legacy-identifiers.json").read_text())
result = subprocess.run(["xcrun", "swift-format", "lint", "--strict", "--configuration", ".swift-format", *files], cwd=root, text=True, capture_output=True)
counts = collections.Counter()
failures = []
for line in (result.stdout + result.stderr).splitlines():
    match = re.fullmatch(r"([^:]+):\d+:\d+: (?:error|warning): \[AlwaysUseLowerCamelCase\] (.*)", line)
    if match and match[1] + "\t" + match[2] in allowed:
        counts[match[1] + "\t" + match[2]] += 1
    else:
        failures.append(line)
for key, count in counts.items():
    if count > allowed[key]:
        failures.append("Legacy identifier count increased: " + key)
if result.returncode and not counts and not failures:
    failures.append("swift-format failed without diagnostics")
if failures:
    print("\n".join(failures))
    sys.exit(1)
print(f"Checked {len(files)} Swift files; retained {sum(counts.values())} existing mathematical/fixture identifiers")
