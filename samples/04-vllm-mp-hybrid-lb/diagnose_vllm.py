"""Read the FIRST worker traceback from saved NFS logs; never starts GPU jobs."""
import argparse
from collections import deque
from pathlib import Path
import re

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ACTOR = re.compile(r"^\(([^)]*(?:Worker|RayWorker)[^)]*)\)")
EXCEPTION = re.compile(r"\b(?:[A-Za-z_]\w*(?:Error|Exception)|StopIteration|KeyboardInterrupt)(?::.*)?$")


def first_failure(path, max_lines=160):
    """Follow one worker through interleaved logs, keeping bounded context."""
    context = deque(maxlen=8)
    selected = None
    collected = []
    fallback = []
    fallback_left = 0
    with Path(path).open(errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            line = ANSI.sub("", raw.rstrip())
            # A line can contain a tqdm carriage-return before the actor prefix.
            line = line.rsplit("\r", 1)[-1]
            actor = ACTOR.match(line)
            label = actor.group(1) if actor else None
            if selected is None and label and "ERROR" in line:
                selected = label
                collected = list(context)
            if selected is not None and label == selected:
                collected.append((number, line))
                if EXCEPTION.search(line) or len(collected) >= max_lines:
                    return "first worker error", collected
            # Keep a fallback for unprefixed Python tracebacks/native errors.
            if not fallback and ("Traceback (most recent call last)" in line or "ERROR" in line):
                fallback = list(context)
                fallback_left = max_lines - len(fallback)
            if fallback_left:
                fallback.append((number, line))
                fallback_left -= 1
            context.append((number, line[:4000]))
    if collected:
        return "first worker error (exception terminator not found)", collected
    return "first error context (no worker traceback found)", fallback


def failure_report(run):
    run = Path(run)
    try:
        key = (run / "api-key").read_text().strip()
    except FileNotFoundError:
        key = ""
    sections = []
    for name in ("vllm.log", "vllm-worker.log"):
        path = run / name
        if not path.is_file():
            continue
        kind, lines = first_failure(path)
        rendered = "\n".join(f"{number}: {line}" for number, line in lines)
        sections.append(f"{name}: {kind}\n{rendered or '(No errors found)'}")
    report = "\n\n".join(sections) or "No vLLM logs found in this directory."
    return report.replace(key, "[API KEY REDACTED]") if key else report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error("Run directory does not exist")
    print(failure_report(args.run_dir))


if __name__ == "__main__":
    main()
