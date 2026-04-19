#!/usr/bin/env python3
"""Update processId in .vscode/launch.json for the Go attach configuration."""

import json
import os
import re
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: update_launch_go_attach_process_id.py <launch.json> <pid>",
            file=sys.stderr,
        )
        return 2

    path = sys.argv[1]
    try:
        pid = int(sys.argv[2])
    except ValueError:
        print("pid must be an integer", file=sys.stderr)
        return 2

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    parseable = "".join(
        line
        for line in content.splitlines(keepends=True)
        if not re.match(r"^\s*//", line)
    )
    try:
        data = json.loads(parseable)
    except json.JSONDecodeError as e:
        print(
            f"launch.json: invalid JSON (after stripping full-line // comments): {e}",
            file=sys.stderr,
        )
        return 1

    found = False
    for cfg in data.get("configurations", []):
        if cfg.get("type") == "go" and cfg.get("request") == "attach":
            found = True
            break
    if not found:
        print(
            "launch.json: no configuration with type=go and request=attach",
            file=sys.stderr,
        )
        return 1

    lines = content.splitlines(keepends=True)
    out: list[str] = []
    phase = 0
    seen_go = False
    seen_attach = False

    for line in lines:
        if re.match(r"^\s*//", line):
            out.append(line)
            continue
        if phase == 0:
            if '"name"' in line and '"Attach to Go Process"' in line:
                phase = 1
                seen_go = False
                seen_attach = False
            out.append(line)
        elif phase == 1:
            if '"type"' in line and re.search(
                r":\s*\"go\"\s*,?\s*$", line.rstrip("\r\n")
            ):
                seen_go = True
            if '"request"' in line and re.search(
                r":\s*\"attach\"\s*,?\s*$", line.rstrip("\r\n")
            ):
                seen_attach = True
            if '"processId"' in line:
                if seen_go and seen_attach:
                    m = re.match(r"^(\s*)\"processId\"\s*:", line)
                    if not m:
                        print(
                            "launch.json: could not parse processId line indentation",
                            file=sys.stderr,
                        )
                        return 1
                    indent = m.group(1)
                    out.append(f'{indent}"processId": {pid},\n')
                    phase = 2
                else:
                    out.append(line)
            else:
                out.append(line)
        else:
            out.append(line)

    if phase != 2:
        print(
            "launch.json: could not find processId after name Attach to Go Process "
            "(expected key order name, type, request, ..., processId)",
            file=sys.stderr,
        )
        return 1

    text = "".join(out)
    parseable_out = "".join(
        line
        for line in text.splitlines(keepends=True)
        if not re.match(r"^\s*//", line)
    )
    try:
        data_out = json.loads(parseable_out)
    except json.JSONDecodeError as e:
        print(f"launch.json: wrote invalid JSON: {e}", file=sys.stderr)
        return 1

    for cfg in data_out.get("configurations", []):
        if cfg.get("type") == "go" and cfg.get("request") == "attach":
            if int(cfg.get("processId")) != pid:
                print(
                    "launch.json: verification failed for go attach processId",
                    file=sys.stderr,
                )
                return 1
            break
    else:
        print(
            "launch.json: verification missing go attach after update",
            file=sys.stderr,
        )
        return 1

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
