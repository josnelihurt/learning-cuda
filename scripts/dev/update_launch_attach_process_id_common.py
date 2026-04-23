#!/usr/bin/env python3
"""Shared helpers for launch.json processId updates."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable

ConfigMatcher = Callable[[dict], bool]
LineStateMatcher = Callable[[str, dict], None]
LineStateChecker = Callable[[dict], bool]


def parse_args(argv: list[str], usage: str) -> tuple[str, int] | None:
    if len(argv) != 3:
        print(usage, file=sys.stderr)
        return None

    path = argv[1]
    try:
        pid = int(argv[2])
    except ValueError:
        print("pid must be an integer", file=sys.stderr)
        return None

    return path, pid


def update_launch_process_id(
    path: str,
    pid: int,
    name_line_contains: str,
    config_matcher: ConfigMatcher,
    line_state_matcher: LineStateMatcher,
    line_state_checker: LineStateChecker,
    not_found_error: str,
    process_id_not_found_error: str,
    verification_failed_error: str,
    verification_missing_error: str,
) -> int:
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    parseable = _strip_jsonc_line_comments(content)
    try:
        data = json.loads(parseable)
    except json.JSONDecodeError as error:
        print(
            f"launch.json: invalid JSON (after stripping full-line // comments): {error}",
            file=sys.stderr,
        )
        return 1

    if not any(config_matcher(cfg) for cfg in data.get("configurations", [])):
        print(not_found_error, file=sys.stderr)
        return 1

    text, updated = _rewrite_process_id_line(
        content=content,
        pid=pid,
        name_line_contains=name_line_contains,
        line_state_matcher=line_state_matcher,
        line_state_checker=line_state_checker,
    )
    if not updated:
        print(process_id_not_found_error, file=sys.stderr)
        return 1

    parseable_out = _strip_jsonc_line_comments(text)
    try:
        data_out = json.loads(parseable_out)
    except json.JSONDecodeError as error:
        print(f"launch.json: wrote invalid JSON: {error}", file=sys.stderr)
        return 1

    for cfg in data_out.get("configurations", []):
        if config_matcher(cfg):
            if int(cfg.get("processId")) != pid:
                print(verification_failed_error, file=sys.stderr)
                return 1
            break
    else:
        print(verification_missing_error, file=sys.stderr)
        return 1

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as file:
        file.write(text)
    os.replace(tmp, path)
    return 0


def _strip_jsonc_line_comments(content: str) -> str:
    return "".join(
        line
        for line in content.splitlines(keepends=True)
        if not re.match(r"^\s*//", line)
    )


def _rewrite_process_id_line(
    content: str,
    pid: int,
    name_line_contains: str,
    line_state_matcher: LineStateMatcher,
    line_state_checker: LineStateChecker,
) -> tuple[str, bool]:
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    phase = 0
    state: dict = {}

    for line in lines:
        if re.match(r"^\s*//", line):
            out.append(line)
            continue

        if phase == 0:
            if '"name"' in line and name_line_contains in line:
                phase = 1
                state = {}
            out.append(line)
            continue

        if phase == 1:
            line_state_matcher(line, state)
            if '"processId"' in line and line_state_checker(state):
                match = re.match(r'^(\s*)"processId"\s*:', line)
                if match:
                    indent = match.group(1)
                    out.append(f'{indent}"processId": {pid},\n')
                    phase = 2
                else:
                    out.append(line)
            else:
                out.append(line)
            continue

        out.append(line)

    return "".join(out), phase == 2
