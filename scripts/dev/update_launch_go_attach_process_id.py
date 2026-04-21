#!/usr/bin/env python3
"""Update processId in .vscode/launch.json for the Go attach configuration."""

import re
import sys

from update_launch_attach_process_id_common import parse_args, update_launch_process_id


def _line_state_matcher(line: str, state: dict) -> None:
    if '"type"' in line and re.search(r':\s*"go"\s*,?\s*$', line.rstrip("\r\n")):
        state["seen_go"] = True
    if '"request"' in line and re.search(r':\s*"attach"\s*,?\s*$', line.rstrip("\r\n")):
        state["seen_attach"] = True


def _line_state_checker(state: dict) -> bool:
    return state.get("seen_go", False) and state.get("seen_attach", False)


def main() -> int:
    parsed = parse_args(
        sys.argv, "usage: update_launch_go_attach_process_id.py <launch.json> <pid>"
    )
    if parsed is None:
        return 2
    path, pid = parsed

    return update_launch_process_id(
        path=path,
        pid=pid,
        name_line_contains='"Attach to Go Process"',
        config_matcher=lambda cfg: cfg.get("type") == "go"
        and cfg.get("request") == "attach",
        line_state_matcher=_line_state_matcher,
        line_state_checker=_line_state_checker,
        not_found_error="launch.json: no configuration with type=go and request=attach",
        process_id_not_found_error=(
            "launch.json: could not find processId after name Attach to Go Process "
            "(expected key order name, type, request, ..., processId)"
        ),
        verification_failed_error="launch.json: verification failed for go attach processId",
        verification_missing_error="launch.json: verification missing go attach after update",
    )


if __name__ == "__main__":
    raise SystemExit(main())
