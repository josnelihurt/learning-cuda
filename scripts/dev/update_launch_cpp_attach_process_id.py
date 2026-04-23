#!/usr/bin/env python3
"""Update processId in .vscode/launch.json for the C++ attach configuration."""

import sys

from update_launch_attach_process_id_common import parse_args, update_launch_process_id


def _line_state_matcher(_line: str, _state: dict) -> None:
    return


def _line_state_checker(_state: dict) -> bool:
    return True


def main() -> int:
    parsed = parse_args(
        sys.argv, "usage: update_launch_cpp_attach_process_id.py <launch.json> <pid>"
    )
    if parsed is None:
        return 2
    path, pid = parsed

    return update_launch_process_id(
        path=path,
        pid=pid,
        name_line_contains='"Attach to C++ Accelerator"',
        config_matcher=lambda cfg: cfg.get("name") == "Attach to C++ Accelerator",
        line_state_matcher=_line_state_matcher,
        line_state_checker=_line_state_checker,
        not_found_error='launch.json: no configuration with name "Attach to C++ Accelerator"',
        process_id_not_found_error=(
            "launch.json: could not find processId after name Attach to C++ Accelerator"
        ),
        verification_failed_error="launch.json: verification failed for C++ attach processId",
        verification_missing_error="launch.json: verification missing C++ attach after update",
    )


if __name__ == "__main__":
    raise SystemExit(main())
