#!/usr/bin/env python3
"""Print a bug report message for unplanned failures.

Usage in bash:
    some_command || { python3 report_bug.py; exit 1; }

Usage in Python:
    try:
        ...
    except Exception:
        import report_bug
        report_bug.main()
        raise
"""

import sys


def main():
    """Print bug report instructions to stderr."""
    msg = """\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Something went wrong — this appears to be a bug.

  To report it, please open a GitHub issue at:
    https://github.com/ETHRoboticsClub/Hydra/issues/new

  Include:
    • The command you ran
    • The full error output above
    • Your OS and any relevant environment details
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"""
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
