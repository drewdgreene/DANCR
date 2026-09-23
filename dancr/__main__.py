"""Entry point for `python -m dancr` and the packaged app: no arguments opens the window, otherwise the CLI."""
import sys

from dancr.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1].lower().endswith(".json")):
        sys.argv.insert(1, "gui")            # double-click, or a project file dropped on the app
    main()
