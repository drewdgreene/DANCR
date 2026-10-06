"""Constants and the logger shared by Document and its mixins."""
from __future__ import annotations

import logging

log = logging.getLogger("dancr.ui")
AUTOSAVE_SECS = 60
UNDO_LIMIT = 200
POLL_MS = 2500              # how often states are re-read for changes made elsewhere (a CLI run, a rewritten source)
AUTO_RUN_DELAY_MS = 700     # quiet time after an edit before an automatic run
SOURCE_SETTLE_MS = 1500     # a data file being written fires many change events: wait for it to settle


from ..core import PipelineError


class ChangedOnDisk(PipelineError):
    """Saving would overwrite changes another program made to the project file."""
