"""Operating-system file locks, shared by the project-file locks and the persistent search index.

A lock is an exclusive operating-system lock on a small file (``fcntl`` on POSIX, ``msvcrt`` on Windows),
released by the OS however the process ends. It is deliberately self-contained — it imports nothing else in
DANCR — because the code fingerprint (``core/engine_hash.py``) follows the imports of every step module, and a
step that pulled in the answer engine through a lock helper would inflate what invalidates a cache.
"""
from __future__ import annotations

import errno
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_LOCK_WAIT = 30.0      # seconds to wait for another program to release a lock by default


class LockUnavailable(OSError):
    """The lock file's folder cannot be locked at all (some network drives), as opposed to a held lock."""


def try_lock(fd: int) -> bool:
    """True when the lock is ours, False when another holder has it. A folder that cannot lock at all raises
    :class:`LockUnavailable`."""
    busy = (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK))
    try:
        if sys.platform == "win32":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        # Windows msvcrt.locking reports a held lock as winerror 33 (ERROR_LOCK_VIOLATION), which some builds
        # surface without a matching errno; treat it as busy rather than an unlockable folder.
        if e.errno in busy or getattr(e, "winerror", None) == 33:
            return False
        raise LockUnavailable(str(e)) from e
    return True


def unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def lock_path(lock: Path | str, wait: float | None = None, message: str = "the file is being used") -> Iterator[None]:
    """Hold the raw operating-system lock on ``lock``, waiting up to ``wait`` seconds. Raises ``TimeoutError``
    with ``message`` when the wait is overrun, and :class:`LockUnavailable` when the folder cannot lock."""
    lock = Path(lock).expanduser()
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + (DEFAULT_LOCK_WAIT if wait is None else wait)
        while not try_lock(fd):
            if time.monotonic() >= deadline:
                raise TimeoutError(message)
            time.sleep(0.05)
        try:
            yield
        finally:
            unlock(fd)
    finally:
        os.close(fd)
