"""Private, crash-bounded controller quiescence for role repair commands."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import psutil

REPAIR_PAUSE_PROTOCOL = "senpai-controller-repair-pause/v1"
_LEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


class RepairPauseError(RuntimeError):
    """A private controller pause could not be established or released."""


@dataclass(frozen=True, slots=True)
class RepairPause:
    protocol: str
    lease_id: str
    expires_at: float
    requester_pid: int
    requester_start_time: float

    def __post_init__(self) -> None:
        if self.protocol != REPAIR_PAUSE_PROTOCOL:
            raise ValueError("unsupported repair-pause protocol")
        if not _LEASE_ID.fullmatch(self.lease_id):
            raise ValueError("invalid repair-pause lease id")
        if (
            not self.expires_at > 0
            or self.requester_pid <= 0
            or self.requester_start_time <= 0
        ):
            raise ValueError("invalid repair-pause expiry")


@dataclass(frozen=True, slots=True)
class RepairPauseAcknowledgement:
    protocol: str
    lease_id: str
    expires_at: float
    acknowledged_at: float
    supervisor_pid: int

    def __post_init__(self) -> None:
        if self.protocol != REPAIR_PAUSE_PROTOCOL:
            raise ValueError("unsupported repair-pause protocol")
        if not _LEASE_ID.fullmatch(self.lease_id):
            raise ValueError("invalid repair-pause lease id")
        if (
            not self.expires_at > 0
            or not self.acknowledged_at > 0
            or self.supervisor_pid <= 0
        ):
            raise ValueError("invalid repair-pause acknowledgement")


class RepairPauseStore:
    """Exchange one pause lease through a main-container-only directory."""

    def __init__(self, control_dir: Path):
        self.control_dir = control_dir.resolve()
        self.request_path = self.control_dir / "repair-pause.json"
        self.ack_path = self.control_dir / "repair-pause-ack.json"
        self.lock_path = self.control_dir / "repair-pause.lock"

    def request(self, lease_id: str, *, duration_seconds: float) -> RepairPause:
        if duration_seconds <= 0:
            raise ValueError("repair-pause duration must be positive")
        pause = RepairPause(
            protocol=REPAIR_PAUSE_PROTOCOL,
            lease_id=lease_id,
            expires_at=time.time() + duration_seconds,
            requester_pid=os.getpid(),
            requester_start_time=psutil.Process().create_time(),
        )
        self._prepare_directory()
        with self._locked():
            self._discard_expired_unlocked()
            if self.request_path.exists():
                raise RepairPauseError("another role repair pause is already active")
            self.ack_path.unlink(missing_ok=True)
            self._write_atomic(self.request_path, asdict(pause))
        return pause

    def current(self) -> RepairPause | None:
        try:
            pause = self._read(self.request_path, RepairPause)
        except FileNotFoundError:
            return None
        if pause.expires_at <= time.time():
            self._remove_if_matches(pause.lease_id)
            return None
        return pause

    def acknowledge(
        self,
        pause: RepairPause,
        *,
        supervisor_pid: int,
    ) -> RepairPauseAcknowledgement:
        with self._locked():
            current = self._current_unlocked()
            if current != pause:
                raise RepairPauseError(
                    "repair-pause lease changed before acknowledgement"
                )
            acknowledgement = RepairPauseAcknowledgement(
                protocol=REPAIR_PAUSE_PROTOCOL,
                lease_id=pause.lease_id,
                expires_at=pause.expires_at,
                acknowledged_at=time.time(),
                supervisor_pid=supervisor_pid,
            )
            self._write_atomic(self.ack_path, asdict(acknowledgement))
        return acknowledgement

    def acknowledgement(self) -> RepairPauseAcknowledgement | None:
        pause = self.current()
        if pause is None:
            return None
        try:
            acknowledgement = self._read(
                self.ack_path,
                RepairPauseAcknowledgement,
            )
        except FileNotFoundError:
            return None
        if (
            acknowledgement.lease_id != pause.lease_id
            or acknowledgement.expires_at != pause.expires_at
            or acknowledgement.expires_at <= time.time()
        ):
            return None
        return acknowledgement

    def wait_for_acknowledgement(
        self,
        pause: RepairPause,
        *,
        timeout_seconds: float,
    ) -> RepairPauseAcknowledgement:
        if timeout_seconds <= 0:
            raise ValueError("repair-pause wait must be positive")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            current = self.current()
            if current is None or current.lease_id != pause.lease_id:
                raise RepairPauseError("repair-pause lease disappeared before acknowledgement")
            acknowledgement = self.acknowledgement()
            if acknowledgement is not None:
                return acknowledgement
            time.sleep(0.05)
        self._remove_if_matches(pause.lease_id)
        raise RepairPauseError("controller did not quiesce before the repair-pause deadline")

    def release(self, lease_id: str) -> None:
        self._prepare_directory()
        with self._locked():
            pause = self._current_unlocked()
            if pause is None:
                return
            if pause.lease_id != lease_id:
                raise RepairPauseError("refusing to release another repair-pause lease")
            self.request_path.unlink(missing_ok=True)
            self.ack_path.unlink(missing_ok=True)

    def is_acknowledged_by_live_supervisor(self) -> bool:
        acknowledgement = self.acknowledgement()
        if acknowledgement is None:
            return False
        try:
            os.kill(acknowledgement.supervisor_pid, 0)
        except (OSError, ValueError):
            return False
        return True

    def _remove_if_matches(self, lease_id: str) -> None:
        self._prepare_directory()
        with self._locked():
            try:
                pause = self._read(self.request_path, RepairPause)
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
                return
            if pause.lease_id != lease_id:
                return
            self.request_path.unlink(missing_ok=True)
            self.ack_path.unlink(missing_ok=True)

    def _discard_expired_unlocked(self) -> None:
        try:
            pause = self._read(self.request_path, RepairPause)
        except FileNotFoundError:
            return
        if pause.expires_at > time.time():
            return
        self.request_path.unlink(missing_ok=True)
        self.ack_path.unlink(missing_ok=True)

    def _current_unlocked(self) -> RepairPause | None:
        try:
            pause = self._read(self.request_path, RepairPause)
        except FileNotFoundError:
            return None
        if pause.expires_at <= time.time():
            self.request_path.unlink(missing_ok=True)
            self.ack_path.unlink(missing_ok=True)
            return None
        return pause

    def _prepare_directory(self) -> None:
        self.control_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = self.control_dir.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise RepairPauseError("repair-pause control directory must be private")

    @staticmethod
    def _read(path: Path, model):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise RepairPauseError("repair-pause state must be a private regular file")
        return model(**json.loads(path.read_text(encoding="utf-8")))

    def _write_atomic(self, path: Path, payload: dict[str, object]) -> None:
        self._prepare_directory()
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            os.write(descriptor, json.dumps(payload, sort_keys=True).encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
