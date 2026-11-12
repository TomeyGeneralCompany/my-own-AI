"""Durable JSON checkpoints and a Linux advisory lease for a memory directory."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
MAX_CHECKPOINT = 8 * 1024 * 1024
HEADER = {"format", "version", "module", "state", "sha256"}


class MemoryCorrupt(ValueError):
    pass


def encode_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False).encode("ascii")


def json_copy(value):
    return json.loads(encode_json(value))


class CheckpointStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.recovered = set()

    @contextmanager
    def lease(self):
        with (self.directory / ".controller.lock").open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This memory directory already has a controller") from exc
            yield

    def path(self, name):
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
            raise ValueError("Invalid module identifier")
        return self.directory / (name + ".json")

    def _decode(self, raw, name, version, validate, *, legacy=False):
        try:
            if len(raw) > MAX_CHECKPOINT:
                raise ValueError("Checkpoint exceeds size limit")
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("Expected an object")
            if "format" in record:
                checksum = record.pop("sha256", None)
                if (set(record) != HEADER - {"sha256"}
                        or type(record["format"]) is not int or record["format"] != 1
                        or record["module"] != name or record["version"] != version
                        or checksum != hashlib.sha256(encode_json(record)).hexdigest()):
                    raise ValueError("Invalid envelope, version or checksum")
                state = record["state"]
            elif legacy and not ({"module", "sha256", "version"} & record.keys()):
                state = record
            else:
                raise ValueError("Missing checkpoint envelope")
            if not isinstance(state, dict):
                raise ValueError("Expected checkpoint state object")
            encode_json(state)
            if validate:
                validate(state)
            return state
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
            raise MemoryCorrupt(f"Invalid checkpoint for {name}: {exc}") from exc

    def _read_raw(self, path):
        with path.open("rb") as handle:
            return handle.read(MAX_CHECKPOINT + 1)

    def read(self, name, version="1.0.0", validate=None, *, legacy=True):
        path = self.path(name)
        backup = path.with_suffix(".prev")
        errors = []
        for candidate in (path, backup):
            try:
                raw = self._read_raw(candidate)
                state = self._decode(raw, name, version, validate, legacy=legacy and candidate == path
                                     and not backup.exists())
            except FileNotFoundError:
                continue
            except MemoryCorrupt as exc:
                errors.append(exc)
                continue
            if candidate == backup:
                self.recovered.add(name)
            return state
        if errors:
            raise errors[0]
        return {}

    def _replace(self, path, payload, verify):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".save-", delete=False) as handle:
                tmp = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            verify(self._read_raw(tmp))
            os.replace(tmp, path)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def write(self, name, state, version="1.0.0", validate=None):
        path = self.path(name)
        backup = path.with_suffix(".prev")
        record = {"format": 1, "version": version, "module": name, "state": state}
        record["sha256"] = hashlib.sha256(encode_json(record)).hexdigest()

        def verify(raw):
            return self._decode(raw, name, version, validate)

        payload = encode_json(record)
        verify(payload)
        try:
            previous = self._read_raw(path)
            previous_state = self._decode(previous, name, version, validate, legacy=not backup.exists())
        except (FileNotFoundError, MemoryCorrupt):
            pass
        else:
            if previous_state == state:
                # Rewrite legacy records even when the payload is unchanged.
                try:
                    verify(previous)
                except MemoryCorrupt:
                    pass
                else:
                    return
            previous_record = {"format": 1, "version": version, "module": name, "state": previous_state}
            previous_record["sha256"] = hashlib.sha256(encode_json(previous_record)).hexdigest()
            self._replace(backup, encode_json(previous_record), verify)
        self._replace(path, payload, verify)
