"""Linux worker entry point with a parent-death signal before exec."""
import ctypes
import os
import signal
import sys


def main():
    parent = int(sys.argv[1])
    ready_fd = int(sys.argv[2])
    argv = sys.argv[3:]
    try:
        os.set_inheritable(ready_fd, False)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if os.getppid() != parent:
            raise RuntimeError("Owner exited before worker initialization")
        os.execvp(argv[0], argv)
    except Exception as exc:
        os.write(ready_fd, str(exc).encode("utf-8", errors="replace")[:4000])
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
