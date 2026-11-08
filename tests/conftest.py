"""Test setup: compile the optional multi-core kernel synchronously (so every test sees the same code path) and
leave the cores free for the rest of the machine."""
import os

os.environ.setdefault("LINGESTRA_SYNC_WARM", "1")
