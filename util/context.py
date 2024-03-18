import sys
from contextlib import contextmanager

@contextmanager
def AddPath(p):
    old_path = sys.path[:]
    sys.path.append(p)
    try:
        yield
    finally:
        sys.path = old_path

