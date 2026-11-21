import importlib
import sys
import unittest
from pathlib import Path


def load_tests(loader, tests, pattern):
    if "django" in sys.modules:
        return unittest.TestSuite()
    for path in sorted(Path(__file__).parent.glob(pattern or "test*.py")):
        if path.is_file() and path.name != "__init__.py":
            module = importlib.import_module(f"{__name__}.{path.stem}")
            tests.addTests(loader.loadTestsFromModule(module))
    return tests