import ast
from pathlib import Path
import unittest


FORBIDDEN_MODULES = (
    "django.http",
    "django.shortcuts",
    "django.views",
    "django.middleware",
    "rest_framework",
)
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


class ServiceBoundaryTests(unittest.TestCase):
    def test_services_do_not_import_framework_adapters_or_accept_requests(self):
        root = Path(__file__).parent
        violations = []
        for path in root.rglob("*.py"):
            if path.name.startswith("test_") or path.name.endswith("_test.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    modules = []
                for module in modules:
                    if any(module == forbidden or module.startswith(f"{forbidden}.") for forbidden in FORBIDDEN_MODULES):
                        violations.append(f"{path}: forbidden import {module}")
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                    if node.args.vararg:
                        arguments.append(node.args.vararg)
                    if node.args.kwarg:
                        arguments.append(node.args.kwarg)
                    if any(argument.arg == "request" for argument in arguments):
                        violations.append(f"{path}:{node.lineno}: forbidden parameter request")
                    if any(argument.arg == "method" for argument in arguments):
                        violations.append(f"{path}:{node.lineno}: forbidden parameter method")
                if isinstance(node, ast.Compare):
                    operands = [node.left, *node.comparators]
                    if any(
                        isinstance(candidate, ast.Constant) and candidate.value in HTTP_METHODS
                        for operand in operands
                        for candidate in ast.walk(operand)
                    ):
                        violations.append(f"{path}:{node.lineno}: HTTP method comparison in service")
        self.assertEqual(violations, [])
