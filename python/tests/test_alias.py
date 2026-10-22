"""`perseus_core` is the former name of the package: its modules are perseus's own module objects."""
import importlib
import warnings


def test_perseus_core_alias():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("perseus_core.plan.contract")
        pkg = importlib.import_module("perseus_core")
    assert old is importlib.import_module("perseus.plan.contract")
    assert pkg is importlib.import_module("perseus")
