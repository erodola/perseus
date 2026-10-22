"""Former name of `perseus`, kept so existing scripts and pickled module paths keep working.

`import perseus_core.x` returns the module object of `perseus.x` (one module, not a copy), with a
DeprecationWarning once. Remove when nothing imports the old name.
"""
import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_OLD, _NEW = "perseus_core", "perseus"


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD):])

    def exec_module(self, module):
        pass


warnings.warn("perseus_core is now perseus; import perseus instead", DeprecationWarning, stacklevel=2)
if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
sys.modules[_OLD] = importlib.import_module(_NEW)
