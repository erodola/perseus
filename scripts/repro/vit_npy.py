"""ViT runner on pre-normalized (224, 224, 3) .npy inputs (scripts/repro/data/vit_eurosat): weights.pixels reads the
array, so no PIL. Without scipy the float64 reference's erf (GELU, plaintext side only) comes from math.erf."""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
try:
    import scipy.special  # noqa: F401
except ImportError:
    import math
    import types
    _sp, _sps = types.ModuleType("scipy"), types.ModuleType("scipy.special")
    _sps.erf = np.vectorize(math.erf, otypes=[np.float64])
    _sp.special = _sps
    sys.modules["scipy"], sys.modules["scipy.special"] = _sp, _sps
import examples.vit_from_primitives.weights as V  # noqa: E402

V.pixels = lambda path: np.load(path)
from examples.vit_from_primitives.run_prefill import main  # noqa: E402

sys.exit(main(sys.argv[1:]))
