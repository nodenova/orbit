"""Keeps `tests/` itself importable now that the tests live in subdirectories.

`tests/fake_mlx.py` is shared infrastructure rather than a test, so it stays at the
root of this tree while its importers (`backends/test_mlx_tier0.py`,
`gateway/test_constrain_mlx.py`) sit one level down. pytest puts the *test file's*
directory on `sys.path`, not this one, so the bare `import fake_mlx` those modules
use would stop resolving without something here — and a `conftest.py` is exactly the
hook pytest inserts this directory for.

Deliberately not solved by adding `__init__.py` files: `tests/` must not become a
package, or the MLX stand-in ships inside the distribution (see the `fake_mlx`
override in `pyproject.toml`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
