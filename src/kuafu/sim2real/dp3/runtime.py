"""Locate optional upstream packages without changing their checkouts."""

import os
from pathlib import Path
import sys


def configure(kuafu2real_root=None, dp3_root=None):
    import importlib.util

    root = kuafu2real_root or os.environ.get("KUAFU2REAL_ROOT")
    if root:
        root = Path(root).expanduser().resolve()
        if not (root / "kuafu2real" / "dataset.py").is_file():
            raise FileNotFoundError(f"Kuafu2Real package not found: {root}")
        sys.path.insert(0, str(root))
    elif importlib.util.find_spec("kuafu2real") is None:
        raise ImportError("Install Kuafu2Real or set KUAFU2REAL_ROOT to its checkout")
    import kuafu2real
    root = Path(kuafu2real.__file__).resolve().parents[1]
    from kuafu2real.cli_utils import resolve_existing_dp3_root
    requested = dp3_root or os.environ.get("KUAFU_DP3_ROOT")
    dp3 = Path(requested).expanduser().resolve() if requested else resolve_existing_dp3_root()
    if (dp3 / "3D-Diffusion-Policy" / "diffusion_policy_3d").is_dir():
        dp3 = dp3 / "3D-Diffusion-Policy"
    if not (dp3 / "diffusion_policy_3d" / "policy" / "dp3.py").is_file():
        raise FileNotFoundError(f"DP3 package not found: {dp3}")
    compat = root / "support" / "dp3_compat"
    sys.path[:0] = [str(dp3), str(compat)]
    return {"kuafu2real": root, "dp3": dp3}
