#!/usr/bin/env python3
"""Make robomimic's diffusion policy import optional.

The current OpenX environment can use a JAX version that is incompatible with
the old diffusers package pinned by robomimic. Transformer-GMM BC does not use
diffusion policy, but robomimic imports it at package import time. This patch
keeps non-diffusion algorithms usable while preserving diffusion imports when
the dependency stack supports them.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "robomimic" / "robomimic" / "algo" / "__init__.py"

OLD = "from robomimic.algo.diffusion_policy import DiffusionPolicyUNet\n"
NEW = """try:
    from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
except Exception as exc:
    DiffusionPolicyUNet = None
    _DIFFUSION_POLICY_IMPORT_ERROR = exc
"""


def main():
    text = TARGET.read_text()
    if NEW in text:
        print(f"Already patched: {TARGET}")
        return
    if OLD not in text:
        raise RuntimeError(f"Expected import line not found in {TARGET}")
    TARGET.write_text(text.replace(OLD, NEW))
    print(f"Patched optional diffusion import in {TARGET}")


if __name__ == "__main__":
    main()
