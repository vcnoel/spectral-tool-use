"""
Deterministic extraction.

Import this module BEFORE torch: cuBLAS reads CUBLAS_WORKSPACE_CONFIG when the
CUDA context is created, and without it `torch.use_deterministic_algorithms`
cannot select deterministic matmul kernels. Two extractions of one model and
benchmark differed by 2-3 labels in ~850 before this was set (audit
2026-09-11); the drift that remains on a given GPU is measured in the
replication table (make_paper_tables.py, section 6) rather than assumed away.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def enable_deterministic_torch(warn_only: bool = True) -> dict:
    """Turn on every determinism switch torch offers; return their state."""
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:  # very old torch without warn_only
        torch.use_deterministic_algorithms(True)
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
