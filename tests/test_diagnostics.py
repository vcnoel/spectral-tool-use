import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from spectral_guardrails.spectral.graph import symmetrize, aggregate_heads, laplacian
from spectral_guardrails.spectral.diagnostics import compute_all, fiedler_value, smoothness, spectral_entropy, hfer, _get_eigendecomposition

def test_laplacian_properties():
    print("Testing Laplacian properties...")
    N = 100
    W = torch.rand(N, N)
    W = (W + W.T) / 2 # Symmetric
    L = laplacian(W)
    
    # 1. Symmetry
    assert torch.allclose(L, L.T, atol=1e-6), "Laplacian is not symmetric"
    
    # 2. Positive semi-definiteness (eigenvalues >= 0)
    evals, _ = torch.linalg.eigh(L)
    assert (evals >= -1e-4).all(), f"Laplacian has negative eigenvalues: {evals.min()}"
    
    # 3. Null space: L @ ones = 0
    ones = torch.ones(N, 1)
    L_ones = L @ ones
    assert torch.max(torch.abs(L_ones)) < 2e-4, f"L @ ones should be zero, got max abs {torch.max(torch.abs(L_ones))}"
    print("Laplacian properties: PASS")

def test_fiedler():
    print("Testing Fiedler value...")
    N = 50
    # Fully connected graph with weights 1.0
    W = torch.ones(N, N)
    L = laplacian(W)
    f = fiedler_value(L)
    assert f > 0, f"Fiedler value for connected graph should be > 0, got {f}"
    print("Fiedler value: PASS")

def test_metrics_range():
    print("Testing metrics range [0, 1]...")
    for _ in range(100):
        N = torch.randint(10, 100, (1,)).item()
        d = 32
        W = torch.rand(N, N)
        L = laplacian(W)
        X = torch.randn(N, d)
        
        s = smoothness(L, X)
        assert -1e-6 <= s <= 1 + 1e-6, f"Smoothness {s} out of range [0, 1]"
        
        # HFER
        h = hfer(L, X)
        assert -1e-6 <= h <= 1 + 1e-6, f"HFER {h} out of range [0, 1]"
        
        # Entropy
        e = spectral_entropy(L, X)
        assert e >= -1e-6, f"Spectral entropy {e} should be >= 0"
        
    print("Metrics range: PASS")

def test_lanczos_consistency():
    print("Testing Lanczos consistency (N=400)...")
    N = 400
    W = torch.rand(N, N)
    W = (W + W.T) / 2
    L = laplacian(W)
    
    # Full eigh
    evals_full, _ = torch.linalg.eigh(L.to(torch.float32))
    
    # Lanczos internal fallback
    evals_lanc, _ = _get_eigendecomposition(L)
    
    # Check top-k eigenvalues
    k = 50
    evals_f = evals_full[:k]
    evals_l = evals_lanc[:k]
    
    # Use hybrid error: abs error for small values, rel error for larger
    # This avoids explosion at near-zero eigenvalues (0 and Fiedler)
    diff = torch.abs(evals_f - evals_l)
    error = diff / (torch.abs(evals_f) + 1.0) # Relative to (val + 1)
    max_err = torch.max(error).item()
    
    assert max_err < 0.05, f"Lanczos error too high: {max_err:.4f}"
    print(f"Lanczos consistency: PASS (max hybrid error: {max_err:.4f})")

if __name__ == "__main__":
    try:
        test_laplacian_properties()
        test_fiedler()
        test_metrics_range()
        test_lanczos_consistency()
        print("\nALL MATH TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
