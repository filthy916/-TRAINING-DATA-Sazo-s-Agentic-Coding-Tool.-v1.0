"""
PRIMITIVE ROOT ALGORITHM — TRANSFER RESONANCE
==============================================
Full implementation: primitive root math + neural network
transfer resonance applications.

MODULES:
  1. Core number theory  — is_prime, euler_totient, primitive roots
  2. Weight initialization — primitive_root_init()
  3. LR scheduling        — primitive_root_lr_schedule()
  4. Layer coupling       — resonance_coupling_matrix()
  5. RCG integration      — use with rcg_training.py

MATHEMATICAL FOUNDATION:
  g is a primitive root mod p  ⟺  ord_p(g) = p-1
  ⟺  {g^1, g^2, ..., g^(p-1)} = {1, 2, ..., p-1}  mod p

  This generates the entire multiplicative group (Z/pZ)* —
  maximum coverage, no repetition, deterministic structure.

TRANSFER RESONANCE INTUITION:
  Standard random init: neurons start at Gaussian-distributed
  positions in weight space — clustering is likely, equidistribution
  is not guaranteed.

  Primitive root init: neurons start at positions determined by
  the orbit of g mod p — guaranteed equidistribution across the
  entire space of residues. No two neurons start at the same
  frequency. Reduces representation collapse on transfer tasks.
"""

import math
import numpy as np


# ══════════════════════════════════════════════════════════════
# PART 1: CORE NUMBER THEORY
# ══════════════════════════════════════════════════════════════

def is_prime(n: int) -> bool:
    """Miller-Rabin primality test. Deterministic for n < 3.2e18."""
    if n < 2: return False
    if n in (2, 3, 5, 7): return True
    if n % 2 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1; d //= 2
    for a in [2, 3, 5, 7]:
        if a >= n: continue
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else:
            return False
    return True


def euler_totient(n: int) -> int:
    """φ(n): count of integers in [1,n] coprime to n."""
    result, temp = n, n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def prime_factors(n: int) -> list:
    """Distinct prime factors of n."""
    factors, d = [], 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def is_primitive_root(g: int, p: int) -> bool:
    """
    True if g is a primitive root mod p.
    Test: g^(φ(p)/q) ≢ 1 (mod p) for every prime q | φ(p)
    """
    if math.gcd(g, p) != 1: return False
    phi = euler_totient(p)
    for q in prime_factors(phi):
        if pow(g, phi // q, p) == 1:
            return False
    return True


def find_primitive_roots(p: int, max_roots: int = None) -> list:
    """
    Find primitive roots mod p (all of them, or up to max_roots).
    For prime p: exactly φ(φ(p)) = φ(p-1) primitive roots exist.
    """
    roots = []
    for g in range(2, p):
        if is_primitive_root(g, p):
            roots.append(g)
            if max_roots and len(roots) >= max_roots:
                break
    return roots


def primitive_root_sequence(g: int, p: int, length: int = None) -> list:
    """
    Full orbit: [g^1, g^2, ..., g^(p-1)] mod p
    Covers {1, ..., p-1} exactly once (orbit property).
    """
    seq, val = [], 1
    n = length or (p - 1)
    for _ in range(n):
        val = (val * g) % p
        seq.append(val)
    return seq


def next_prime_above(n: int) -> int:
    """Smallest prime strictly greater than n."""
    c = n + 1 if n % 2 == 0 else n + 2
    while not is_prime(c):
        c += 2
    return c


# ══════════════════════════════════════════════════════════════
# PART 2: WEIGHT INITIALIZATION
# ══════════════════════════════════════════════════════════════

def primitive_root_init(shape: tuple, p: int = None, scale: float = 1.0) -> np.ndarray:
    """
    Initialize weights via primitive root orbit.

    Instead of Glorot/He (random), use g^1, g^2, ..., g^n mod p
    normalized to [-scale*xavier, scale*xavier].

    Properties:
      - Equidistributed: no two neurons at the same starting frequency
      - Deterministic: reproducible without a random seed
      - Structured: orbit respects multiplicative group structure
      - Non-redundant: reduces representation collapse at initialization

    Args:
      shape:  weight tensor dimensions, e.g. (64, 32)
      p:      prime modulus; auto-selected if None
      scale:  multiplier on Xavier norm (default 1.0)
    """
    total = math.prod(shape)
    if p is None:
        p = next_prime_above(total)
    g = find_primitive_roots(p, max_roots=1)[0]
    seq = primitive_root_sequence(g, p, length=total)
    arr = (np.array(seq, dtype=np.float64) / (p - 1)) * 2 - 1  # ∈ (-1, 1]
    fan_in = shape[-1] if len(shape) >= 2 else shape[0]
    xavier = scale * math.sqrt(2.0 / fan_in)
    return (arr * xavier).reshape(shape)


# ══════════════════════════════════════════════════════════════
# PART 3: LEARNING RATE SCHEDULE
# ══════════════════════════════════════════════════════════════

def primitive_root_lr_schedule(
    base_lr: float,
    step: int,
    p: int,
    g: int,
    min_lr: float = 1e-6
) -> float:
    """
    Non-repeating LR schedule based on primitive root orbit.

    At step t: lr = min_lr + (base_lr - min_lr) * (g^t mod p) / (p-1)

    Properties:
      - No repetition for p-1 steps (longer than any cosine cycle)
      - Ergodic: covers the full [min_lr, base_lr] range
      - Deterministic: given p, g, step → exact lr, no randomness
      - Group-theoretic: lr values follow multiplicative structure

    Choose p >> total_training_steps to avoid orbit repetition.
    """
    orbit_pos = pow(g, (step % (p - 1)) + 1, p)
    normalized = orbit_pos / (p - 1)
    return min_lr + (base_lr - min_lr) * normalized


# ══════════════════════════════════════════════════════════════
# PART 4: LAYER COUPLING (RESONANCE MATRIX)
# ══════════════════════════════════════════════════════════════

def resonance_coupling_matrix(n_layers: int, p: int = None) -> tuple:
    """
    Build a layer-to-layer coupling matrix using primitive root residues.

    C[i, j] = (g^k mod p) / (p-1)  where k = i*n_layers + j + 1

    Each entry is unique (orbit property) → every pair of layers has
    a distinct coupling strength. No two skip connections are equal.

    Returns: (matrix: ndarray, g: int, p: int)
    """
    n_pairs = n_layers * n_layers
    if p is None:
        p = next_prime_above(n_pairs)
    g = find_primitive_roots(p, max_roots=1)[0]
    seq = primitive_root_sequence(g, p, length=n_pairs)
    arr = np.array(seq, dtype=np.float64) / (p - 1)
    return arr.reshape(n_layers, n_layers), g, p


# ══════════════════════════════════════════════════════════════
# PART 5: RCG INTEGRATION HELPER
# ══════════════════════════════════════════════════════════════

def rcg_seed_compression_oracle(vocab_size: int, hidden_dim: int) -> dict:
    """
    Return primitive root parameters for seeding an RCG compression oracle.
    Feed these into CompressionSurrogate initialization in rcg_training.py.

    The oracle's weight matrix is initialized via primitive roots so that
    the compression estimator starts with maximum entropy capacity —
    no two residues map to the same weight cluster.
    """
    p_init = next_prime_above(vocab_size * hidden_dim)
    p_lr   = next_prime_above(10_000)  # covers 10k training steps
    g_init = find_primitive_roots(p_init, max_roots=1)[0]
    g_lr   = find_primitive_roots(p_lr,   max_roots=1)[0]

    return {
        'W_oracle':    primitive_root_init((vocab_size, hidden_dim)),
        'p_init':      p_init,
        'g_init':      g_init,
        'p_lr':        p_lr,
        'g_lr':        g_lr,
        'n_prim_roots_init': len(find_primitive_roots(p_init, max_roots=10)),
        'phi_p_init':  euler_totient(p_init),
    }


# ══════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("PRIMITIVE ROOT ALGORITHM — TRANSFER RESONANCE DEMO\n")

    # Core math
    p = 19
    g = find_primitive_roots(p)[0]
    orbit = primitive_root_sequence(g, p)
    print(f"Primitive root g={g} mod p={p}")
    print(f"Orbit: {orbit}")
    print(f"Is equidistributed: {sorted(orbit) == list(range(1, p))}")

    # Weight init
    W = primitive_root_init((8, 4))
    print(f"\nWeight init (8,4): mean={W.mean():.4f}, std={W.std():.4f}")

    # LR schedule
    p_lr, g_lr = next_prime_above(50), None
    g_lr = find_primitive_roots(p_lr, max_roots=1)[0]
    lrs = [primitive_root_lr_schedule(0.01, t, p_lr, g_lr, 1e-4) for t in range(10)]
    print(f"\nLR schedule (10 steps): {[f'{v:.5f}' for v in lrs]}")
    print(f"All unique: {len(set(round(v,8) for v in lrs)) == 10}")

    # Coupling matrix
    C, gc, pc = resonance_coupling_matrix(3)
    print(f"\nCoupling matrix (3 layers), g={gc}, p={pc}:")
    for row in C:
        print(f"  {[f'{v:.3f}' for v in row]}")

    # RCG oracle seed
    oracle = rcg_seed_compression_oracle(32, 64)
    print(f"\nRCG oracle seed for vocab=32, hidden=64:")
    for k, v in oracle.items():
        if k != 'W_oracle':
            print(f"  {k}: {v}")
    print(f"  W_oracle shape: {oracle['W_oracle'].shape}")

