"""Classical step-up/down conversion between LPC and reflection coefficients."""
import numpy as np


def to_reflection(coefficients):
    a = np.asarray(coefficients, dtype=np.float64).copy()
    if a.ndim != 1 or len(a) < 2 or a[0] != 1 or not np.isfinite(a).all():
        raise ValueError('Expected a finite monic filter')
    result = np.empty(len(a)-1)
    for order in range(len(result), 0, -1):
        k = a[order]
        if abs(k) >= 1:
            raise ValueError('Filter is not strictly stable')
        result[order-1] = k
        a = (a[:order] - k*a[order:0:-1])/(1-k*k)
    return result


def from_reflection(reflection):
    reflection = np.asarray(reflection, dtype=np.float64)
    if reflection.ndim != 1 or not np.isfinite(reflection).all() or np.any(np.abs(reflection)>=1):
        raise ValueError('Reflection coefficients must be finite and strictly inside (-1,1)')
    a = np.array([1.])
    for k in reflection:
        a = np.r_[a, 0.] + k*np.r_[0., a[::-1]]
    return a
