"""Normalized all-pole synthesis using orthogonal scattering junctions.

Classical DSP, based on the equations in Julius O. Smith's Physical Audio Signal
Processing, 'Normalized Scattering Junctions' (C.66/C.67):
https://www.dsprelated.com/freebooks/pasp/Normalized_Scattering_Junctions.html
Implementation written here; no upstream source code copied.

For fixed reflection k, transfer is prod(sqrt(1-k*k)) / A(z). To match a
conventional 1/A(z) filter, divide its excitation by that numerator. State is
normalized backward waves, NOT the preceding output samples. Changing k does
not increase zero-input stored state energy. This does not bound arbitrary
driven output to playback levels or make the vocal model physically complete.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _scatter(excitation, reflection, transmission, state):
    output = np.empty(len(excitation))
    order = len(state)
    for n in range(len(excitation)):
        forward = excitation[n]
        for m in range(order-1, -1, -1):
            backward = state[m]
            next_forward = transmission[m]*forward - reflection[m]*backward
            # The outgoing highest-order wave is absorbed at the termination.
            # Descending writes leave every not-yet-read old state intact.
            if m+1 < order:
                state[m+1] = reflection[m]*forward + transmission[m]*backward
            forward = next_forward
        state[0] = forward
        output[n] = forward
    return output


@njit(cache=True)
def _scatter_varying(excitation, reflection, transmission, state):
    output = np.empty(len(excitation))
    order = len(state)
    for n in range(len(excitation)):
        forward = excitation[n]
        for m in range(order-1, -1, -1):
            backward = state[m]
            next_forward = transmission[n,m]*forward - reflection[n,m]*backward
            if m+1 < order:
                state[m+1] = reflection[n,m]*forward + transmission[n,m]*backward
            forward = next_forward
        state[0] = forward
        output[n] = forward
    return output


def normalized_lattice(excitation, reflection, state, previous_reflection=None):
    """Filter one block, updating the caller's normalized state in place."""
    excitation = np.asarray(excitation, dtype=np.float64)
    reflection = np.asarray(reflection, dtype=np.float64)
    if (excitation.ndim != 1 or reflection.ndim != 1 or not len(reflection)
            or not np.isfinite(excitation).all() or not np.isfinite(reflection).all()
            or np.any(np.abs(reflection)>=1)):
        raise ValueError('Expected finite excitation and strictly stable reflection coefficients')
    if (not isinstance(state,np.ndarray) or state.dtype != np.float64
            or state.shape != reflection.shape or not state.flags.writeable
            or not np.isfinite(state).all()):
        raise ValueError('Expected writable finite float64 lattice state')
    if previous_reflection is None:
        return _scatter(excitation,reflection,np.sqrt(1-reflection*reflection),state)
    previous=np.asarray(previous_reflection,dtype=np.float64)
    if (previous.shape!=reflection.shape or not np.isfinite(previous).all()
            or np.any(np.abs(previous)>=1)):
        raise ValueError('Invalid preceding reflection coefficients')
    if not len(excitation):return np.empty(0)
    # Interpolate scattering angles, preserving orthogonal stages at every
    # sample. This uses current/previous controls, never the following frame.
    old=np.arcsin(previous);new=np.arcsin(reflection)
    blend=(np.arange(len(excitation))+1)/len(excitation)
    angles=old[None,:]+blend[:,None]*(new-old)[None,:]
    return _scatter_varying(excitation,np.sin(angles),np.cos(angles),state)
