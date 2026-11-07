"""Archived pre-fix LPC for reproducing earlier research results ONLY.

Do not use in new synthesis. In-recursion clipping invalidates the subsequent
normal-equation/error recurrence. Human evaluation exposed extreme coefficients.
The active corrected implementation is streaming_voicebox.lpc (LPC_VERSION 2).
"""
import numpy as np


def lpc(history,config):
    signal=(history-np.mean(history))*np.hanning(len(history))
    autocorrelation=np.array([signal[k:]@signal[:len(signal)-k] for k in range(config.order+1)])
    autocorrelation[0]+=1e-6+autocorrelation[0]*1e-4
    coefficients=np.array([1.]);error=autocorrelation[0]
    for order in range(1,config.order+1):
        reflection=-(autocorrelation[order]+coefficients[1:]@autocorrelation[order-1:0:-1])/error
        reflection=np.clip(reflection,-.98,.98)
        coefficients=np.r_[coefficients,0.]+reflection*np.r_[0.,coefficients[::-1]]
        error=max(error*(1-reflection*reflection),1e-12)
    return coefficients*config.bandwidth**np.arange(config.order+1)
