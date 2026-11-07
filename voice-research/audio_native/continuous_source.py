"""Reference-fitted continuous source coordinates; classical PCA, no codebook.

This tests whether varying source detail matters beyond a single pulse profile.
It encodes analysis-time residual descriptors, not linguistic content. Decoder
inputs are real-valued coordinates plus the existing continuous vocal controls.
This is a small analysis/synthesis representation, not learned generative speech.
"""
import numpy as np


def descriptor(frame, harmonics=32, rate=24000):
    if frame.f0<=0 or frame.periodicity<.6:
        return None
    window=np.hanning(len(frame.excitation_window))
    times=np.arange(len(window))/rate
    k=np.arange(1,harmonics+1)
    coefficients=2*(np.exp(-2j*np.pi*k[:,None]*frame.f0*times)
                    @(window*frame.excitation_window))/sum(window)
    coefficients[k*frame.f0>=rate*.47]=0
    energy=np.sqrt(.5*np.sum(np.abs(coefficients)**2))
    if energy<1e-5 or abs(coefficients[0])/energy<.05:return None
    aligned=coefficients*np.exp(-1j*k*np.angle(coefficients[0]))/energy
    return np.r_[aligned.real,aligned.imag]


class ContinuousSource:
    def __init__(self,mean,basis):
        self.mean=np.asarray(mean,dtype=np.float64)
        self.basis=np.asarray(basis,dtype=np.float64)
        if (self.mean.ndim!=1 or len(self.mean)<2 or len(self.mean)%2
                or self.basis.ndim!=2 or self.basis.shape[1]!=len(self.mean)):
            raise ValueError('Invalid continuous source model')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.basis).all():
            raise ValueError('Nonfinite continuous source model')

    @classmethod
    def fit(cls,frames,dimensions=8,harmonics=32):
        if not isinstance(dimensions,int) or not 0<=dimensions<=2*harmonics:
            raise ValueError('Invalid source dimension count')
        rows=[value for frame in frames if (value:=descriptor(frame,harmonics=harmonics)) is not None]
        if len(rows)<max(10,dimensions+1):raise ValueError('Reference lacks usable voiced frames')
        data=np.array(rows);mean=data.mean(axis=0)
        _,_,basis=np.linalg.svd(data-mean,full_matrices=False)
        return cls(mean,basis[:dimensions]),len(rows)

    def encode(self,frame):
        value=descriptor(frame,harmonics=len(self.mean)//2)
        return np.zeros(len(self.basis)) if value is None else self.basis@(value-self.mean)

    def decode(self,coordinates):
        coordinates=np.asarray(coordinates,dtype=np.float64)
        if coordinates.shape!=(len(self.basis),) or not np.isfinite(coordinates).all():
            raise ValueError('Invalid continuous source coordinates')
        value=self.mean+coordinates@self.basis
        half=len(self.mean)//2
        return value[:half]+1j*value[half:]
