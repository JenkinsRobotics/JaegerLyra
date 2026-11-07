"""Delayed-feedback continuous auditory-feature experiment, not language learning.

The forecaster sees at most 550 ms of past feature frames and predicts 20–100 ms
ahead. Actual future frames enter observe() only when they arrive. There is no
backpropagation through time. Raw audio remains in the independent residual path.
"""
from collections import deque
import numpy as np

LAGS = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55)


class AuditoryForecast:
    def __init__(self, model, horizon, adaptive=True, lags=LAGS):
        if horizon < 1 or not lags or min(lags)!=0 or list(lags)!=sorted(set(lags)):
            raise ValueError('Invalid causal forecast configuration')
        self.model = model
        self.horizon = horizon
        self.adaptive = adaptive
        self.lags = lags
        self.history = deque(maxlen=max(lags)+1)
        self.pending = deque()
        self.index = -1

    def push(self, feature):
        feature = np.asarray(feature, dtype=np.float64)
        if feature.ndim!=1 or not np.isfinite(feature).all():
            raise ValueError('Invalid auditory features')
        self.index += 1
        completed = None
        if self.pending and self.pending[0]['targetIndex'] == self.index:
            completed = self.pending.popleft()
            # Its prediction was committed horizon frames ago. Only now is its
            # actual target available to local weight adaptation.
            if self.model is not None:
                self.model.observe(completed['context'], feature-completed['base'], learn=self.adaptive)
            completed['target'] = feature.copy()
        self.history.append(feature.copy())
        emitted = None
        if len(self.history) > max(self.lags):
            context = np.concatenate([self.history[-lag-1] for lag in self.lags])
            delta = self.model.predict(context) if self.model is not None else np.zeros_like(feature)
            emitted = {'at':self.index,'targetIndex':self.index+self.horizon,
                       'context':context,'base':feature.copy(),
                       'prediction':feature+delta}
            self.pending.append(emitted)
        return emitted, completed
