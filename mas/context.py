from collections import deque
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from mas.hypercube import AdaptiveHypercube
from sklearn.linear_model import LinearRegression


class ContextAgent:
    def __init__(self, origin, side_lengths, memory_length, model_kwargs={}) -> None:
        self.validity = AdaptiveHypercube(origin, side_lengths)
        self.memory_length = memory_length
        self.dataset = deque([], maxlen=memory_length)
        self.target_dataset = deque([], maxlen=memory_length)

    def is_valid(self, x):
        return self.validity.contains(x).all()

    def update_confidence(self, feedback):
        raise NotImplementedError("update_confidence not implemented.")

    def predict(self, x, **kwargs):
        raise NotImplementedError("predict method not implemented.")

    def update(self, X, y):
        raise NotImplementedError("update method not implemented.")

    def is_mature(self):
        raise NotImplementedError("is_mature method not implemented.")

    def confidence(self, X):
        return 0

    def clean_dataset(self):
        idxs = [
            id for id in range(len(self.dataset)) if self.is_valid(self.dataset[id])
        ]
        self.dataset = deque(
            [self.dataset[id] for id in idxs], maxlen=self.memory_length
        )
        self.target_dataset = deque(
            [self.target_dataset[id] for id in idxs], maxlen=self.memory_length
        )
        self.n_seen = len(self.dataset)

    def expand_towards(self, X):
        vol = self.validity.volume()
        self.validity.expand_towards(X, self.alpha)
        return self.validity.volume() - vol

    def retract_towards(self, X):
        self.clean_dataset()
        vol = self.validity.volume()
        self.validity.retract_towards(X, self.alpha)
        return self.validity.volume() - vol


class LinearContextAgent(ContextAgent):
    def __init__(
        self, origin, side_lengths, alpha=0.1, memory_length=None, model_kwargs={}
    ) -> None:
        super().__init__(origin, side_lengths, memory_length, model_kwargs)
        self.local_model = LinearRegression(**model_kwargs)
        # self.local_model = SGDRegressor(eta0=0.1)
        self.n_seen = 0
        self.p = len(side_lengths)
        self.alpha = alpha

    def predict(self, X, **kwargs):
        if len(X.shape) < 2:
            if self.p > 1:
                X = X.reshape(1, -1)
            else:
                X = X.reshape(-1, 1)
        y = self.local_model.predict(X, **kwargs)
        return y

    def update(self, X, y):
        self.output_dim = y.shape[-1]

        self.dataset.append(X)
        self.target_dataset.append(y)

        X = np.array(self.dataset)
        y = np.array(self.target_dataset)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        if not y.shape[-1] > 1:
            y = y.reshape(-1, 1)

        self.local_model.fit(X, y)
        self.n_seen = len(self.dataset)

    def to_destroy(self):
        return (
            self.p > 1
            and self.n_seen > (self.p + 1)
            and np.all(
                np.isclose(pd.DataFrame(self.dataset).corr().abs(), 1.0, atol=1e-5)
            )
        )

    def is_mature(self):
        if self.p > 1:
            return self.n_seen > (self.p + 1) and not np.all(
                np.isclose(pd.DataFrame(self.dataset).corr().abs(), 1.0, atol=1e-5)
            )
        return self.n_seen > (self.p + 1)


class GaussianContextAgent(ContextAgent):
    def __init__(
        self,
        origin,
        side_lengths,
        alpha=0.1,
        memory_length=None,
        model_kwargs={"alpha": 0.1},
    ) -> None:
        super().__init__(origin, side_lengths, memory_length)

        self.local_model = GaussianProcessRegressor(**model_kwargs)
        # self.local_model = GaussianProcessRegressor(kernel, alpha=0.1)
        # self.local_model = BayesianRidge()
        self.n_seen = 0
        self.p = len(side_lengths)
        self.alpha = alpha

        self.dataset = deque([], maxlen=memory_length)
        self.target_dataset = deque([], maxlen=memory_length)

    def predict(self, X, **kwargs):
        if len(X.shape) < 2:
            if self.p > 1:
                X = X.reshape(1, -1)
            else:
                X = X.reshape(-1, 1)
        y = self.local_model.predict(X, **kwargs)
        return y

    def update(self, X, y):
        self.output_dim = y.shape[-1]
        self.dataset.append(X)
        self.target_dataset.append(y)
        self.n_seen = len(self.dataset)

        X = np.array(self.dataset)
        y = np.array(self.target_dataset)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        if not y.shape[-1] > 1:
            y = y.reshape(-1, 1)

        if self.is_mature():
            self.local_model.fit(X, y.squeeze())

    def to_destroy(self):
        return False

    def is_mature(self):
        return self.n_seen > (self.p + 1)

    def confidence(self, X):
        if len(X.shape) < 2:
            if self.p > 1:
                X = X.reshape(1, -1)
            else:
                X = X.reshape(-1, 1)
        _, std = self.local_model.predict(X, return_std=True)
        return np.mean(1 / std)
