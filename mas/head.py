import time
import numpy as np
import pandas as pd
import tqdm
import pickle

from sklearn.base import BaseEstimator
from rtree import index
from mas.context import LinearContextAgent
from mas.hypercube import AdaptiveHypercube, overlap, push, overlapping_index
from dataclasses import dataclass, fields
from scipy.special import softmax


def save(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_from_pickle(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    if isinstance(model, FastHeadAgent):
        # build agent index
        model.build_index()
    return model


@dataclass
class Metrics:
    time: float = 0
    nb_agents: int = 0
    nb_destroyed: int = 0
    nb_created: int = 0
    nb_updated: int = 0
    nb_mature: int = 0
    volume_gained: float = 0
    volume_lost: float = 0
    ncs1: int = 0
    ncs2: int = 0
    ncs3: int = 0

    def __add__(self, other):
        return Metrics(
            *(
                getattr(self, field.name) + getattr(other, field.name)
                for field in fields(self)
            )
        )


class HeadAgent(BaseEstimator):
    def __init__(
        self,
        R,
        imprecise_th,
        bad_th,
        alpha,
        min_vol,
        memory_length=None,
        context_cls=LinearContextAgent,
        context_kwargs={},
        step_callback=[],
        epoch_callback=[],
    ) -> None:
        self.context_cls = context_cls
        self.context_kwargs = context_kwargs
        self.context_agents = {}
        self.R = R  # initial radius of context agents
        self.imprecise_th = imprecise_th  # beyond prediction is imprecise
        self.bad_th = bad_th  # beyond prediction is bad
        self.alpha = alpha  # context agents volume variation coefficient
        self.min_vol = min_vol  # minimum volume for an agent to not be destroyed
        self.neighbor_radius = R  # radius below which agent is neighbor
        self.n_created = 0
        self.memory_length = memory_length
        self.n_input_dim = len(self.R)
        # callbacks used only for debug purpose at the moment
        self.step_callback = step_callback
        self.epoch_callback = epoch_callback

        self.reset()

    def reset(self):
        self.context_agents = {}
        # logging utilities for training
        self.epoch_stats = pd.DataFrame()
        self.stats = pd.DataFrame()

    def valid_agents(self, X):
        agents = [
            (k, a)
            for k, a in self.context_agents.items()
            if a.validity.contains(X).all()
        ]
        return agents

    def create_agent(self, X, radius):
        self.n_created += 1
        new_agent = self.context_cls(
            X,
            radius,
            alpha=self.alpha,
            memory_length=self.memory_length,
            model_kwargs=self.context_kwargs,
        )
        self.context_agents[self.n_created] = new_agent
        return new_agent, self.n_created

    def destroy_agent(self, key):
        # delete in index then in dict
        self.context_agents.pop(key)

    def get_neighbors(self, X):
        neighborhood = AdaptiveHypercube(X, self.neighbor_radius)
        return [
            (k, a)
            for k, a in self.context_agents.items()
            if overlap(neighborhood, a.validity) is not None
        ]

    def retrieve_propositions(self, X, **kwargs):
        agents = self.valid_agents(X)
        propositions = [a.predict(X, **kwargs) for _, a in agents]
        return propositions, agents

    def get_closest_agent(self, X):
        dists = [
            a.validity.dist_to_outer_boundaries(X) for a in self.context_agents.values()
        ]
        closest = list(self.context_agents.items())[np.argmin(dists)]
        return closest

    def _get_pred(self, x, **kwargs):
        propositions, agents = self.retrieve_propositions(x, **kwargs)
        n_propositions = len([p for p in propositions if p is not None])
        if n_propositions == 0:  # find closest agent
            key, closest = self.get_closest_agent(x)
            preds = [a.predict(x, **kwargs) for a in closest]
            pred = np.mean(preds)
        else:
            # compute confidence values
            confidences = [a.confidence(x) for _, a in agents]
            pred = propositions[np.argmax(confidences)]
        return np.mean(pred), agents

    def predict(self, X, return_agents=False, **kwargs):
        if len(X.shape) < 1:
            X = X.reshape(-1, 1)
        predictions = []
        agents_hist = []
        for x in X:
            pred, agents = self._get_pred(x, **kwargs)
            agents_hist.append(agents)
            if isinstance(pred, tuple):
                predictions.append([p.squeeze() for p in pred])
            else:
                predictions.append(pred.ravel())

        if isinstance(pred, tuple):
            payload = [np.array(p) for p in zip(*predictions)]
        else:
            payload = np.array(predictions).squeeze()
        if return_agents:
            return payload, agents_hist
        return payload

    def score(self, y_predict, y):
        return np.mean((y_predict - y) ** 2)

    def _update_validity(self, key, agent, update_fn, X):
        volume_diff = update_fn(X)
        return volume_diff

    def partial_fit(self, X, y):
        _step_metrics = Metrics()

        agents = self.valid_agents(X)
        agents_to_update = []
        agents_to_destroy = set({})
        n_valid = len(agents)
        new_agent = None
        # incompetence (no agent available)
        if n_valid == 0:
            neighbors = self.get_neighbors(X)
            create_agent = True
            if len(self.context_agents) > 0:
                _step_metrics.ncs1 += 1
                for key, a in neighbors:
                    if a.is_mature():
                        expanded = a.validity.duplicate()
                        expanded.expand(a.alpha)
                        if expanded.contains(X).all():
                            score = self.score(a.predict(X), y)
                            good = score <= self.imprecise_th
                            bad = score > self.bad_th
                            if not bad:
                                _step_metrics.volume_gained += self._update_validity(
                                    key, a, a.expand_towards, X
                                )
                                if not good:
                                    agents_to_update.append((key, a))
                                create_agent = False
            if create_agent:
                radius = self.R
                if len(neighbors) > 1:
                    radius = np.array([n.validity.side_lengths() for _, n in neighbors])
                    radius = np.mean(radius, axis=0)
                new_agent, key = self.create_agent(X, radius)
                agents_to_update.append((key, new_agent))
        # conflict / concurrence
        if n_valid > 1:
            _step_metrics.ncs2 += 1
            for key, a in agents:
                if a.is_mature():
                    # check quality of prediction
                    score = self.score(a.predict(X), y)
                    # classify score into good / imprecise / bad prediction
                    good = score <= self.imprecise_th
                    bad = score > self.bad_th
                    if not bad:
                        _step_metrics.volume_gained += self._update_validity(
                            key, a, a.expand_towards, X
                        )
                    if bad:
                        _step_metrics.volume_lost += self._update_validity(
                            key, a, a.retract_towards, X
                        )
                    if not bad and not good:
                        agents_to_update.append((key, a))
                else:
                    agents_to_update.append((key, a))

        # incompetence (correctness of prediction)
        if n_valid == 1:
            _step_metrics.ncs3 += 1
            key, agent = agents[0]
            score = self.score(agent.predict(X), y)
            # classify score into good / imprecise / bad prediction
            good = score <= self.imprecise_th
            bad = score > self.bad_th
            if agent.is_mature():
                if not bad:
                    _step_metrics.volume_gained += self._update_validity(
                        key, agent, agent.expand_towards, X
                    )
                if bad:
                    _step_metrics.volume_lost += self._update_validity(
                        key, agent, agent.retract_towards, X
                    )
                if not bad and not good:
                    agents_to_update.append((key, agent))
            else:
                agents_to_update.append((key, agent))

        # update agents if needed
        for k, a in agents_to_update:
            a.update(X, y)
            if a.to_destroy():
                agents_to_destroy.add(k)

        # destroy all designated agents
        for k in agents_to_destroy:
            self.destroy_agent(k)

        # logging training metrics
        _step_metrics.nb_updated = len(agents_to_update)
        _step_metrics.nb_destroyed = len(agents_to_destroy)
        _step_metrics.nb_created = 1 if new_agent is not None else 0
        return _step_metrics

    def fit(
        self,
        X,
        y,
        epochs=50,
        batch_size=None,
        early_stop=True,
        shuffle=True,
        verbose=False,
    ):
        self.reset()
        nb_mature = 0
        for e in range(epochs):
            t = time.time()

            metrics_list = []
            epoch_metrics = Metrics()
            idxs = np.arange(len(X))
            if shuffle:
                np.random.shuffle(idxs)

            pbar = tqdm.tqdm(idxs[:batch_size], disable=(not verbose))
            for idx in pbar:
                x = X[idx]
                y_hat = y[idx]
                metrics = self.partial_fit(x.ravel(), y_hat.ravel())
                epoch_metrics += metrics
                metrics_list.append(metrics)
                for f in self.step_callback:
                    f(X=x, y=y_hat)
                pbar.set_description(
                    f"[agents: {len(self.context_agents)} | maturity: {nb_mature/len(self.context_agents)*100:.2f}% | vol+: {epoch_metrics.volume_gained:.4f} | vol-: {epoch_metrics.volume_lost:.4f}]"
                )

            # destroy low volume agents
            agents_to_destroy = set({})
            for k, a in self.context_agents.items():
                if (a.validity.side_lengths() < self.min_vol).any():
                    agents_to_destroy.add(k)

            epoch_metrics.nb_destroyed += len(agents_to_destroy)
            # destroy all designated agents
            for k in agents_to_destroy:
                self.destroy_agent(k)

            epoch_metrics.nb_agents = len(self.context_agents)
            nb_mature = len([a for a in self.context_agents.values() if a.is_mature()])
            epoch_metrics.nb_mature = nb_mature
            epoch_metrics.time = time.time() - t

            # log epoch metrics
            self.epoch_stats = pd.concat(
                [self.epoch_stats, pd.DataFrame([epoch_metrics])], ignore_index=True
            )
            # log step metrics
            self.stats = pd.concat(
                [self.stats, pd.DataFrame(metrics_list)], ignore_index=True
            )

            for f in self.epoch_callback:
                f(self, epoch_metrics.__dict__, e)

            # early stopping if system does not evolve
            if (
                early_stop
                and self.epoch_stats["nb_mature"].iloc[-1] > 0
                and self.epoch_stats["nb_created"].iloc[-1] == 0
                and (
                    self.epoch_stats["nb_mature"].iloc[-1]
                    == self.epoch_stats["nb_agents"].iloc[-1]
                    or self.epoch_stats["nb_mature"].iloc[-1]
                    == self.epoch_stats["nb_mature"].iloc[-2]
                )
                and (
                    self.epoch_stats[["volume_gained", "volume_lost"]].iloc[-1] == 0
                ).all()
            ):
                print(f"Early stopping because system not evolving... [{e}]")
                break

        agents_to_destroy = set({})
        self.dropped_immature = []
        # get rid of immature agents
        for k, a in self.context_agents.items():
            if not a.is_mature():
                agents_to_destroy.add(k)
                self.dropped_immature.append(self.context_agents[k])
        for k in agents_to_destroy:
            self.destroy_agent(k)


class FastHeadAgent(HeadAgent):
    def __init__(
        self,
        R,
        imprecise_th,
        bad_th,
        alpha,
        min_vol,
        memory_length=None,
        context_cls=LinearContextAgent,
        context_kwargs={},
        step_callback=[],
        epoch_callback=[],
    ) -> None:
        assert len(R) > 1
        super().__init__(
            R,
            imprecise_th,
            bad_th,
            alpha,
            min_vol,
            memory_length,
            context_cls,
            context_kwargs,
            step_callback,
            epoch_callback,
        )
        self.reset()

    def reset(self):
        super().reset()
        self.build_index()

    def valid_agents(self, X):
        agents = [(a, self.context_agents[a]) for a in self.agent_index.intersection(X)]
        return agents

    def create_agent(self, X, radius):
        new_agent, key = super().create_agent(X, radius)
        self.agent_index.insert(key, new_agent.validity.bounding_box)
        return new_agent, key

    def destroy_agent(self, key):
        self.agent_index.delete(key, self.context_agents[key].validity.bounding_box)
        return super().destroy_agent(key)

    def get_neighbors(self, X):
        neighborhood = AdaptiveHypercube(X, self.neighbor_radius)
        return [
            (a, self.context_agents[a])
            for a in self.agent_index.intersection(neighborhood.bounding_box)
        ]

    def get_closest_agent(self, X):
        neighborhood_sides = self.R
        neighborhood = AdaptiveHypercube(X, neighborhood_sides)
        neighborhood_agents_keys = [
            i for i in self.agent_index.intersection(neighborhood.bounding_box)
        ]
        neighborhood_agents = [self.context_agents[k] for k in neighborhood_agents_keys]

        if len(neighborhood_agents) == 0:
            closest_key = next(self.agent_index.nearest(X, num_results=1))
            closest = self.context_agents[closest_key]
            return [closest_key], [closest]

        return neighborhood_agents_keys, neighborhood_agents

    def _update_validity(self, key, agent, update_fn, X):
        self.agent_index.delete(key, agent.validity.bounding_box)
        volume_diff = super()._update_validity(key, agent, update_fn, X)
        self.agent_index.insert(key, agent.validity.bounding_box)
        return volume_diff

    def build_index(self):
        p = index.Property()
        p.dimension = len(self.R)
        self.agent_index = index.Index(properties=p)
        for k, a in self.context_agents.items():
            self.agent_index.insert(k, a.validity.bounding_box)


class HybridHeadAgent(FastHeadAgent):
    def __init__(
        self,
        R,
        imprecise_th,
        bad_th,
        alpha,
        min_vol,
        memory_length=None,
        context_cls=LinearContextAgent,
        context_kwargs={},
        step_callback=[],
        epoch_callback=[],
    ) -> None:
        super().__init__(
            R,
            imprecise_th,
            bad_th,
            alpha,
            min_vol,
            memory_length,
            context_cls,
            context_kwargs,
            step_callback,
            epoch_callback,
        )
        self.eval_mode = False

    def train(self):
        self.eval_mode = False

    def eval(self):
        self.eval_mode = True
        self.build_index()

    def create_agent(self, X, radius):
        return super(FastHeadAgent, self).create_agent(X, radius)

    def destroy_agent(self, key):
        return super(FastHeadAgent, self).destroy_agent(key)

    def _update_validity(self, key, agent, update_fn, X):
        return super(FastHeadAgent, self)._update_validity(key, agent, update_fn, X)

    def valid_agents(self, X):
        if self.eval_mode:
            return super().valid_agents(X)
        return super(FastHeadAgent, self).valid_agents(X)

    def get_neighbors(self, X):
        if self.eval_mode:
            return super().get_neighbors(X)
        return super(FastHeadAgent, self).get_neighbors(X)

    def get_closest_agent(self, X):
        if self.eval_mode:
            return super().get_closest_agent(X)
        return super(FastHeadAgent, self).get_closest_agent(X)

    def fit(self, X, y, epochs=10, batch_size=None, early_stop=True, verbose=False):
        self.train()
        res = super().fit(X, y, epochs, batch_size, early_stop, verbose)
        self.eval()
        return res
