import numpy as np
import copy


class AdaptiveHypercube:
    def __init__(self, x, side_lengths) -> None:
        assert x.shape == side_lengths.shape
        self.low, self.high = x - side_lengths / 2, x + side_lengths / 2
        self.p = x.shape[0]

    @property
    def bounding_box(self):
        return np.concatenate([self.low, self.high])

    def dist_to_outer_boundaries(self, x):
        dist_to_center = x - self.center()
        side_lengths = self.side_lengths() / 2
        boundary_to_x = dist_to_center - side_lengths * np.sign(dist_to_center)
        boundary_to_x[self.contains(x)] = 0.0
        return np.sum(np.square(boundary_to_x))

    def contains(self, x):
        return (self.low <= x) & (x <= self.high)

    def _update_towards(self, x, alpha, dims):
        dist_low = np.abs(self.low - x)
        dist_high = np.abs(self.high - x)
        diff = self.side_lengths()

        p = np.sum(dims)

        new_high = diff * np.power((1 + alpha), 1 / p) + self.low
        new_low = self.high - diff * np.power((1 + alpha), 1 / p)
        mask = dist_high < dist_low
        self.high[mask & dims] = new_high[mask & dims]
        self.low[~mask & dims] = new_low[~mask & dims]

    def expand_towards(self, x, alpha):
        # assert not self.contains(x).all()
        contains = self.contains(x)
        if not contains.all():
            self._update_towards(x, alpha, dims=~contains)
        # else:
        #     self._update_towards(x, alpha, dims=contains)

    def retract_towards(self, x, alpha):
        assert self.contains(x).all()
        contains = self.contains(x)
        self._update_towards(x, -alpha, contains)

    def _udpdate(self, alpha):
        self.high = self.high * np.power(1 + alpha, 1 / self.p)
        self.low = self.low * np.power(1 + alpha, 1 / self.p)

    def expand(self, alpha):
        self._udpdate(alpha)

    def retract(self, alpha):
        self._udpdate(-alpha)

    def side_lengths(self):
        return abs(self.high - self.low)

    def volume(self):
        diff = self.side_lengths()
        vol = np.prod(diff)
        return vol

    def center(self):
        return (self.high + self.low) / 2

    def duplicate(self):
        h2 = AdaptiveHypercube(self.center(), self.side_lengths())
        return h2

    def __repr__(self) -> str:
        return f"Rectangle({self.low}, {self.high})"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, AdaptiveHypercube):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return (self.low == __value.low).all() and (self.high == __value.high).all()


# compute overlap
def overlap(h1: AdaptiveHypercube, h2: AdaptiveHypercube) -> AdaptiveHypercube | None:
    max_start = np.maximum(h1.low, h2.low)
    min_end = np.minimum(h1.high, h2.high)
    if (max_start >= min_end).all():
        return None  # no overlap
    elif (max_start == h2.low).all() and (min_end == h2.high).all():
        return copy.deepcopy(h2)
    else:
        x = (min_end + max_start) / 2
        side_lengths = min_end - max_start
        return AdaptiveHypercube(x, side_lengths)


def overlapping_index(h1: AdaptiveHypercube, h2: AdaptiveHypercube):
    inter = overlap(h1, h2)
    if inter is None:
        return 0
    return inter.volume() / np.min([h1.volume(), h2.volume()])


# make h1 push h2 (minimum volume reduction for h2)
def push(h1: AdaptiveHypercube, h2: AdaptiveHypercube):
    inter = overlap(h1, h2)
    assert (inter is not None) and (not inter == h2)
    inter_sidelengths = inter.side_lengths()
    dim_to_push = inter_sidelengths.argmin()

    new_low = np.maximum(h2.low[dim_to_push], h1.high[dim_to_push])
    new_high = np.maximum(h2.high[dim_to_push], h1.low[dim_to_push])
    h2.low[dim_to_push] = new_low
    h2.high[dim_to_push] = new_high
