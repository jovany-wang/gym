"""Implementation of a space that represents the cartesian product of `Set` spaces."""
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from gym import logger
from gym.spaces.set import Set
from gym.spaces.space import Space
from gym.utils import seeding

SAMPLE_MASK_TYPE = Tuple[Union["SAMPLE_MASK_TYPE", np.ndarray], ...]


class MultiSet(Space[np.ndarray]):
    """This represents the cartesian product of arbitrary :class:`Set` spaces.

    It is useful to represent game controllers or keyboards where each key can be represented as a set action space.

    It can be initialized as ``MultiSet([ 5, 2, 2 ])`` such that a sample might be ``array([2, 5, 2])``.
    The given set stands for choice space(currently the same for each place).

    """

    def __init__(
        self,
        nvec: Union[np.ndarray, List[int]],
        n: int,
        dtype=np.int64,
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
    ):
        """Constructor of :class:`MultiSet` space.

        The argument ``nvec`` will determine the number of values each categorical variable can take.

        Args:
            nvec: vector of set range of each categorical variable. This will usually be a list of integers. 
            n: number of elements for each vector to contain.
            dtype: This should be some kind of integer type.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
        """
        self.nvec = np.array(nvec, dtype=dtype, copy=True)
        self.n = n
        assert (self.nvec >= 0).all(), "nvec have to be positive"

        super().__init__(self.nvec.shape, dtype, seed)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than :class:`gym.Space` - never None."""
        return self._shape  # type: ignore

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def sample(self, mask: Optional[SAMPLE_MASK_TYPE] = None) -> np.ndarray:
        # TODO mask case not fixed yet
        """Generates a single random sample this space.

        Args:
            mask: An optional mask for multi-set, expects tuples with a `np.ndarray` mask in the position of each
                action with shape `(n,)` where `n` is the number of actions and `dtype=np.int8`.
                Only mask values == 1 are possible to sample unless all mask values for an action are 0 then the default action 0 is sampled.

        Returns:
            An `np.ndarray` of shape `space.shape`
        """
        if mask is not None:

            def _apply_mask(
                sub_mask: SAMPLE_MASK_TYPE, sub_nvec: np.ndarray
            ) -> Union[int, List[int]]:
                if isinstance(sub_mask, np.ndarray):
                    assert np.issubdtype(
                        type(sub_nvec), np.integer
                    ), f"Expects the mask to be for an action, actual for {sub_nvec}"
                    assert (
                        len(sub_mask) == sub_nvec
                    ), f"Expects the mask length to be equal to the number of actions, mask length: {len(sub_mask)}, nvec length: {sub_nvec}"
                    assert (
                        sub_mask.dtype == np.int8
                    ), f"Expects the mask dtype to be np.int8, actual dtype: {sub_mask.dtype}"

                    valid_action_mask = sub_mask == 1
                    assert np.all(
                        np.logical_or(sub_mask == 0, valid_action_mask)
                    ), f"Expects all masks values to 0 or 1, actual values: {sub_mask}"

                    if np.any(valid_action_mask):
                        return self.nvec[self.np_random.choice(np.where(valid_action_mask)[0])]
                    else:
                        return self.nvec[0]
                else:
                    assert isinstance(
                        sub_mask, tuple
                    ), f"Expects the mask to be a tuple or np.ndarray, actual type: {type(sub_mask)}"
                    assert len(sub_mask) == len(
                        sub_nvec
                    ), f"Expects the mask length to be equal to the number of actions, mask length: {len(sub_mask)}, nvec length: {len(sub_nvec)}"
                    return [
                        _apply_mask(new_mask, new_nvec)
                        for new_mask, new_nvec in zip(sub_mask, sub_nvec)
                    ]

            return np.array(_apply_mask(mask, self.nvec), dtype=self.dtype)

        return np.array([self.nvec[self.np_random.random(self.nvec.shape) * self.nvec].astype(self.dtype) for i in range(self.n)])

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check
        return bool(x.shape == self.shape and (0 <= x).all() and (x in self.nvec).all())

    def to_jsonable(self, sample_n: Iterable[np.ndarray]):
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        return np.array(sample_n)

    def __repr__(self):
        """Gives a string representation of this space."""
        return f"MultiSet({self.nvec})"

    def __getitem__(self, index):
        """Extract a subspace from this ``MultiSet`` space."""
        nvec = self.nvec[index]
        if nvec.ndim == 0:
            subspace = Set(nvec)
        else:
            subspace = MultiSet(nvec, self.dtype)  # type: ignore
        subspace.np_random.bit_generator.state = self.np_random.bit_generator.state
        return subspace

    def __len__(self):
        """Gives the ``len`` of samples from this space."""
        if self.nvec.ndim >= 2:
            logger.warn("Get length of a multi-dimensional MultiSet space.")
        return len(self.nvec)

    def __eq__(self, other):
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, MultiSet) and np.all(self.nvec == other.nvec)
