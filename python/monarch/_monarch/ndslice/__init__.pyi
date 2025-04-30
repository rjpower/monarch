from typing import final, Iterator, overload

@final
class Slice:
    """
    A wrapper around [ndslice::Slice] to expose it to python.
    It is a compact representation of indices into the flat
    representation of an n-dimensional array. Given an offset, sizes of
    each dimension, and strides for each dimension, Slice can compute
    indices into the flat array.

    Arguments:
    - `offset`: Offset into the flat array.
    - `sizes`: Sizes of each dimension.
    - `strides`: Strides for each dimension.
    """

    def __init__(
        self, *, offset: int, sizes: list[int], strides: list[int]
    ) -> None: ...
    @property
    def ndim(self) -> int:
        """The number of dimensions in the slice."""
        ...

    @property
    def offset(self) -> int:
        """
        The offset of the slice i.e. the first number from which
        values in the slice begin.
        """
        ...

    @property
    def sizes(self) -> list[int]:
        """The sizes of each dimension in the slice."""
        ...

    @property
    def strides(self) -> list[int]:
        """The strides of each dimension in the slice."""
        ...

    def index(self, value: int) -> int:
        """
        Returns the index of the given `value` in the slice or raises a
        `ValueError` if `value` is not in the slice.
        """
        ...

    def coordinates(self, value: int) -> list[int]:
        """
        Returns the coordinates of the given `value` in the slice or raises a
        `ValueError` if the `value` is not in the slice.
        """
        ...

    def nditem(self, coordinates: list[int]) -> int:
        """
        Returns the value at the given `coordinates` or raises an `IndexError`
        if the `coordinates` are out of bounds.
        """
        ...

    def __eq__(self, value: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __getnewargs_ex__(self) -> tuple[tuple, dict]: ...
    @overload
    def __getitem__(self, i: int) -> int: ...
    @overload
    def __getitem__(self, i: slice) -> tuple[int, ...]: ...
    def __len__(self) -> int:
        """Returns the complete size of the slice."""
        ...

    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over the values in the slice."""
        ...

    @staticmethod
    def from_list(ranks: list[int]) -> list["Slice"]:
        """Returns a list of slices that cover the given list of ranks."""
        ...
