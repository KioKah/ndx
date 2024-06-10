from typing import List, Union
from math import gcd as math_gcd


class CountArray:
    def __init__(self, data: List[int]):
        # Checks
        if not isinstance(data, List):
            raise TypeError("data must be a List")
        size = len(data)
        if size == 0:
            raise ValueError("data must not be empty")
        if not all(isinstance(value, int) for value in data):
            raise TypeError("data must be a List of ints")
        if not all(value >= 0 for value in data):
            raise ValueError("data values must be >= 0")
        # Init
        self.__data = data
        self.__size = size

    # Getters
    @property
    def data(self) -> List[int]:
        """Getter for `__data`"""
        return self.__data

    @property
    def size(self) -> int:
        """Getter for `__size`"""
        return self.__size

    # Static methods
    @staticmethod
    def zeros(length: int) -> List[int]:
        """Create a list of zeros.

        Args:
            `length` (int): Length of the list.

        Returns:
            List[int]: A list of zeros of the given length.
        """
        return CountArray([0] * length)

    @staticmethod
    def ones(length: int) -> List[int]:
        """Create a list of ones.

        Args:
            `length` (int): Length of the list.

        Returns:
            List[int]: A list of ones of the given length.
        """
        return CountArray([1] * length)

    @staticmethod
    def single(value: int) -> List[int]:
        """
        Create a list of a given value.

        Args:
            value (int): The value to fill the list with.

        Returns:
            List[int]: A list containing the given value.
        """
        return CountArray([value])

    # Array operations
    def reversed(self) -> "CountArray":
        return CountArray(self.__data[::-1])

    def total(self) -> int:
        """Total count of the count_array

        Returns:
            int: Sum of the values from the count_array
        """
        return sum(self.__data)

    def gdc(self) -> int:
        """Greatest common divisor of the count_array

        Returns:
            int: Greatest common divisor of the count_array
        """
        return math_gcd(*self.__data)

    # Accessors
    def __getitem__(self, index: Union[int, slice]) -> Union[int, "CountArray"]:
        if isinstance(index, slice):
            return CountArray(self.__data[index])
        if isinstance(index, int):
            return self.__data[index]
        raise TypeError("index must be an int or a slice")

    def __setitem__(self, index: Union[int, slice], value: Union[int, "CountArray"]):
        # Checks
        if not isinstance(value, (int, CountArray)):
            raise TypeError("value must be an int or a CountArray")
        # SetItem
        if isinstance(index, slice):
            if isinstance(value, CountArray):
                self.__data[index] = value.data
            else:
                self.__data[index] = value
        elif isinstance(index, int):
            self.__data[index] = value
        else:
            raise TypeError("index must be an int or a slice")

    # Math operations
    def __add__(self, other):
        if isinstance(other, CountArray):
            return CountArray(
                [x + y for x, y in zip(self.__data, other.data, strict=True)]
            )
        if isinstance(other, int):
            return CountArray([x + other for x in self.__data])
        raise TypeError("other must be a CountArray or an int")

    def __sub__(self, other):
        if isinstance(other, CountArray):
            return CountArray(
                [x - y for x, y in zip(self.__data, other.data, strict=True)]
            )
        if isinstance(other, int):
            return CountArray([x - other for x in self.__data])
        raise TypeError("other must be a CountArray or an int")

    def __mul__(self, other):
        if isinstance(other, CountArray):
            return CountArray(
                [x * y for x, y in zip(self.__data, other.data, strict=True)]
            )
        if isinstance(other, int):
            return CountArray([x * other for x in self.__data])
        raise TypeError("other must be a CountArray or an int")

    def __floordiv__(self, other):
        if isinstance(other, CountArray):
            return CountArray(
                [x // y for x, y in zip(self.__data, other.data, strict=True)]
            )
        if isinstance(other, int):
            return CountArray([x // other for x in self.__data])
        raise TypeError("other must be a CountArray or an int")

    # Comparisons
    def __eq__(self, other) -> bool:
        if not isinstance(other, CountArray):
            return False
        return self.__data == other.data

    def __ne__(self, other) -> bool:
        if not isinstance(other, CountArray):
            return True
        return self.__data != other.data

    # In-place operations
    def __iadd__(self, other: Union["CountArray", int]):
        if isinstance(other, CountArray):
            if self.__size != other.size:
                raise ValueError("CountArrays must have the same size")
            for i in range(self.__size):
                self.__data[i] += other[i]
        elif isinstance(other, int):
            for i in range(self.__size):
                self.__data[i] += other
        else:
            raise TypeError("other must be a CountArray or an int")
        return self

    def __isub__(self, other: Union["CountArray", int]):
        if isinstance(other, CountArray):
            for i in range(self.__size):
                if self.__size != other.size:
                    raise ValueError("CountArrays must have the same size")
                self.__data[i] -= other[i]
        elif isinstance(other, int):
            for i in range(self.__size):
                self.__data[i] -= other
        else:
            raise TypeError("other must be a CountArray or an int")
        return self

    def __imul__(self, other: Union["CountArray", int]):
        if isinstance(other, CountArray):
            for i in range(self.__size):
                if self.__size != other.size:
                    raise ValueError("CountArrays must have the same size")
                self.__data[i] *= other[i]
        elif isinstance(other, int):
            for i in range(self.__size):
                self.__data[i] *= other
        else:
            raise TypeError("other must be a CountArray or an int")
        return self

    def __ifloordiv__(self, other: Union["CountArray", int]):
        if isinstance(other, CountArray):
            for i in range(self.__size):
                if self.__size != other.size:
                    raise ValueError("CountArrays must have the same size")
                self.__data[i] //= other[i]
        elif isinstance(other, int):
            for i in range(self.__size):
                self.__data[i] //= other
        else:
            raise TypeError("other must be a CountArray or an int")
        return self

    # String representation
    def __str__(self):
        return f"[{' '.join([str(x) for x in self.__data])}]"

    def __repr__(self):
        return f"CountArray({repr(self.__data)})"
