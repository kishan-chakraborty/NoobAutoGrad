from typing import Callable, List, NamedTuple, Union

import numpy as np

# To convert any object which is arrayable into a numpy array.
Arrayable = Union[list, np.ndarray]


def to_array(obj: Arrayable) -> np.ndarray:
    """
    Converts any object which is Arrayable into a numpy array.
    """
    if isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)


class Parent(NamedTuple):
    """
    Object representing the parent corresponding to a Tensor.
    """

    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    """
    Tensor module for autodifferentiation.
    This is a wrapper over numpy arrays.
    Tracks the previous algebraic operations performed to obtain the tensor.

    Args:
        data (numpy.ndarray): The data to be stored in the tensor.
        requires_grad (bool, optional): Whether the tensor requires gradient.
        Defaults to False.

    """

    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        parents: List[Parent] = None,
    ) -> None:
        self.data = to_array(data)
        self.requires_grad = requires_grad
        self.parents = parents or []
        self.shape = self.data.shape
        self.grad: Tensor = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        """Gives a string representation of an object"""
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        """
        Initialize the gradient value to zero (default)"""
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def backward(self, grad: "Tensor" = None) -> None:
        """
        Backward automatic gradient calculator.

        Args:
            grad: Incoming gradient.
        """
        assert self.requires_grad, "cannot backward through a non-requires-grad tensor"
        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data += grad.data

        for parent in self.parents:
            backward_grad = parent.grad_fn(grad.data)
            parent.tensor.backward(Tensor(backward_grad))

    def sum(self) -> "Tensor":
        """
        Takes a tensor as input and returns a 0-Tensor which is the sum of all its
        elements.
        """
        return tensor_sum(self)

    def __add__(self, other: "Tensor") -> "Tensor":
        """
        Adds the data of the current Tensor object with the data of another Tensor
        object. Support the + symbol for addition.

        Allowed broadcasting:
            (m, n) + (m, n) -> (m, n)
            (m, n) + (1, n) or (1, n) + (m, n) -> (m, n) [Row addition]

        Args:
            other (Tensor): The Tensor object to be added to the current Tensor object.

        Returns:
            Tensor: A new Tensor object with the sum of the data from the current Tensor
            object.
        """
        # If both the vectors are 1 dimensional arrays.
        if self.data.ndim == 1 and other.data.ndim == 1:
            return add(self, other)

        # If one of the array is > one dimensional then the other must have the same dimension.
        if self.shape[1] != other.shape[1]:
            raise ValueError("Invalid dimension, operation not allowed.")

        return add(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        """
        Subtract the data of the other Tensor object from the data of current Tensor
        object. Support the - symbol for addition.

        Allowed broadcasting:
            (m, n) - (m, n) -> (m, n)
            (m, n) - (1, n) or (1, n) - (m, n) -> (m, n) [Row addition]

        Args:
            other (Tensor): The Tensor object to be added to the current Tensor object.

        Returns:
            Tensor: A new Tensor object with the sum of the data from the current Tensor
            object.
        """
        # If both the vectors are 1 dimensional arrays.
        if self.data.ndim == 1 and other.data.ndim == 1:
            return sub(self, other)

        # If one of the array is > one dimensional then the other must have the same dimension.
        if self.shape[1] != other.shape[1]:
            raise ValueError("Invalid dimension, operation not allowed.")

        return sub(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        """
        Element wise multiplication of the data of two tensor objects.
        Support the * symbol for multiplication.

        Args:
            other (Tensor): The Tensor object to be multiplied with the current Tensor object.

        Returns:
            Tensor: A new Tensor object obtained by element wise multiplication.
        """
        if self.data.ndim != other.data.ndim:
            raise ValueError(
                f"This operation between dim {self.data.ndim} and {other.data.ndim} is not allowed."
            )

        return mul(self, other)


def add(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Adding two tensors and return the result.

    Args:
        tensor1: The first tensor
        tensor2: The second tensor

    Returns:
        tensor1 + tensor2
    """
    # Resultant has required grad (RG) = True if any of the tensor has RG True.
    requires_grad = tensor1.requires_grad or tensor2.requires_grad

    parents: List[Parent] = []

    if tensor1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            if grad.shape == tensor1.shape:
                return grad

            if grad.shape[0] > tensor1.shape[0]:
                return grad.sum(axis=0)

            return grad

        parents.append(Parent(tensor1, grad_fn1))

    if tensor2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            if grad.shape == tensor2.shape:
                return grad

            if grad.shape[0] > tensor2.shape[0]:
                return grad.sum(axis=0)

            return grad

        parents.append(Parent(tensor2, grad_fn2))

    return Tensor(tensor1.data + tensor2.data, requires_grad, parents=parents)


def neg(tensor: Tensor) -> Tensor:
    """Calculate negative of a tensor object.
    Args:
        tensor: Input tensor object.

    Return:
        tensor: Corresponding negative of the input.
        -1 * tensor
    """
    parents = []
    if tensor.requires_grad:
        parents.append(Parent(tensor, lambda x: -x))

    return Tensor(-tensor.data, tensor.requires_grad, parents=parents)


def sub(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Subtract tensor2 from tensor 1 and return the result.

    Args:
        tensor1: The first tensor
        tensor2: The second tensor

    Returns:
        tensor1 - tensor2
    """
    return add(tensor1, neg(tensor2))


def mul(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Element wise multiplication of two tensors and return the result.

    Args:
        tensor1: The first tensor
        tensor2: The second tensor

    Returns:
        tensor1 * tensor2
    """

    # Resultant has required grad (RG) = True if any of the tensor has RG True.
    requires_grad = tensor1.requires_grad or tensor2.requires_grad

    parents: List[Parent] = []

    if tensor1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Both the tensor has same shape.
            if tensor1.shape == tensor2.shape:
                return grad * tensor2.data

            # tensor1 is a row vector
            if tensor1.shape[0] < grad.shape[0]:
                temp = grad * tensor2.data
                return temp.sum(axis=0)

            # tensor1 is a row vector
            temp = np.array([tensor2.data[0].tolist() for _ in range(tensor1.shape[0])])
            return temp

        parents.append(Parent(tensor1, grad_fn1))

    if tensor2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Both the tensor has same shape.
            if tensor1.shape == tensor2.shape:
                return grad * tensor1.data

            # tensor2 is a row vector
            if tensor2.shape[0] < grad.shape[0]:
                temp = grad * tensor1.data
                return temp.sum(axis=0)

            # tensor1 is a row vector
            temp = np.array([tensor1.data[0].tolist() for _ in range(tensor2.shape[0])])
            return temp

        parents.append(Parent(tensor2, grad_fn2))

    return Tensor(tensor1.data * tensor2.data, requires_grad, parents=parents)


def tensor_sum(tensor: Tensor) -> Tensor:
    """
    Takes a tensor as input and returns a 0-Tensor which is the sum of all its elements.
    """

    data = tensor.data.sum()
    requires_grad = tensor.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a zero function so all elements contribute equally to
            its gradient.
            """
            return grad * np.ones_like(tensor.data, dtype=np.float64)

        parents = [Parent(tensor, grad_fn)]

    else:
        parents = None

    return Tensor(data, requires_grad, parents=parents)
