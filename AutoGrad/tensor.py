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


class Parents(NamedTuple):
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
        parents: List[Parents] = None,
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
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: "Tensor" = None) -> None:
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

        Args:
            other (Tensor): The Tensor object to be added to the current Tensor object.

        Returns:
            Tensor: A new Tensor object with the sum of the data from the current Tensor
            object.
        """
        return NotImplementedError


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
            return grad * np.ones_like(tensor.data)

        parents = [Parents(tensor, grad_fn)]

    else:
        parents = None

    return Tensor(data, requires_grad, parents=parents)
