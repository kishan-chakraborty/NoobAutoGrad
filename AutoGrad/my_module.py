from typing import Callable, List, NamedTuple, Union, Optional

import numpy as np

# To convert any object which is arrayable into a numpy array.
Arrayable = Union[int, float, list, np.ndarray]

def ensure_array(obj: Arrayable) -> np.ndarray:
    """
    Converts any object which is Arrayable into a numpy array.
    """
    if isinstance(obj, np.ndarray):
        return obj
    return np.array(obj)

Tensorable = Union[int, float, "Tensor", np.ndarray]

def ensure_tensor(obj: "Tensor") -> "Tensor":
    """Make sure that the input object is tensor or can be converted to one."""
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)


class Parent(NamedTuple):
    """
    Object representing the parent corresponding to a Tensor.
    tensor: Tensor object
    grad_fn: Gradient wrt the tensor correspondig to the operation.
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
        self.data = ensure_array(data)  
        self.requires_grad = requires_grad
        self.parents = parents or []
        self.shape = self.data.shape
        self.grad: Optional["Tensor"] = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        """Gives a string representation of an object"""
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        """Initialize the gradient value to zero (default)"""
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def sum(self) -> "Tensor":
        """
        Takes a tensor as input and returns a 0-Tensor which is the sum of all its
        elements.
        """
        return tensor_sum(self)

    def __add__(self, other) -> "Tensor":
        """
        Adds the data of the current Tensor object with the data of another Tensor
        object. Support the + symbol for addition. self + other.

        Allowed broadcasting:
            (m, n) + (m, n) -> (m, n)
            (m, n) + (1, n) or (1, n) + (m, n) -> (m, n) [Row addition]

        Args:
            other (Tensor): The Tensor object to be added to the current Tensor object.

        Returns:
            Tensor: A new Tensor object with the sum of the data from the current Tensor
            object.
        """
        return add(self, ensure_tensor(other))

    def __radd__(self, other) -> "Tensor":
        """
        Exactly same as add but it is other + self.
        """
        return add(ensure_tensor(other), self)

    def __iadd__(self, other) -> "Tensor":
        """
        Implementing += other
        """
        self.data = self.data + ensure_tensor(other).data
        self.grad = None
        return self

    def __sub__(self, other) -> "Tensor":
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
        return sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> "Tensor":
        """Implementing - self"""
        return sub(ensure_tensor(other), self)

    def __isub__(self, other) -> "Tensor":
        """Implementing -="""
        self.data = self.data - ensure_tensor(other).data
        self.grad = None

        return self

    def __mul__(self, other) -> "Tensor":
        """
        Element wise multiplication of the data of two tensor objects.
        Support the * symbol for multiplication.

        Args:
            other (Tensor): The Tensor object to be multiplied with the current Tensor object.

        Returns:
            Tensor: A new Tensor object obtained by element wise multiplication.
        """

        return mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> "Tensor":
        """Same as mul, other * self"""
        return mul(ensure_tensor(other), self)

    def __imul__(self, other) -> "Tensor":
        """Same as mul, other self *= """
        self.data = self.data * ensure_tensor(other).data
        self.grad = None

        return self

    def __neg__(self) -> "Tensor":
        """ Implementing -self """
        return neg(self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return matmul(self, other)

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

        self.grad.data = self.grad.data + grad.data

        for parent in self.parents:
            backward_grad = parent.grad_fn(grad.data)
            parent.tensor.backward(Tensor(backward_grad))


def add(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Adding two tensors and return the result.

    Args:
        tensor1: The first tensor
        tensor2: The second tensor

    Returns:
        tensor1 + tensor2
    """
    requires_grad = tensor1.requires_grad or tensor2.requires_grad
    data = tensor1.data + tensor2.data

    parents: List[Parent] = []

    if tensor1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:

            n_dims_added = grad.ndim - tensor1.data.ndim

            for _ in range(n_dims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(tensor1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        parents.append(Parent(tensor1, grad_fn1))

    if tensor2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:

            n_dims_added = grad.ndim - tensor2.data.ndim

            for _ in range(n_dims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(tensor2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        parents.append(Parent(tensor2, grad_fn2))

    return Tensor(data, requires_grad, parents=parents)


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
    return tensor1 + neg(tensor2)


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
            grad = grad * tensor2.data

            n_dims_added = grad.ndim - tensor1.data.ndim
            for _ in range(n_dims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but not added dims)
            for i, dim in enumerate(tensor1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        parents.append(Parent(tensor1, grad_fn1))

    if tensor2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Both the tensor has same shape.
            grad = grad * tensor1.data

            n_dims_added = grad.ndim - tensor2.data.ndim
            for _ in range(n_dims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but not added dims)
            for i, dim in enumerate(tensor2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

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


def matmul(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    data = tensor1.data @ tensor2.data
    requires_grad = tensor1.requires_grad or tensor2.requires_grad

    parents: List[Parent] = []

    if tensor1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ tensor2.data.T

        parents.append(Parent(tensor1, grad_fn1))

    if tensor2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return tensor1.data.T @ grad

        parents.append(Parent(tensor2, grad_fn2))

    return Tensor(data, requires_grad, parents)
