from AutoGrad.my_module import Tensor
import unittest

A = Tensor([[1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]], requires_grad=True)

x = Tensor([[2], [4], [8]], requires_grad=True)
y = Tensor([[1], [8], [0]], requires_grad=True)

t1 = A @ x + 3
t2 = 2 * x + 3 * y

t3 = t1 + t2
t4 = t3.sum()

t4.backward()

assert t3.grad.data.tolist(), [[1], [1], [1]]