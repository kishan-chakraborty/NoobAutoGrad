import unittest

from AutoGrad.my_module import Tensor

class TestTensorSum(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 + t2
        t3.backward(Tensor([-1., -2., -3.]))

        assert t1.grad.data.tolist() == [-1., -2., -3.]
        assert t2.grad.data.tolist() == [-1., -2., -3.]

    def test_broadcast_add(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)       # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad = True)                  # (3,)

        t3 = t1 + t2   # shape (2, 3)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [2, 2, 2]

    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)    # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad = True)               # (1, 3)

        t3 = t2 + t1
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [[2, 2, 2]]

if __name__ == "__main__":
    unittest.main()
