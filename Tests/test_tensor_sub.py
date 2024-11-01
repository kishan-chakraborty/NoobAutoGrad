import unittest

from AutoGrad.tensor import Tensor


class TestTensorSub(unittest.TestCase):
    def test_simple_sub(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 - t2
        t3.backward(Tensor([-1.0, -2.0, -3.0]))

        assert t1.grad.data.tolist() == [-1.0, -2.0, -3.0]
        assert t2.grad.data.tolist() == [1.0, 2.0, 3.0]

    def test_broadcast_sub(self):
        # What is broadcasting? A couple of things:
        # If I do t1 + t2 and t1.shape == t2.shape, it's obvious what to do.
        # but I'm also allowed to add 1s to the beginning of either shape.
        #
        # t1.shape == (10, 5), t2.shape == (5,) => t1 + t2, t2 viewed as (1, 5)
        # t2 = [1, 2, 3, 4, 5] => view t2 as [[1, 2, 3, 4, 5]]
        #
        # The second thing I can do, is that if one tensor has a 1 in some dimension,
        # I can expand it
        # t1 as (10, 5) t2 as (1, 5) is [[1, 2, 3, 4, 5]]

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)  # (3,)
        t3 = Tensor([[10, 11, 12]], requires_grad=True)  # (3,)

        t4 = t1 - t2  # shape (2, 3)
        t5 = t4 - t3
        t5.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [[-2, -2, -2]]
        assert t3.grad.data.tolist() == [[-2, -2, -2]]

    def test_broadcast_sub2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)  # (1, 3)

        t3 = t2 - t1
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[-1, -1, -1], [-1, -1, -1]]
        assert t2.grad.data.tolist() == [[2, 2, 2]]


if __name__ == "__main__":
    unittest.main()
