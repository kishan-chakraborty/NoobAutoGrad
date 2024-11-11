import unittest

from AutoGrad.my_module import Tensor


class TestTensorSub(unittest.TestCase):
    
    def test_simple_ops(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = 3-t1
        t2.backward(Tensor([-1.0, -2.0, -3.0]))

        # t3 = 4*t1
        # t3.backward(Tensor([1, 2, 8]))

        assert t1.grad.data.tolist() == [1.0, 2.0, 3.0]
        # assert t3.grad.data.tolist() == [4, 8, 32]

    @unittest.skip
    def test_broadcast_sub(self):

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad=True)  # (3,)
        t3 = Tensor([[10, 11, 12]], requires_grad=True)  # (3,)

        t4 = [1, 4, 10] - t2  # shape (2, 3)
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
