from AutoGrad.tensor import Tensor


def min_sum_square(x: Tensor):
    """
    For a given values of x, do a gradient descent to find the point which minimize the
    sum of squares of the given points.

    Args:
        x: Tensor of points
    """

    for _ in range(100):
        # Find sum of square for the given points
        x2 = x * x
        x2_sum = x2.sum()

        # Find the gradient of the sum of squares
        x2_sum.backward()

        # Update the points
        del_x = 0.1 * x.grad.data

        x = Tensor(x.data - del_x, requires_grad=True)

        print(x2_sum.data)


if __name__ == "__main__":
    x = Tensor([10, -10, 3, 5, 7.0 - 6, -4], requires_grad=True)
    min_sum_square(x)
