from AutoGrad import tensor

A = tensor.Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]], requires_grad=True)

x = tensor.Tensor([4, 5, 1], requires_grad=True)
y = tensor.Tensor([5, 2, 3], requires_grad=True)

z = 3 * x + 4 * y

z.sum()

z.backward()
