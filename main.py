import time
from AutoGrad.my_module import Tensor

A = Tensor([[1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]], requires_grad=True)

x = Tensor([[2], [4], [8]], requires_grad=True)
y = Tensor([[1], [8], [0]], requires_grad=True)

start_time = time.time()
t1 = A @ x + 3
t2 = 2 * x + 3 * y

t3 = t1 + t2
t4 = t3.sum()

t4.backward()
end_time = time.time()

print('Execution time: ', end_time-start_time)
print(x.grad.data.tolist())
print(y.grad.data.tolist())