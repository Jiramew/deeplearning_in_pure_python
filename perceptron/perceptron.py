import numpy as np

data = np.array([[3., 3.], [4., 3.], [2., 2.], [1., 1.], [0., 0.]])
label = np.array([1., 1., 1., -1., -1.])

b = -0.3
w = np.array([0.1, 0.1])

stop = 0

alpha = 0.1

while True:
    for i in range(len(data)):
        if label[i] * (np.dot(data[i], w) + b) <= 0:
            w += alpha * label[i] * data[i]
            b += alpha * label[i]
            stop = 0
            break  # sgd
        else:
            stop = 1
    if stop == 1:
        break

print(w)
print(b)


######
# x{i} for i in range(1,n)
# y{i} for i in range(1,n), y{i} in {-1,1}
# loss: min L(w,b) = - sum(y{i}(w*x{i} + b)) for w,b
# lagrange for w,b
# # w -sum(y{i}*x{i})
# # b -sum(y{i})
# update for w,b
# # w = w - alpha(-sum(y{i}*x{i})) # opposite direction of gradient
# # b = b - alpha(-sum(y{i}))
