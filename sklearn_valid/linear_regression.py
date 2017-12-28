from sklearn import linear_model
import numpy as np

lir = linear_model.LinearRegression()
data = np.array([[0, 0, 1],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 0, 1],
                 [1, 2, 1],
                 [2, 3, 4],
                 [0, 0, 1],
                 [1, 2, 3],
                 [1, 2, 4]])

label = np.array([[0],
                  [2],
                  [2],
                  [3],
                  [1],
                  [4],
                  [9],
                  [1],
                  [6],
                  [7]])

lir.fit(data, label)

print(lir.coef_)
