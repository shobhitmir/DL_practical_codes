import itertools
import numpy as np


def predict(X, y, weights, threshold):
    net = np.dot(X,weights)
    y_pred = (net >= threshold).astype(int).flatten()

    print('Net : ', net.flatten())
    print('Weights : ',weights.flatten())
    print('Threshold : ',threshold)
    print('Actual Values : ', y)
    print('Predicted Values : ',y_pred)


# 1) NOT GATE
X = np.array([[0], [1]])
y = np.array([1, 0])
w = np.array([-1]).reshape((1,1))
T = 0
print('---- NOT GATE ----')
predict(X,y,w,T)
print()


n = int(input('Enter number of bits : '))
X = np.array([list(i) for i in itertools.product([0, 1], repeat=n)])
print()

# 2) AND GATE
y = np.array([0]*(2**n))
y[-1] = 1
w = np.array([1]*(n)).reshape((n,1))
T = n
print('---- AND GATE ----')
predict(X,y,w,T)
print()


# 3) OR GATE
y = np.array([1]*(2**n))
y[0] = 0
w = np.array([1]*(n)).reshape((n,1))
T = 1
print('---- OR GATE ----')
predict(X,y,w,T)
print()


# 4) NAND GATE
y = np.array([1]*(2**n))
y[-1] = 0
w = np.array([-1]*(n)).reshape((n,1))
T = -n+1
print('---- NAND GATE ----')
predict(X,y,w,T)
print()


# 5) NOR GATE
y = np.array([0]*(2**n))
y[0] = 1
w = np.array([-1]*(n)).reshape((n,1))
T = 0
print('---- NOR GATE ----')
predict(X,y,w,T)
print()
