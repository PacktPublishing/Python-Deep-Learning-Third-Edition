import numpy as np

q = np.array([0.6, 1.2, -1.2, 1.8])

k = np.array([
    [-0.2, 0.4, 1.2, 0.8],
    [0.2, 0.4, -0.6, 0.6],
    [0.6, -0.4, 1.4, 0.8],
    [1.6, 0.2, 1, 0.2]
])

v = np.array([
    [4, 5, 6, 7],
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [6, 7, 8, 9]
])


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


e = [np.dot(q, k[0]), np.dot(q, k[1]), np.dot(q, k[2]), np.dot(q, k[3])]
print(e)
a = softmax(e)
print(a)
result = [0, 0, 0, 0]
for i in range(len(v)):
    result += a[i] * v[i]

print(result)
