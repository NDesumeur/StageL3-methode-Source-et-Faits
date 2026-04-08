import numpy as np
import os

a = np.array([1, 2, 3, 4, 5])
b = np.zeros((2, 3))
c = np.ones((2, 3))
d = np.arange(0, 10, 2)
e = np.linspace(0, 1, 5)

print(a.shape)
print(b.size)
print(a.dtype)

f = np.arange(6).reshape(2, 3)
g = np.concatenate((b, c), axis=0)
h = np.vstack((b, c))
i = np.hstack((b, c))
j = np.column_stack((a, a))

print(np.sum(a))
print(np.mean(a))
print(np.std(a))
print(np.min(a))
print(np.max(a))
print(np.argmin(a))
print(np.argmax(a))

k = np.where(a > 3, 1, 0)
l = np.unique(np.array([1, 1, 2, 2, 3, 3, 3]))

m = np.array([[1, 2], [3, 4]])
n = np.array([[5, 6], [7, 8]])
o = np.dot(m, n)
p = f.T

q = np.array([3, 1, 4, 1, 5, 9, 2])
print(np.sort(q))
print(np.argsort(q))
print(np.bincount([0, 1, 1, 2, 2, 2]))

np.save("temp_array.npy", a)
r = np.load("temp_array.npy")
os.remove("temp_array.npy")

print(np.any(a > 4))
print(np.all(a > 0))
s = a.copy()
t = np.expand_dims(a, axis=1)
u = np.squeeze(t)

np.random.seed(42)
v = np.random.rand(2, 2)
w = np.random.randn(2, 2)
x = np.random.randint(0, 10, (2, 2))
y = np.random.choice(a, 3, replace=False)

z1 = np.exp(a)
z2 = np.log(a)
z3 = np.sqrt(a)

nan_arr = np.array([1.0, 2.0, np.nan, 4.0])
print(np.isnan(nan_arr))
print(np.isin(a, [2, 4, 6]))
