
from mxnet import ndarray as nd


nd_1 = nd.zeros(3, 4)

print(nd_1)

x = nd.ones((3, 4))
print(x)

nd_2 = nd.array([1, 2], [3, 4])
print("nd_2:", nd_2)

y = nd.random_normal(0, 1, shape=(3, 4))
print("y:", y)
print("y.shape:", y.shape)
print("y.size:", y.size)

print("x+y:")
print(x+y)

print("x*y:")
print(x*y)

print("nd.exp(y):")
print(nd.exp(y))

print("nd.dot(x, y.T):")
print(nd.dot(x, y.T))


x = nd.ones((3, 4))
y = nd.ones((3, 4))
before_y = id(y)
y += x
print(id(y) == before_y)

z = nd.zeros_like(x)
before_z = id(z)
z[:] = x + y
print(id(z) == before_z)

nd.elemwise_add(x, y, out=z)
print(id(z) == before_z)