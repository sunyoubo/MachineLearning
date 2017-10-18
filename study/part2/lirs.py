
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from study.part2.perceptron import Perceptron


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# print(df.tail())
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# plt.xlabel('petal length')
# plt.ylabel('sepal length')
# plt.legend(loc='upper left')
# plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epoches')
plt.ylabel('Number of misclassifications')
plt.show()

print(ppn.predict(np.ndarray([5, 2])))
print(ppn.predict(np.ndarray([4.7, 1.5], dtype=float)))
print(ppn.predict(np.ndarray([4.8, 1.5])))
print(ppn.predict(np.ndarray([4.7, 1.5])))
print(ppn.predict(np.ndarray([4.9, 1.3])))
