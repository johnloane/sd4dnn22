import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def plot_decision_boundary(X, model):
    x_span = np.linspace(min(X[:, 0])-0.25, max(X[:, 0])+0.25, 50)
    y_span = np.linspace(min(X[:, 1])-0.25, max(X[:, 1])+0.25, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

np.random.seed(0)
n_pts = 500
X, labels = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

# plt.scatter(X[labels == 0, 0], X[labels == 0, 1])
# plt.scatter(X[labels == 1, 0], X[labels == 1, 1])
# plt.show()

model = Sequential()
# The hidden layer
model.add(Dense(4, input_shape=(2,), activation='sigmoid'))
# Dense means that every node is connected to the all nodes in the previous layer
# Output layer with a single node
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(learning_rate=0.01), 'binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=X, y=labels, verbose=1, batch_size=20, epochs=100, shuffle='true')
# plt.plot(h.history['accuracy'])
# plt.xlabel('epoch')
# plt.legend(['accuracy'])
# plt.title('accuracy')
# plt.show()


plot_decision_boundary(X, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
x1 = 0.1
x2 = 0
point = np.array([(x1, x2)])
prediction = model.predict(point)
plt.plot([x1], [x2], marker='x', markersize = 8, color='red')
print(prediction)

plt.show()