import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
# 假设 CSV 文件中的数据列名为 'X1', 'X2', ..., 'X6' 和 'Y1', 'Y2'
df = pd.read_csv(r'C:\Users\Hans\Desktop\NUS课件\GP-soft.csv')
X = df[['x1', 'x2']].values
y = df[['y1','y2']].values

# 划分训练集和测试集
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=32)

# Create a Gaussian Process Regressor with RBF kernel for each output
kernels = [C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3) for _ in range(train_y.shape[1])]
gprs = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42) for kernel in kernels]

# Fit each GPR model to the training data for each output
for i in range(train_y.shape[1]):
 gprs[i].fit(train_X, train_y[:, i])

# Predict on the test data for each output
test_y_pred = np.column_stack([gpr.predict(test_X) for gpr in gprs])

# Evaluate the model on the test set
mse = mean_squared_error(test_y, test_y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

# Plot the true and predicted values for each output in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the true values
ax.scatter(test_X[:, 0], test_X[:, 1], test_y[:, 0], label='True Output 1', c='b')
ax.scatter(test_X[:, 0], test_X[:, 1], test_y[:, 1], label='True Output 2', c='g')

# Plot the predicted values
ax.scatter(test_X[:, 0], test_X[:, 1], test_y_pred[:, 0], label='Predicted Output 1', c='r', marker='x')
ax.scatter(test_X[:, 0], test_X[:, 1], test_y_pred[:, 1], label='Predicted Output 2', c='m', marker='x')

ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')
ax.set_title('Gaussian Process Regression with inverse kinematic')
ax.legend()

plt.show()