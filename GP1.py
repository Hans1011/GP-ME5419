import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 从 CSV 文件加载数据
# 假设 CSV 文件中的数据列名为 'X1', 'X2', ..., 'X6' 和 'Y1', 'Y2'
df = pd.read_csv(r'C:\Users\Hans\Desktop\NUS课件\GP.csv')
X = df[['x1', 'x2','x3']].values
y = df[['y1','y2']].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=32)
# 标准化和归一化

# 高斯过程拟合
kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
gpr.fit(X_train, y_train)

# 预测
y_pred = gpr.predict(X_test)

# 评估性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')
# 逆标准化


# 绘图
plt.figure()

# 绘制 y1 的散点图
plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0], y_pred[:, 0])
plt.xlabel('True Values (pitch)')
plt.ylabel('Predictions (pitch)')
plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], '--', color='red', label='Ideal line')
plt.title('True Values vs. Predictions (pitch)')

# 绘制 y2 的散点图
plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 1], y_pred[:, 1])
plt.xlabel('True Values (yaw)')
plt.ylabel('Predictions (yaw)')
plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], '--', color='red', label='Ideal line')
plt.title('True Values vs. Predictions (yaw)')

plt.tight_layout()
plt.show()


