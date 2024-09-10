import pickle as pkl
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
targets = pkl.load(open("/home/zhx/Project/RheoMars_stable/fluidlab/assets/targets/Pouring.pkl", 'rb'))
b = 99
# 创建图和轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 归一化直方图数据
normalized_values = targets['last_grid'][b].flatten() / targets['last_grid'][b].max()

# 绘制直方图
xpos, ypos, zpos = np.indices((64, 64, 64)).reshape(3, -1)
dx = dy = dz = np.ones_like(zpos)

# 使用 colormap
cmap = plt.get_cmap('viridis')
colors = cmap(normalized_values)  # 将归一化的值映射到颜色

mask = normalized_values > 0  # 只绘制非零的格点
ax.bar3d(xpos[mask], ypos[mask], zpos[mask], dx[mask], dy[mask], dz[mask], color=colors[mask], shade=True)

# 绘制空间点
ax.scatter(targets['last_pos'][b][:, 0] * 64, targets['last_pos'][b][:, 1] * 64, targets['last_pos'][b][:, 2] * 64, c='red', marker='o')

# 设置标题和轴标签
ax.set_title('3D Histogram with Scatter Points')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

ax.set_xlim(0, 63)
ax.set_ylim(0, 63)
ax.set_zlim(0, 63)
# 显示图形
plt.show()