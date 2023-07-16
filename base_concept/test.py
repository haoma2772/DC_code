import torch
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

    # 使用plt.imshow()显示图像
    plt.imshow(image, cmap='gray')

    # 添加标题和标签
    plt.title('Image')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图像窗口
    plt.show()