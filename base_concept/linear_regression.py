import matplotlib_inline
import tensorboard.compat.tensorflow_stub.dtypes
import torch
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim

# linear regression 适用于 输出为连续值的问题
# 比如predication  问题


class LinearNet(nn.Module):
    # 定义 线性回归的神经网络
    def __init__(self, n_feature):
        # 这个应该表示 输入特征的数量
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self,x):
        y = self.linear(x)
        return y


if __name__ == '__main__':

    # 12. linear_regression
    # 手写 从0开始
    print('12. linear_regression')
    num_inputs = 2
    examples = 1000
    true_w = [2, -3.4]          # 输入特征
    true_b = 4.2
    features = torch.normal(0, 1, (examples, num_inputs))    # 一共有1000行
    # 每一行 两个 feature 共有examples个例子

    features = torch.tensor(np.random.normal(0, 1, (examples, num_inputs)), dtype=torch.float64)

    true_labels = true_w[0] * features[:, 0] + true_w[1]*features[:, 0] + true_b
    # print(true_labels)
    labels = true_labels + torch.tensor(np.random.normal(0, 0.01, size=true_labels.size()), dtype=torch.float64)
    # 散点图
    # plt.scatter(features[:,1].numpy(), labels.numpy())
    # plt.show()

    # 读取数据
    batch_size = 10
    # 把特征和 labels结合 这就相当于是训练集
    data_set = Data.TensorDataset(features, labels)
    # 随机读取小批量 data_iter 有 examples / batch_size个
    data_iter = Data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    my_net = LinearNet(num_inputs)
    # 使用 nn.sequential()更好地搭建网络
    my_net = nn.Sequential(
        nn.Linear(num_inputs, 1).double()
        # 表示 输入 和 输出的个数
    )
    # torch.nn 仅支持batch的样本 输入 不支持单个样本 可以使用input
    # 查看网络的参数
    # for param in my_net.parameters():
    #     print(param)

    # 初始化 网络
    # init.normal 函数的第一个参数是要初始化的张量，即 my_net.weight，
    # 第二个参数 mean 是正态分布的均值，这里设为0，第三个参数 std 是正态分布的标准差，这里设为0.01。
    # 要对模型的权重进行初始化，你需要访问具体的层（如线性层）才能获取到权重参数
    nn.init.normal_(my_net[0].weight, mean=0, std=0.01)
    nn.init.constant_(my_net[0].bias, val=0)

    # 定义 loss
    loss = nn.MSELoss()
    # 定义优化器
    optimizer = optim.Adam(
        my_net.parameters(),
        lr=0.005,

    )

    # print(optimizer)
    # 训练模型
    epoches = 100
    for it in range(epoches):
        for x,y in data_iter:
            # x是features y是labels
            output = my_net(x)
            # y的维度不一致 output是[10,1]
            y = y.unsqueeze(1)

            l = loss(output, y)
            # 置为0 梯度

            optimizer.zero_grad()
            # backward 梯度计算
            l.backward()
            # step更新梯度参数
            optimizer.step()

        print(l.item())
    print(my_net[0].weight)
    print(my_net[0].bias)








