import torch
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import torch.nn as nn
from collections import OrderedDict
import time


# soft_max 用于离散问题的处理 输出有多个 用于分类的
# 全连接层（Fully Connected Layer），也称为线性层或密集层，
# 是神经网络中最常见的一种层类型。全连接层的每个神经元与上一层的所有神经元都有连接，它的输出由上一层的所有输入加权和经过激活函数得到。
def get_fashion_mnist_text(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker',
            'bag', 'ankle', 'boot']
    # 返回的是一个 labeles 的list标签
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    # 这是subplots
    figs, axs = plt.subplots(1, len(images),figsize=(12,12))
    # 遍历每个子图对象，设置其位置和大小

    # 在每个子图上绘制图形或进行其他操作
    for i, ax in enumerate(axs):
        ax.imshow(images[i].reshape(28, 28).numpy())
        ax.set_title(str(labels[i]))
        ax.axis('off')

    # 显示图形

    plt.show()


class FlattenLayer(nn.Module):
    # 形状变换
    def __int__(self):
        super(FlattenLayer, self).__int__()

    def forward(self, x):
        # 把形状变成了 第一个是通道数
        return x.view(x.shape[0], -1)


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        # x 是 (batch,1,28,28)
        # view 重新展开 -1表示自动计算
        # 因为输入要求是 (batch_size, 28*28=784)的
        # 现在已经预处理好了
        y = self.linear(x)
        return y

if __name__ == '__main__':

    # 获得数据
    the_path = '../Datasets/FashionMNIST'
    # 这里是通过 train 等于 true 和 false 得到测试集和训练集
    train_dataset = torchvision.datasets.FashionMNIST(root=the_path, train=True,
                                                      download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.FashionMNIST(root=the_path, train=False,
                                                     download=True, transform=transforms.ToTensor())
    print(len(train_dataset))
    print(len(test_dataset))
    feature, label = train_dataset[0]
    # 第一维是通道数 后面是 高和宽  灰色图像 通道只有1
    print(feature.shape, label)

    # 读取数据
    batch_size = 256
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # model
    num_inputs = 28 * 28
    out_puts = 10
    my_net = nn.Sequential(
        # 看好怎么写
        OrderedDict(
            [('FlattenLayer', FlattenLayer()),
             ('Linear', nn.Linear(num_inputs,out_puts))])
    )

    # 初始化
    # 第一个参数肯定是你要初始化的东西呢
    nn.init.normal_(my_net.Linear.weight, mean=0, std=0.01)
    nn.init.constant_(my_net.Linear.bias, val=0)

    # loss
    loss = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(
        # 网络的 参数 和 lr
        my_net.parameters(),
        lr=0.001
    )

    # 开始训练
    num_epoches = 20
    for it in range(num_epoches):
        total_correct = 0
        total_samples = 0
        for x, y in train_iter:
            output = my_net(x)
            l = loss(output, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            predictions = torch.argmax(output, dim=1)
            # Count the number of correct predictions and the total number of samples in the current batch.
            total_correct += (predictions == y).sum().item()
            total_samples += y.size(0)

        acc = total_correct / total_samples * 100
        print(acc)


