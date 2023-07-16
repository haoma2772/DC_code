import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import sys
import torch.nn as nn
from collections import OrderedDict
def show_relu():
    x = torch.arange(-8, 8, 0.1, requires_grad=True)
    y = x.relu()
    # 在 PyTorch 中，如果一个张量需要梯度，
    # 即设置了 requires_grad=True，那么就不能直接使用 numpy()
    # 方法将其转换为 NumPy 数组，因为梯度的计算可能会受到影响。
    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.title('Relu')
    plt.show()


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


if __name__ == '__main__':
    file_name = the_path = '../Datasets/FashionMNIST'
    # 首先是 获得数据
    train_dataset = torchvision.datasets.FashionMNIST(root=file_name,
                                                    train=True, download=True,
                                                    transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.FashionMNIST(root=file_name,
                                                   train=False, download=True,
                                                   transform=torchvision.transforms.ToTensor())

    # 这才是加载数据
    batch_size = 256
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_inputs = 28 * 28
    num_outputs = 10
    num_hiddens = 256
    # 定义网络结构 model
    drop_prob = 0.01
    my_net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        # 丢弃层
        nn.Dropout(drop_prob),
        nn.Linear(num_hiddens, num_outputs)
    )
    my_net.train()
    for params in my_net.parameters():
        nn.init.normal_(params, mean=0, std=0.01)

    # loss
    loss = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(
        my_net.parameters(),
        lr=0.01
    )

    # 开始训练
    num_epochs = 50
    for it in range(num_epochs):
        total_correct = 0
        total_example = 0
        for x, y in train_iter:
            output = my_net(x)
            the_loss = loss(output, y)
            optimizer.zero_grad()
            the_loss.backward()
            optimizer.step()
            # 因为输出时 (batch, outputs)
            predictions = torch.argmax(output, dim=1)
            # Count the number of correct predictions and the total number of samples in the current batch.
            total_correct += (predictions == y).sum().item()
            total_example += y.size(0)

        acc = total_correct / total_example * 100
        print(acc)



    # 测试集 试一下
    my_net.eval()
    total_correct = 0
    total_example = 0
    for x,y in test_iter:
        output = my_net(x)
        predictions = torch.argmax(output,dim=1)
        total_correct += (predictions == y).sum().item()
        # 每个y 第一维度是batch的数量
        total_example += y.size(0)

    acc = total_correct / total_example * 100
    print('在测试集上的效果 %d' % acc)
