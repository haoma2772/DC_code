import torch
import numpy as np
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Linear(784,256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256,10)


    def forward(self, x):
        y = self.act(self.hidden(x))
        return self.output(y)


if __name__ == '__main__':

    x = torch.rand((2, 784))
    net = MLP()
    print(net)
    res = net(x)
    print(res)

    # 访问参数
    # for name, params in net.named_parameters():
    #     print(name, params.size())

    # 初始化参数
    for name, params in net.named_parameters():
        if 'weight' in name:
            # 这是用正态分布来初始化
            nn.init.normal_(params, mean=0, std=0.01)
            # print(params)
        if 'bias' in name:
            # 常数初始化
            nn.init.constant_(params, val=0)

    # 保存 和 读取 tensor
    # save 和 load 函数分别读取 tensor
    # 可以保存一个 也可以保存一个list
    x = torch.rand((3, 1))
    y = torch.rand((3,2))
    x = [x, y]
    print(x)
    torch.save(x, 'x.pt')
    x, y = torch.load('x.pt')
    print(y)

    # 保存和加载模型
    # 保存
    torch.save(net.state_dict(), 'model.pt')

    # 加载
    new_net = MLP()
    new_net.load_state_dict(torch.load('model.pt'))

    # GPU
    cuda_index = 0
    device = torch.device(
        'cuda:{}'.format(cuda_index) if torch.cuda.is_available() and cuda_index != -1 else 'cpu'
    )
    # 是不能在两台设备发生运算的
    x = x.to(device)
    print(x)
    x = x + 1
    print(x)






