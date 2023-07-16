import numpy as np
import torch


if __name__ == '__main__':
    # tensor的创建
    # 1. numpy
    print('1. numpy')
    device = 'cuda'
    # torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
    arr = np.ones((3, 3))
    print("ndarray的数据类型：", device,arr.dtype)
    # 创建存放在 GPU 的数据
    # t = torch.tensor(arr, device='cuda')
    q= torch.tensor(arr)
    print(q)
    # 2. torch.zeors()
    print('2. torch.zeors()')
    # torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    out_t = torch.tensor([1])
    # 这里制定了 out
    # 感觉是 主要关注 size size 可以是 (3)  (3,3)  (3,3,3) (3,3,3,3) 每个代表不同维度
    t = torch.zeros((3,3), out=out_t)
    print(t, '\n', out_t)
    # id 是取内存地址。最终 t 和 out_t 是同一个内存地址
    print(id(t), id(out_t), id(t) == id(out_t))
    # 3. torch.zeros_like(input, dtype=None,
    # layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
    print('3. torch.zeros_like()')
    t = torch.zeros_like(q)
    print(t)

    # 4. ：torch.ones()，torch.ones_like()  全 1的

    # 5. torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # size 与之前的一样  fill_value 就是矩阵里面的元素

    # 6. torch.arange(start=0, end, step=1,
    # out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    #  [Start, end)  step 默认是1 创建一维的等差数列
    print('6. torch.arrange()')
    t = torch.arange(1,10,2)
    print(t)

    # 7. torch.linspace(start, end, steps=100, out=None, dtype=None,
    # layout=torch.strided, device=None, requires_grad=False)
    # 创建均分的一维向量 [start,end) step是有多少个元素
    print('7. torch.linspace()')
    t = torch.linspace(1, 10, 100)
    print(t)

    # 8. torch.logspace(start, end, steps=100, base=10.0,
    # out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # [Start, end) 创建对数均分 step仍然是元素个数 对数的底数是base 默认10
    print('8. torch.logspace()')
    t = torch.logspace(1,10,100)
    print(t)

    # 9. torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # 创建单位对角矩阵 2维 默认方阵 只设置n即可
    print('9. torch.eye()')
    t = torch.eye(2)
    print(t)

    # 依据 概率来生成 tensor

    # 10. torch.normal(mean, std, *, generator=None, out=None)
    # 正态分布 mean是均值 std 方差
    # 当 mean std均为 标量是 需要加入size
    print('10. torch.normal()')
    t = torch.normal(0,10, size=(3,3))
    print(t)

    # 11. torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # 生成标准的正态分布

    # 12.torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    #  生成[0,1) 均匀分布
    print('12. torch.rand()')
    t = torch.rand((3,3))
    print(t)

    # 13. randint(low=0, high, size, *, generator=None, out=None,
    # dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # 生成[low,hight) 上的整数均匀分布

    print('13. torch.randint()')
    t = torch.randint(1,10,size=(3,3))
    print(t)

    # 14. torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
    # 生成0- n-1的随机排列 通常用于生成索引
    print('14. torch.randperm')
    t= torch.randperm(15)
    print(t)

    # 15.torch.bernoulli(input, *, generator=None, out=None)
    # input为概率值是一个tensor  生成伯努利分布 input输入必须是tensor
    print('15. torch.bernoulli')
    # 生成一个形状为 (2, 3) 的随机张量，其中的元素服从伯努利分布
    input = torch.tensor([[0.7, 0.4, 0.9], [0.2, 0.6, 0.3]])
    output = torch.bernoulli(input)
    print(output)

    # 16. item()函数
    print('16. item()函数')
    t = torch.randn(1)
    print(t.item())

    # 17. tensor 与 numpy的相互转化  但是仍然共享内存
    # 使用numpy 将 tensor 转化为 numpy
    # 使用 from_numpy 将 numpy转化为 tensor 这两个的内存是一样的   但是呢 如果我直接 torch.tensor() 内存是不一样的
    # 即使内存是共享的 但是id 还是不一样的 因为 id 唯一标识对象
    print('17. tensor 与 numpy转化')
    a = np.zeros((3,3))
    print(id(a))
    b = torch.from_numpy(a)
    print(id(b))

    # 18. tensor on gpu
    # tensor.to()
    print('18. tensor.to(device)')
    device = 'cuda'
    t = torch.rand((5,5),device=device)
    print(t)

    # 19. 自动求梯度
    # tensor 的属性 .requires.grad 设置为true
    #  他将追踪 在其上的所有操作 可以调用 .detach() 将追踪记录 分离出来
    # .backward()进行计算  然后将梯度积累到了.grad当中
    # 每个tensor 都有一个 grad_fn属性 即创建该tensor的 function
    # 只允许 标量对张量求导 所以最后得出的结果是一个张量 才能求导
    print('19. requires_grad grad backward grad_fn')
    t = torch.ones((2,2), requires_grad=True)
    print(t)
    # print(t.grad)   # 梯度
    # print(t.grad_fn)    # 函数吗？
    y = t + 3
    print(y)
    # print(y.grad)     没法现在就使用
    print(y.grad_fn)
    z = y * y + 20*y + 300
    print(z)
    z = z.mean()        # 计算的是所有元素的平均值
    print(z)
    z.backward()    # 计算梯度
    print(t.grad)















