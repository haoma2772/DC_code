import  torch


if __name__ == '__main__':

    # 1. torch.cat(tensors, dim=0, out=None)
    # torch.cat(tensors, dim=0, out=None)
    # tensors 是 需要操作的tensor list dim 是按照的维度 就是扩展哪一维吧
    print('1. torch.cat()')
    t = torch.ones((2, 3))
    t_0 = torch.cat([t, t], dim=0)
    t_1 = torch.cat([t, t], dim=1)
    print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

    # 2. torch.stack(tensors, dim=0, out=None)
    # 功能：将张量在新创建的 dim 维度上进行拼接
    print('2. torch.stack()')
    t = torch.ones((2, 3))
    print(t)
    # dim =2
    t_stack = torch.stack([t, t, t], dim=2)
    print("\nt_stack.shape:{}".format(t_stack.shape))
    print(t_stack)
    # dim =0
    t_stack = torch.stack([t, t, t], dim=0)
    print("\nt_stack.shape:{}".format(t_stack.shape))
    print(t_stack)

    # 比较 ：总结来说，torch.cat用于在现有维度上进行拼接，而torch.stack用于在新的维度上进行堆叠。

    # 3. torch.chunk(input, chunks, dim=0) 切分
    # 功能：将张量按照维度 dim 进行平均切分。若不能整除，则最后一份张量小于其他张量。
    # 所以 如果是按照某一维 切割   所形成的的是一个tuple 竟然 而且每个元素的 维度是一样的
    # chunks 表示切分的数量
    print('3. torch.chunk()')
    a = torch.ones((2, 7))  # 7
    list_of_tensors = torch.chunk(a, dim=1, chunks=3)
    print(type(list_of_tensors))
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))

    # 4. torch.split(tensor, split_size_or_sections, dim=0)
    # 功能：将张量按照维度 dim 进行平均切分。可以指定每一个分量的切分长度。
    # split_size_or_sections: 为 int 时，表示每一份的长度，如果不能被整除，则最后一份张量小于其他张量；
    # 为 list 时，按照 list 元素作为每一个分量的长度切分。如果 list 元素之和不等于切分维度 (dim) 的值，就会报错。
    print('4. torch.split()')
    t = torch.ones((2, 5))
    list_of_tensors = torch.split(t, [2, 1, 2], dim=1)
    print(type(list_of_tensors))
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))

    # 索引idx
    # 5. torch.index_select(input, dim, index, out=None)
    # 功能：在维度 dim 上，按照 index 索引取出数据拼接为张量返回。
    print('5. torch.index_select()')

    # 创建均匀分布
    t = torch.randint(0, 9, size=(3, 3))
    # 注意 idx 的 dtype 不能指定为 torch.float 索引嘛 肯定得是整数了
    idx = torch.tensor([0, 2], dtype=torch.long)
    # 取出第 0 行和第 2 行
    t_select = torch.index_select(t, dim=0, index=idx)
    print("t:\n{}\nt_select:\n{}".format(t, t_select))

    # 6. torch.masked_select(input, mask, out=None)
    # 功能：按照 mask 中的 True 进行索引拼接得到一维张量返回。
    # 按照条件 选择 之前是索引
    print('6. torch.mask_select()')
    t = torch.randint(0, 9, size=(3, 3))
    mask = t.le(5)   # ge means greater than or equal/   gt: greater than  le  lt
    # 取出小于等于5
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_select))

    # 变换
    # 7. torch.reshape(input, shape)
    # 功能：变换张量的形状。当张量在内存中是连续时，返回的张量和原来的张量共享数据内存，改变一个变量时，另一个变量也会被改变。
    print('7. torch.reshape()')
    # 生成 0 到 8 的随机排列
    t = torch.randperm(8)
    # -1 表示这个维度是根据其他维度计算得出的
    # 下面是三维的
    t_reshape = torch.reshape(t, (-1, 2, 2))
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
    # 在上面代码的基础上，修改原来的张量的一个元素，新张量也会被改变。

    # 8.torch.transpose(input, dim0, dim1)
    # 功能：交换张量的两个维度。常用于图像的变换，比如把c*h*w变换为h*w*c。
    # 把 c * h * w 变换为 h * w * c
    print('8. torch.transpose(0')
    t = torch.rand((2, 3, 4))
    t_transpose = torch.transpose(t, dim0=1, dim1=2)  # c*h*w     h*w*c
    print("t shape:{}\nt_transpose shape: {}".format(t.shape, t_transpose.shape))

    # 9. torch.t()
    # 功能：2 维张量转置，对于 2 维矩阵而言，等价于torch.transpose(input, 0, 1)。 只能是二维的
    print('9. torch.t()')
    t = torch.ones((4,2))
    print(t)
    t = torch.t(t)
    print(t)

    # 10. torch.squeeze(input, dim=None, out=None)         没太看懂有啥用
    # 功能：压缩长度为 1 的维度。
    # dim: 若为 None，则移除所有长度为 1 的维度；若指定维度，则当且仅当该维度长度为 1 时可以移除。

    print('10 torch.squeeze()')
    # 维度 0 和 3 的长度是 1
    t = torch.rand((1, 2, 3, 1))
    # 可以移除维度 0 和 3
    print(t)
    t_sq = torch.squeeze(t)
    # 可以移除维度 0
    print(t_sq)
    t_0 = torch.squeeze(t, dim=0)
    # 不能移除 1
    t_1 = torch.squeeze(t, dim=1)
    print("t.shape: {}".format(t.shape))
    print("t_sq.shape: {}".format(t_sq.shape))

    print("t_0.shape: {}".format(t_0.shape))
    print("t_1.shape: {}".format(t_1.shape))

    # 运算
    # 11.  加法
    # torch.add(input, other, out=None)
    # torch.add(input, other, *, alpha=1, out=None)
    # 功能：逐元素计算 input + alpha * other。因为在深度学习中经常用到先乘后加的操作。
    # input: 第一个张量  alpha: 乘项因子 other: 第二个张量   就是说可以在里面指定alpha
    print('11. torch.add()')
    t = torch.ones((3,3))
    alpha = 0.3
    x = 5*torch.ones(3,3)
    t = torch.add(t,x,alpha=alpha)
    print(t)








