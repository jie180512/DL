import numpy as np


def _single_channel_conv(z, k, b=0, padding=(0,0), strides=(1, 1)):
    """
    单通道卷积
    :param z: 卷积层矩阵
    :param k: 卷积核
    :param b: 偏置
    :param padding: 填充
    :param strides: 步长
    """
    padding_z = np.lib.pad(z, ((padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    height, width = padding_z.shape
    k1, k2 = k.shape
    assert(height - k1) % strides[0] == 0
    assert(width - k2) % strides[1] == 0
    conv_z = np.zeros((1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for h in np.arange(height - k1 + 1)[::strides[0]]:
        for w in np.arange(width - k2 + 1)[::strides[1]]:
            conv_z[n, d, h // strides[0], w // strides[1]] = np.sum(padding_z[n, :, h:h+k1, w:w+k2]*k[:, d])
    
def _remove_padding(z, padding):
    """
    移除填充
    """
    if padding[0] > 0 and padding[1] > 0:
        return z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return z[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return z[:, :, :, padding[1]:-padding[1]]
    else:
        return z
    
def conv_forward_bak(z, k, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积
    :param z: 卷积层矩阵,形状(N,C,H,W),N为batch_size,C为通道数
    :param k: 卷积核，形状(C,D,k1,k2),C为输入通道数,D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: 填充
    :param strides: 步长
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = k.shape
    assert(height - k1) % strides[0] == 0
    assert(width - k2) % strides[1] == 0
    conv_z = np.zeros((N, D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 +1)[::strides[1]]:
                   conv_z[n, d, h // strides[0], w // strides[1]] = \
                        np.sum(padding_z[n, :,h:h+k1, w:w+k2] * k[:, d]) + b[d]
    return con_z

def _insert_zeros(dz, strides):
    """
    零填充
    :param dz: (N,D,H,W)
    :param strides: 步长
    """
    _, _, H, W = dz.shape
    pz = dz
    if strides[0] > 1:
        for h in np.arange(H - 1, 0, -1):
            for o in np.arange(strides[0] - 1):
                pz = np.insert(pz, h, 0, axis=2)
    if strides[1] > 1:
        for w in np.arange(W - 1, 0, -1):
            for o in np.arange(strides[1] - 1):
                pz = np.insert(pz, w, 0, axis=3)
    return pz

def conv_backward(next_dz, k, z, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积层反向传播
    :param next_dz: 卷积输出层的梯度,形状(N,D,H,W)
    :param k: 卷积核，形状(C,D,k1,k2),C为输入通道数,D为输出通道数
    :param z: 卷积层矩阵,形状(N,C,H,W)
    :param padding: 填充
    :param strides: 步长
    """
    N, C, H, W = z.shape
    C, D, k1, k2 = K.shape
    padding_next_dz = _insert_zeros(next_dz, strides)

    #卷积核高度和宽度翻转180度
    flip_k = np.flip(k, (2, 3))
    #交换C，D为D，C，即D变为输入通道，C变为输出通道
    swap_flip_k = np.swapaxes(flip_k, 0, 1)
    #零填充增加高度和宽度
    padding_next_dz = np.lib.pad(z, ((0, 0), (0, 0),(k1 - 1, k1 - 1),\
                                 (k2 - 1, k2 - 1)), 'constant', constant_values=0)
    dz = conv_forward(padding_next_dz, swap_flip_k, np.zeros((C,), dtype=np.float))

    #求卷积和的梯度dk
    swap_z = np.swapaxes(z, 0, 1) #(C,N,H,W)
    dk = conv_forward(swap_z, padding_next_dz, np.zeros((D,), dtype=np.float))

    #偏置的梯度
    db  = np.sum(np.sum(np,sum(next_dz, axis=-1), axis=-1), axis=0) #在高宽上相加，批量大小上相加

    #去掉padding
    dz = _remove_padding(dz, padding)

    return dk / N, db / N, dz

def max_pooling_forward_bak(z, pooling, strides=(2,2), padding=(0, 0)):
    """
    最大池化
    :param z: 卷积层矩阵,形状(N,C,H,W),N为bitch_size,C为通道数
    :param pooling: 池化大小(k1, k2)
    :param strides: 步长
    :param padding: 填充
    """
    N, C, H, W = z.shape
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    #输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w), dtype=np.float32)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                   pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                strides[0] * i:strides[0] * i + pooling[0],
                                                strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z

def max_pooling_backward_bak(next_dz, z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化反向过程
    :param next_dz：损失函数关于最大池化输出的损失
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    _, _, out_h, out_w = next_dz.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    # 零填充后的梯度
    padding_dz = np.zeros_like(padding_z)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    # 找到最大值的那个元素坐标，将梯度传给这个坐标
                    flat_idx = np.argmax(padding_z[n, c,
                                         strides[0] * i:strides[0] * i + pooling[0],
                                         strides[1] * j:strides[1] * j + pooling[1]])
                    h_idx = strides[0] * i + flat_idx // pooling[1]
                    w_idx = strides[1] * j + flat_idx % pooling[1]
                    padding_dz[n, c, h_idx, w_idx] += next_dz[n, c, i, j]
    # 返回时剔除零填充
    return _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

def avg_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    平均池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w), dtype=np.float32)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.mean(padding_z[n, c,
                                                 strides[0] * i:strides[0] * i + pooling[0],
                                                 strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z


def avg_pooling_backward(next_dz, z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    平均池化反向过程
    :param next_dz：损失函数关于最大池化输出的损失
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    _, _, out_h, out_w = next_dz.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    # 零填充后的梯度
    padding_dz = np.zeros_like(padding_z)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    # 每个神经元均分梯度
                    padding_dz[n, c,
                    strides[0] * i:strides[0] * i + pooling[0],
                    strides[1] * j:strides[1] * j + pooling[1]] += next_dz[n, c, i, j] / (pooling[0] * pooling[1])
    # 返回时剔除零填充
    return _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]