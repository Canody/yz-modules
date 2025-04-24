# 使用FFT进行图像仿射变换
def fft_imaffine():
    print('可选的仿射变换类型:')
    print(list_functions())

def sheardecomposition(rotmat):
    """
    将旋转矩阵分解为三个剪切变换矩阵和一个缩放因子
    rotmat - 输入的变换矩阵
    返回值: (a, b, c, d) 其中d是缩放因子，a, b, c是剪切参数
    """
    import numpy as np
    
    mstart = rotmat.copy()
    d = np.linalg.det(rotmat)  # 行列式需不小于 0
    assert d > 0, 'fft_imaffine.sheardecomposition: 必须要求 det(rotmat) > 0'
    d = np.sqrt(d)
    m = rotmat / d
    
    if np.max(np.abs(m - np.eye(2))) < 1e-6:
        a = 0
        b = 0
        c = 0
    else:
        b = m[1, 0]  # 如果b=0将会在下面失败
        assert not (b == 0), 'fft_imaffine.sheardecomposition: 必须要求 m[1,0] != 0'
        a = (m[0, 0] - 1) / b
        c = (m[1, 1] - 1) / b
        # m = sa*sb*sc
    
    # 验证分解结果
    sa = np.array([[1, a], [0, 1]])
    sb = np.array([[1, 0], [b, 1]])
    sc = np.array([[1, c], [0, 1]])
    assert np.max(np.abs(mstart - d * sa @ sb @ sc)) < 1e-6, 'fft_imaffine.sheardecomposition 分解有误'
    
    return a, b, c, d

def fft_imshear(im, sheardim, shfactor):
    """
    使用FFT实现图像剪切变换
    
    参数:
    im - 输入图像
    sheardim - 剪切维度 (0表示行方向，1表示列方向，与MATLAB中的1和2对应)
    shfactor - 剪切因子
    
    返回:
    sh - 剪切后的图像
    f - 频域中的图像表示
    """
    import numpy as np
    from numpy.fft import fft, ifft, fftshift, ifftshift
    
    # 仅当剪切因子足够大时进行剪切
    if abs(shfactor) > 1e-6:
        nr, nc = im.shape
        n = im.shape[sheardim]
        
        assert nr % 2 == 0 and nc % 2 == 0, 'fft_imaffine.fft_imshear只适用于偶数尺寸的图像'
        
        y, x = np.meshgrid(np.arange(-nc//2, nc//2), np.arange(-nr//2, nr//2))
        
        # 在指定维度上进行FFT和频谱移位
        f = fftshift(fft(im, axis=sheardim), axes=sheardim)
        
        # 应用相位调整实现剪切变换
        f = f * np.exp(-2.0j * np.pi / n * x * y * shfactor)
        
        # 确保ifft生成实数据（无虚部）
        f[0, :] = 0
        f[:, 0] = 0
        
        # 逆变换
        sh = ifft(ifftshift(f, axes=sheardim), axis=sheardim)
        
        assert np.max(np.abs(np.imag(sh))) < 1e-6, 'fft_imaffine.fft_imshear: 剪切后的图像包含虚部'
        
        # 所有虚部应该在数值精度范围内已为零
        sh = np.real(sh)
    else:
        # 如果剪切因子接近零，则不进行处理
        f = fftshift(fft(fft(im, axis=0), axis=1))
        sh = im
    
    return sh, f

def rotdegmat(rot):
    """
    根据给定的旋转角度（以度为单位）创建2D旋转矩阵
    
    参数:
    rot - 逆时针旋转角度（以度为单位）
    
    返回:
    rotmat - 2x2旋转矩阵
    """
    import numpy as np
    
    # 将角度转换为弧度
    theta = np.deg2rad(rot)
    
    # 创建旋转矩阵
    rotmat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    return rotmat

def fft_imrotate(im, angle):
    """
    使用FFT实现图像旋转
    
    参数:
    im - 输入图像
    angle - 逆时针旋转角度（以度为单位）
    
    返回:
    sh - 旋转后的图像
    """
    # 获取旋转矩阵
    rot_matrix = rotdegmat(angle)
    
    # 将旋转矩阵分解为剪切变换
    a, b, c, d = sheardecomposition(rot_matrix)
    
    # 确保是纯旋转（缩放因子接近1）
    assert abs(d - 1) < 1e-6, 'fft_imaffine.fft_imrotate: 旋转矩阵的行列式应为1'
    
    # 通过三次剪切变换的组合实现旋转
    sh1, _ = fft_imshear(im, 0, c)  # 对应MATLAB中的维度1
    sh2, _ = fft_imshear(sh1, 1, b)  # 对应MATLAB中的维度2
    sh3, _ = fft_imshear(sh2, 0, a)  # 对应MATLAB中的维度1
    
    return sh3

def fft_imbin(im, scale_factor):
    """
    使用FFT实现图像缩放（binning）
    
    参数:
    im - 输入图像或图像集（可以是3D数组，第三个维度代表不同图像）
    scale_factor - 缩放因子（如 0.5 即为 bin2）
    
    返回:
    sc - 缩放后的图像或图像集
    """
    import numpy as np
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    
    # 获取图像数量
    if im.ndim == 2:
        # 单个图像，扩展为3D数组以统一处理
        fs = np.expand_dims(im, axis=2)
        fs_num = 1
    else:
        # 多个图像的3D数组
        fs = im
        fs_num = fs.shape[2]
    
    # 初始化FFT结果数组
    fs_fft = np.zeros_like(fs, dtype=complex)
    
    # 对每个图像进行FFT变换
    for fs_ind in range(fs_num):
        fs_fft[:, :, fs_ind] = fftshift(fft2(fs[:, :, fs_ind]))
        fs_fft[0, :, fs_ind] = 0
        fs_fft[:, 0, fs_ind] = 0
    
    # 获取原始图像尺寸（前两个维度）
    imsize = np.array(fs.shape[:2])
    
    # 计算缩放后的图像尺寸，确保是偶数
    imsize_scaled = np.round(imsize * scale_factor / 2.0) * 2
    imsize_scaled = imsize_scaled.astype(int)
    
    # 计算频域中的偏移量
    poff = (imsize // 2 - imsize_scaled // 2).astype(int)
    
    # 初始化输出数组
    figset_out = np.zeros((imsize_scaled[0], imsize_scaled[1], fs_num), dtype=complex)
    
    # 对每个图像进行处理
    for fs_ind in range(fs_num):
        # 截取频域中的对应部分
        padfft = fs_fft[poff[0]:poff[0]+imsize_scaled[0], 
                         poff[1]:poff[1]+imsize_scaled[1], 
                         fs_ind]
        
        # 逆变换
        figset_out[:, :, fs_ind] = ifft2(ifftshift(padfft))
    
    # 提取实部作为最终结果
    sc = np.real(figset_out)
    
    # 如果输入是单个图像，则返回2D数组
    if fs_num == 1:
        sc = sc[:, :, 0]
    
    return sc

def fft_imshift(im, shift):
    """
    使用FFT实现图像的小数像素级平移
    
    参数:
    im - 输入图像
    shift - 平移量 [上下方向平移, 左右方向平移]，正值表示向下/向右平移
    
    返回:
    sh - 平移后的图像
    f - 频域中的图像表示
    """
    import numpy as np
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    
    # 分解平移量为整数部分和小数部分
    intshift = np.round(shift).astype(int)
    
    # 使用np.roll处理整数部分平移（对应MATLAB的circshift）
    image = np.roll(im, intshift, axis=(0, 1))
    
    # 计算剩余的小数部分平移
    shift = shift - intshift
    
    # 仅当小数部分平移量足够大时进行处理
    if np.max(np.abs(shift)) > 1e-3:
        nr, nc = image.shape
        
        assert nr % 2 == 0 and nc % 2 == 0, 'fft_imaffine.fft_imshift只适用于偶数尺寸的图像'
        
        # 创建频率网格
        y, x = np.meshgrid(np.arange(-nc//2, nc//2), np.arange(-nr//2, nr//2))
        
        # 对图像进行FFT变换
        f = fftshift(fft2(image))
        
        # 在频域应用相位调整实现平移
        # 注意MATLAB和Python中shift[0]和shift[1]的含义相同：shift[0]是y方向（行），shift[1]是x方向（列）
        f = f * np.exp(-2.0j * np.pi / nr * x * shift[1] - 2.0j * np.pi / nc * y * shift[0])
        
        # 确保ifft生成实数据（无虚部）
        f[0, :] = 0
        f[:, 0] = 0
        
        # 逆变换
        sh = ifft2(ifftshift(f))
        
        assert np.max(np.abs(np.imag(sh))) < 1e-6, 'fft_imaffine.fft_imshift: 平移后的图像包含虚部'
        
        # 所有虚部应该在数值精度范围内已为零
        sh = np.real(sh)
    else:
        # 如果小数部分平移量很小，则不进行处理
        f = fftshift(fft2(image))
        sh = image
    
    return sh, f

def fft_imexpand(im, scale_factor):
    """
    使用FFT实现图像扩展（放大）
    
    参数:
    im - 输入图像
    scale_factor - 缩放因子（必须大于1）
    
    返回:
    sc - 放大后的图像
    """
    import numpy as np
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    
    # 验证缩放因子大于1
    assert scale_factor > 1, 'fft_imaffine.fft_imexpand: 缩放因子必须大于1'
    
    # 获取原始图像尺寸
    imsize = np.array(im.shape)
    
    # 对图像进行FFT变换
    f = fftshift(fft2(im))
    
    # 将频谱的第一行和第一列置零
    f[0, :] = 0
    f[:, 0] = 0
    
    # 计算缩放后的尺寸，确保是偶数
    imsize_scaled = np.round(imsize * scale_factor / 2.0) * 2
    imsize_scaled = imsize_scaled.astype(int)
    
    # 计算频域中的偏移量
    poff = (imsize_scaled // 2 - imsize // 2).astype(int)
    
    # 创建更大的频域数组
    padfft = np.zeros(imsize_scaled, dtype=complex)
    
    # 将原始频谱放在中心
    padfft[poff[0]:poff[0]+imsize[0], poff[1]:poff[1]+imsize[1]] = f
    
    # 确保第一行和第一列为零
    padfft[0, :] = 0
    padfft[:, 0] = 0
    
    # 进行逆FFT变换
    sc = ifft2(ifftshift(padfft))
    
    # 验证结果没有明显的虚部
    assert np.max(np.abs(np.imag(sc))) < 1e-6, 'fft_imaffine.fft_imexpand: 放大后的图像包含虚部'
    
    # 提取实部作为最终结果
    sc = np.real(sc)
    
    return sc

def fft_imshrink(im, scale_factor):
    """
    使用FFT实现图像收缩（缩小）
    
    参数:
    im - 输入图像
    scale_factor - 缩放因子（必须大于1，表示收缩比例）
    
    返回:
    sc - 收缩后的图像
    """
    import numpy as np
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    
    # 辅助函数：获取背景值
    def get_background_value(image):
        nr, nc = image.shape
        # 排序所有像素值
        backlist = np.sort(image.flatten())
        # 取排序后位于 nr*nc/128.0 位置的值作为背景值
        back = backlist[int(np.round(nr * nc / 128.0))]
        return back
    
    # 验证缩放因子大于1
    assert scale_factor > 1, 'fft_imaffine.fft_imshrink: 缩放因子必须大于1'
    
    # 获取原始图像尺寸
    imsize = np.array(im.shape)
    
    # 计算缩放后的尺寸，确保是偶数
    imsize_scaled = np.round(imsize * scale_factor / 2.0) * 2
    imsize_scaled = imsize_scaled.astype(int)
    
    # 计算边缘填充大小
    margin_size = (imsize_scaled // 2 - imsize // 2).astype(int)
    
    # 获取背景值
    back = get_background_value(im)
    
    # 创建填充后的图像数组，用背景值填充
    padimage = np.ones(imsize_scaled) * back
    
    # 将原始图像放在中心
    padimage[margin_size[0]:margin_size[0]+imsize[0], 
             margin_size[1]:margin_size[1]+imsize[1]] = im
    
    # 对填充后的图像进行FFT变换
    f = fftshift(fft2(padimage))
    
    # 从频域中提取中心部分（与原始图像相同大小）
    fcroped = f[margin_size[0]:margin_size[0]+imsize[0], 
                margin_size[1]:margin_size[1]+imsize[1]]
    
    # 将频谱的第一行和第一列置零（确保结果是实数）
    fcroped[0, :] = 0
    fcroped[:, 0] = 0
    
    # 进行逆FFT变换
    sc = ifft2(ifftshift(fcroped))
    
    # 验证结果没有明显的虚部
    assert np.max(np.abs(np.imag(sc))) < 1e-6, 'fft_imaffine.fft_imshrink: 收缩后的图像包含虚部'
    
    # 提取实部作为最终结果
    sc = np.real(sc)
    
    return sc

def list_functions():
    """
    列出所有可用函数
    """
    import inspect
    import sys
    
    # 获取当前模块
    current_module = sys.modules[__name__]
    
    # 获取模块中所有函数
    functions = []
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and obj.__module__ == __name__ and name != 'list_functions':
            functions.append(name)
    
    return functions
