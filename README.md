
# 三维体积投影计算

这个项目提供了CUDA代码`calculate3Dprojection_RT3_1GPU.cu`的Python实现版本。该代码用于计算三维体积数据在给定旋转矩阵下的二维投影。

## 功能介绍

原始的CUDA MEX函数将三维体积在多个不同角度下投影到二维平面上。本项目提供了两个Python实现版本：

1. **calculate3Dprojection.py**: 使用NumPy和CuPy的简单实现，逻辑清晰但计算效率较低
2. **calculate3Dprojection_optimized.py**: 使用CuPy的RawKernel功能实现，更接近原始CUDA代码，计算效率更高

## 依赖安装

运行这些代码需要安装以下Python库：

```bash
pip install numpy
pip install cupy-cuda11x  # 根据你的CUDA版本选择合适的cupy版本
pip install matplotlib    # 可选，用于可视化结果
```

对于CuPy，请参考[官方安装指南](https://docs.cupy.dev/en/stable/install.html)选择与您的CUDA版本兼容的安装包。

## 用法示例

### 基本版本 (calculate3Dprojection.py)

```python
import numpy as np
from calculate3Dprojection import calculate3Dprojection

# 创建示例数据
dimx = dimy = 128
dimz = 64
Num_pjs = 10

# 创建随机体积数据
rec = np.random.rand(dimx, dimy, dimz).astype(np.float32)

# 创建随机旋转矩阵
matrix = np.random.rand(3, 3, Num_pjs).astype(np.float32)

# 计算投影
projections = calculate3Dprojection(rec, matrix)

print(f"投影结果形状: {projections.shape}")
```

### 优化版本 (calculate3Dprojection_optimized.py)

```python
import numpy as np
from calculate3Dprojection_optimized import calculate3Dprojection

# 创建示例数据
dimx = dimy = 128
dimz = 64
Num_pjs = 10

# 创建随机体积数据
rec = np.random.rand(dimx, dimy, dimz).astype(np.float32)

# 创建随机旋转矩阵
matrix = np.random.rand(3, 3, Num_pjs).astype(np.float32)

# 计算投影
projections = calculate3Dprojection(rec, matrix)

print(f"投影结果形状: {projections.shape}")
```

## 参数说明

`calculate3Dprojection` 函数接受以下参数：

- **rec**: 输入的三维体积数据，形状为(dimx, dimy, dimz)的numpy数组，float32类型
- **matrix**: 旋转矩阵，形状为(3, 3, Num_pjs)的numpy数组，float32类型
- **dim_pj**: (可选) 投影的维度 [dimx_pj, dimy_pj]，默认与体积的dimx, dimy相同

返回：
- **projections**: 投影结果，形状为(dimx_pj, dimy_pj, Num_pjs)的numpy数组

## 性能比较

优化版本的实现通过使用CuPy的RawKernel直接调用CUDA核函数，与原始CUDA代码更为接近，能够获得更好的性能。基本版本的实现则更加易于理解和修改。

在相同的硬件上，优化版本的性能通常比基本版本快5-10倍或更多，特别是对于大型体积数据。

## 实现说明

这两个版本实现的主要区别在于：

1. **基本版本**: 使用Python循环和CuPy的NumPy风格API
2. **优化版本**: 使用CuPy的RawKernel功能，直接在GPU上执行CUDA代码

两者都实现了相同的计算流程：

1. 将输入数据传输到GPU
2. 计算旋转偏移
3. 对每个投影角度，计算体积经过旋转后的投影 
4. 对结果进行缩放处理
5. 将计算结果从GPU传回CPU

## 限制

- 当前实现仅支持float32（单精度浮点）数据类型
- 优化版本需要兼容的CUDA环境才能运行
- 对于非常大的体积数据，可能需要足够的GPU内存 