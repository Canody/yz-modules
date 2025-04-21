import numpy as np
import cupy as cp
import scipy
import scipy.io as sio
import time


# 定义CUDA核函数，在GPU上进行计算
compute_xy_shift_kernel = cp.RawKernel(r'''
extern "C" __global__ 
void compute_xy_shift(const float *Matrix, const float *shift, float *x_shift, float *y_shift, int Num_pjs) {
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < Num_pjs) {
        int index = 9 * i;
        for (int j = 0; j < 4; j++) {
            x_shift[4*i+j] = Matrix[index+0]*shift[2*j] + Matrix[index+3]*0.0 + Matrix[index+6]*shift[2*j+1];
            y_shift[4*i+j] = Matrix[index+1]*shift[2*j] + Matrix[index+4]*0.0 + Matrix[index+7]*shift[2*j+1];
        }
    }
}
''', 'compute_xy_shift')

radon_tf_kernel = cp.RawKernel(r'''
extern "C" __global__
void radon_tf(const float *data, const float *Matrix, const int dimx, const int dimy, 
              const float *nc, const float *nc_pj, const int o_ratio, 
              const float *x_s, const float *y_s, float *result, 
              const int dimx_pj, const int dimy_pj, int dimz) {
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y;
    int const z = blockIdx.z;
    
    int origin_offset = 1;
    long s;
    
    if (x < dimx) {
        const float data_i = data[z*dimy*dimx + y*dimx + x];
        const float coord_x = x - nc[0] + 1;
        const float coord_y = y - nc[1] + 1;
        const float coord_z = z - nc[2] + 1;

        const float x_i = Matrix[0]*coord_x + Matrix[3]*coord_y + Matrix[6]*coord_z + nc_pj[0];
        const float y_i = Matrix[1]*coord_x + Matrix[4]*coord_y + Matrix[7]*coord_z + nc_pj[1];

        for (s=0; s<o_ratio; s++) {
            float x_is = x_i + x_s[s] - origin_offset;
            float y_is = y_i + y_s[s] - origin_offset;

            // 获取边界网格位置
            long long x_1 = (long long)floor(x_is);
            long long x_2 = x_1 + 1;
            long long y_1 = (long long)floor(y_is);
            long long y_2 = y_1 + 1;
            
            if (x_1 >= -1 && x_2 <= dimx_pj && y_1 >= -1 && y_2 <= dimy_pj) { 
                float w_x1 = x_2 - x_is;
                float w_x2 = 1 - w_x1;
                float w_y1 = y_2 - y_is;
                float w_y2 = 1 - w_y1;
                
                if (x_1 == -1) {
                    if (y_1 == -1) {
                        atomicAdd(&result[x_2 + y_2*dimx_pj], w_x2*w_y2 * data_i);
                    }
                    else if (y_2 == dimy_pj) {
                        atomicAdd(&result[x_2 + y_1*dimx_pj], w_x2*w_y1 * data_i);
                    }
                    else {
                        atomicAdd(&result[x_2 + y_1*dimx_pj], w_x2*w_y1 * data_i);
                        atomicAdd(&result[x_2 + y_2*dimx_pj], w_x2*w_y2 * data_i);                    
                    }
                }
                else if (x_2 == dimx_pj) {
                    if (y_1 == -1) {
                        atomicAdd(&result[x_1 + y_2*dimx_pj], w_x1*w_y2 * data_i);
                    }
                    else if (y_2 == dimy_pj) {
                        atomicAdd(&result[x_1 + y_1*dimx_pj], w_x1*w_y1 * data_i);
                    }
                    else {
                        atomicAdd(&result[x_1 + y_1*dimx_pj], w_x1*w_y1 * data_i);
                        atomicAdd(&result[x_1 + y_2*dimx_pj], w_x1*w_y2 * data_i);                  
                    } 
                }
                else {
                    if (y_1 == -1) {
                        atomicAdd(&result[x_1 + y_2*dimx_pj], w_x1*w_y2 * data_i);
                        atomicAdd(&result[x_2 + y_2*dimx_pj], w_x2*w_y2 * data_i);
                    }
                    else if (y_2 == dimy_pj) {
                        atomicAdd(&result[x_1 + y_1*dimx_pj], w_x1*w_y1 * data_i);
                        atomicAdd(&result[x_2 + y_1*dimx_pj], w_x2*w_y1 * data_i);
                    }
                    else {
                        atomicAdd(&result[x_1 + y_1*dimx_pj], w_x1*w_y1 * data_i);
                        atomicAdd(&result[x_1 + y_2*dimx_pj], w_x1*w_y2 * data_i);
                        atomicAdd(&result[x_2 + y_1*dimx_pj], w_x2*w_y1 * data_i);
                        atomicAdd(&result[x_2 + y_2*dimx_pj], w_x2*w_y2 * data_i);                  
                    }                               
                }
            }
        }
    }
}
''', 'radon_tf')

scale_matrix_kernel = cp.RawKernel(r'''
extern "C" __global__
void scale_matrix(float *matrix, const float scale, long long N) {
    long long const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        matrix[i] *= scale;
    }
}
''', 'scale_matrix')

def forward_projection(rec, matrix, dim_pj=None):
    """
    使用CuPy RawKernel优化的3D投影计算函数
    
    参数:
    rec: 输入的三维体积数据，形状为(dimx, dimy, dimz)的numpy数组，float32类型
    matrix: 旋转矩阵，形状为(3, 3, Num_pjs)的numpy数组，float32类型
    dim_pj: 投影的维度 [dimx_pj, dimy_pj]，默认与体积的dimx, dimy相同
    
    返回:
    projections: 投影结果，形状为(dimx_pj, dimy_pj, Num_pjs)的numpy数组
    """
    # 确保输入数据是单精度浮点数
    if rec.dtype != np.float32:
        rec = rec.astype(np.float32)
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    
    # 获取维度信息
    dimx, dimy, dimz = rec.shape
    Num_pjs = matrix.shape[2]
    
    # 设置投影维度
    if dim_pj is None:
        dimx_pj, dimy_pj = dimx, dimy
    else:
        dimx_pj, dimy_pj = dim_pj
    
    # 计算各种大小
    o_ratio = 4  # 与原始CUDA代码相同
    nrow_cols = dimx_pj * dimy_pj
    nPjsPoints = dimx_pj * dimy_pj * Num_pjs
    
    # 计算中心点（与原始CUDA代码保持一致：使用floor而非ceil）
    ncx = int(np.floor(dimx/2.0) + 1)
    ncy = int(np.floor(dimy/2.0) + 1)
    ncz = int(np.floor(dimz/2.0) + 1)
    nc = cp.array([ncx, ncy, ncz], dtype=np.float32)
    
    # 投影的中心点
    nc_pj = cp.array([float(np.floor(dimx_pj/2.0) + 1), float(np.floor(dimy_pj/2.0) + 1)], dtype=np.float32)
    
    # 将数据转移到GPU
    d_rec = cp.asarray(rec)
    
    # 重组matrix以匹配原始CUDA代码的内存布局
    d_matrix_flat = cp.asarray(matrix.reshape(3, 3, Num_pjs).transpose(2, 1, 0).reshape(-1))
    
    # 创建投影数组 - 按照原始CUDA代码的内存布局
    d_projections = cp.zeros((Num_pjs, dimy_pj, dimx_pj), dtype=np.float32)
    
    # 计算旋转偏移
    shift = cp.array([0.25, 0.25, 0.25, -0.25, -0.25, 0.25, -0.25, -0.25], dtype=np.float32)
    x_shift = cp.zeros(4 * Num_pjs, dtype=np.float32)
    y_shift = cp.zeros(4 * Num_pjs, dtype=np.float32)
    
    # 使用CUDA核计算旋转偏移
    threadsPerBlock = 256
    blocksPerGrid = (Num_pjs + threadsPerBlock - 1) // threadsPerBlock
    
    compute_xy_shift_kernel((blocksPerGrid,), (threadsPerBlock,), 
                          (d_matrix_flat, shift, x_shift, y_shift, np.int32(Num_pjs)))
    
    # 设置CUDA核函数调用的块和线程配置
    threadsPerBlock = 256
    blocksPerGridRec = ((dimx + threadsPerBlock - 1) // threadsPerBlock, dimy, dimz)
    blocksPerGridPrj = (nPjsPoints + threadsPerBlock - 1) // threadsPerBlock
    
    # 对每个投影角度执行前向投影
    print("Start forward projection...")
    start_time = time.time()
    
    # 为每个投影单独处理，与原始CUDA代码相匹配
    for i in range(Num_pjs):
        if i % 10 == 0:
            print(f"Projection num: {i+1}/{Num_pjs}")
        
        # 清空当前投影的结果区域
        d_projections[i].fill(0)
        
        # 调用CUDA核函数进行前向投影，直接使用矩阵的部分内存
        radon_tf_kernel(
            blocksPerGridRec, 
            (threadsPerBlock, 1, 1), 
            (d_rec, d_matrix_flat[9*i:9*(i+1)], np.int32(dimx), np.int32(dimy), 
             nc, nc_pj, np.int32(o_ratio), x_shift[i*o_ratio:(i+1)*o_ratio], 
             y_shift[i*o_ratio:(i+1)*o_ratio], d_projections[i].reshape(-1), 
             np.int32(dimx_pj), np.int32(dimy_pj), np.int32(dimz))
        )
    
    # 缩放投影结果
    scale_matrix_kernel(
        (blocksPerGridPrj,), 
        (threadsPerBlock,), 
        (d_projections.reshape(-1), np.float32(1.0/o_ratio), np.int64(nPjsPoints))
    )
    
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    
    # 将结果从GPU转回CPU
    projections = cp.asnumpy(d_projections)
    
    # 转置结果以匹配原始CUDA代码的输出格式 (dimx_pj, dimy_pj, Num_pjs)
    projections = np.transpose(projections, (2, 1, 0))
    
    return projections

# 示例使用方法
if __name__ == "__main__":
    # 导入 rec
    rec = sio.loadmat('../data/EPB_Volume.mat')['V_std']
    rec = rec.astype(np.float32)

    # 导入欧拉角
    euler_angles = sio.loadmat('../data/euler_angle.mat')['euler_angle']
    Num_pjs = euler_angles.shape[0]
    matrix = np.zeros((3, 3, Num_pjs), dtype=np.float32)
    # 创建旋转矩阵
    for i in range(Num_pjs):
        matrix[:,:,i] = scipy.spatial.transform.Rotation.from_euler('xyz', euler_angles[i, -1::-1], degrees=True).as_matrix().T

    # 计算投影
    projections = forward_projection(rec, matrix)
    
    print(f"{projections.shape}")

    # # 保存投影结果
    # sio.savemat('../data/projections_from_python.mat', {'projections': projections})
    