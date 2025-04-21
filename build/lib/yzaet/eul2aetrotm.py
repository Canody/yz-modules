import numpy as np
import scipy


def eul2aetrotm(eul):
	"""
	为 RESIRE 创建 rotation matrix
	"""
	eul = np.atleast_2d(eul)
	num_pjs = eul.shape[0]
	matrix = np.zeros((3, 3, num_pjs), dtype=np.float32)
    # 创建旋转矩阵
	for i in range(num_pjs):
		matrix[:,:,i] = scipy.spatial.transform.Rotation.from_euler('xyz', eul[i, -1::-1], degrees=True).as_matrix().T
	return matrix
