import numpy as np
import pandas as pd
from scipy.io import loadmat

# ----------------- 您的原始计算代码 -----------------
SUBJECT_ID = 'sub-002S0295'
MAT_FILE = 'ROISignals_sub-002S0295.mat'

print(f"--- 正在处理被试 {SUBJECT_ID} 的 ROI 信号 ---")

# 加载 ROI 信号数据
try:
    mat_roi_signal = loadmat(MAT_FILE)
    # 假设 'Data' 是存储时间序列的键名
    roi_signal = mat_roi_signal['Data'] 
except KeyError:
    print("错误: .mat 文件中没有找到 'Data' 键名。请检查实际键名。")
    exit()
except FileNotFoundError:
    print(f"错误: 找不到文件 {MAT_FILE}。请检查路径。")
    exit()

# 提取并拼接 ROI 信号 (454 ROIs)
# 假设的索引范围: LH: [180:380), RH: [560:760), Subc: [2993:3047)
# 注意: Python 切片是 [起始索引:结束索引]，不包含结束索引
NUM_LH = 200
NUM_RH = 200
NUM_SUBC = 54
TOTAL_ROIS = NUM_LH + NUM_RH + NUM_SUBC

roi_signal_new = np.hstack((
    roi_signal[:, 180:380],  # LH (200个)
    roi_signal[:, 560:760],  # RH (200个)
    roi_signal[:, 2993:3047] # Subc (54个)
))

if roi_signal_new.shape[1] != TOTAL_ROIS:
    print(f"警告: 拼接后的 ROI 数量为 {roi_signal_new.shape[1]}，预期为 {TOTAL_ROIS}。请检查索引切片。")

# 1. 计算相关系数矩阵 (邻接矩阵 A_FC)
connectivity_matrix_from_roi = np.corrcoef(roi_signal_new.T)

# 2. 计算度中心性 (Strength: 加权度)
connectivity_matrix_from_roi_abs = np.abs(connectivity_matrix_from_roi)
degree_centrality = np.sum(connectivity_matrix_from_roi_abs, axis=1)

print(f"功能连接矩阵维度: {connectivity_matrix_from_roi.shape}")
print(f"度中心性向量维度: {degree_centrality.shape}")
# ----------------------------------------------------


# ----------------- 完善后的保存代码 -----------------

# A. 保存度中心性 (Strength) 到 CSV 文件 (F12 节点特征)

# 1. 创建 ROI 标签 (确保顺序与拼接顺序一致)
lh_labels = [f'LH_Schaefer_{i}' for i in range(1, NUM_LH + 1)]
rh_labels = [f'RH_Schaefer_{i}' for i in range(1, NUM_RH + 1)]
subc_labels = [f'Tian_{i}' for i in range(1, NUM_SUBC + 1)]
all_roi_labels = lh_labels + rh_labels + subc_labels

# 2. 创建 Pandas DataFrame
degree_df = pd.DataFrame(
    degree_centrality.reshape(-1, 1),
    index=all_roi_labels,
    columns=['Functional_Strength']
)

# 3. 保存到 CSV 文件
output_degree_filename = f'{SUBJECT_ID}_Functional_Strength_454.csv'
degree_df.to_csv(output_degree_filename, header=True, index=True)

print(f"\n✅ 度中心性 (Strength) 节点特征已保存至: {output_degree_filename}")


# B. 保存连接矩阵 (邻接矩阵 A_FC) 到 NPY 文件

# 1. 定义输出文件名
output_matrix_filename = f'{SUBJECT_ID}_Functional_Connectivity_Matrix_454.npy'

# 2. 使用 NumPy 的 save 函数保存矩阵
np.save(output_matrix_filename, connectivity_matrix_from_roi)

print(f"✅ 功能连接矩阵已保存至: {output_matrix_filename}")

# ----------------------------------------------------