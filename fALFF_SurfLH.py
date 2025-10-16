import nibabel as nib
import numpy as np
import os
import pandas as pd # 用于最终输出和查看结果


def extract_surface_falff(falff_gii_path, atlas_gii_path, num_rois, hemi_prefix='lh'):
    """
    使用 NiBabel 从 fALFF 表面 GIIfi 文件中，根据 Schaefer 表面图谱提取 ROI 平均值。

    Args:
        falff_gii_path (str): fALFF 数据的 GIIfi 文件路径。
        atlas_gii_path (str): Schaefer 图谱标签的 GIIfi 文件路径。
        num_rois (int): 左半球 Schaefer ROI 的数量（例如 200）。
        hemi_prefix (str): 半球前缀，用于命名输出。
        
    Returns:
        numpy.ndarray: 包含 ROI 平均 fALFF 值的向量 (num_rois, 1)。
    """
    
    print(f"--- 正在处理 {hemi_prefix.upper()} 皮层 fALFF ---")

    # 1. 加载 fALFF 数据
    # fALFF 数据通常存储在第一个数据数组 (darrays[0]) 中
    try:
        falff_gii = nib.load(falff_gii_path)
        falff_data = falff_gii.darrays[0].data
        print(f"成功加载 fALFF 数据，维度: {falff_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载 fALFF 文件 {falff_gii_path}. 确保文件存在且格式正确。")
        raise e

    # 2. 加载 Schaefer 图谱标签
    # 图谱标签通常存储在第一个数据数组 (darrays[0]) 中
    try:
        atlas_gii = nib.load(atlas_gii_path)
        atlas_labels = atlas_gii.darrays[0].data
        print(f"成功加载 Atlas 标签，维度: {atlas_labels.shape}")
    except Exception as e:
        print(f"错误: 无法加载 Atlas 文件 {atlas_gii_path}. 确保文件存在且格式正确。")
        raise e

    # 检查数据和标签的顶点数量是否一致
    if falff_data.shape[0] != atlas_labels.shape[0]:
        raise ValueError(
            "fALFF 数据和 Atlas 标签的顶点数量不一致。请检查文件是否来自同一 fsaverage5 表面。"
        )

    # 3. 提取 ROI 平均值
    mean_falff_values = np.zeros(num_rois)
    
    # 假设 LH ROI 标签是从 1 到 num_rois (200)
    for i in range(1, num_rois + 1):
        # 找到所有标签等于当前 ROI 索引 i 的顶点
        vertex_indices = np.where(atlas_labels == i)[0]
        
        if len(vertex_indices) == 0:
            print(f"警告: ROI {i} 没有找到顶点。该 ROI 平均值设置为 0。")
            continue

        # 提取这些顶点对应的 fALFF 值
        roi_falff_data = falff_data[vertex_indices]
        
        # 计算 ROI 内所有顶点的平均值
        mean_falff = np.mean(roi_falff_data)
        
        # 存储结果 (使用 i-1 作为 Python 索引 0 到 199)
        mean_falff_values[i - 1] = mean_falff
        
        # 简单进度输出
        if i % 50 == 0:
            print(f"已处理 ROI {i}/{num_rois}")

    print("--- 提取完成 ---")
    return mean_falff_values.reshape(-1, 1) # 返回 (200, 1) 的列向量


if __name__ == '__main__':
    # --- 替换为您的实际文件路径 ---
    SUBJECT_ID = 'sub-002S0295'
    
    # 假设文件位于当前脚本运行目录下，请根据实际路径修改
    FALFF_LH_GII = f'fALFF_lh_{SUBJECT_ID}.func.gii'
    ATLAS_LH_GII = 'fsaverage5_lh_Schaefer2018_400Parcels_7Networks_order.label.gii'
    LH_NUM_ROIS = 200 # 左半球 ROI 数量

    # 检查文件是否存在
    if not os.path.exists(FALFF_LH_GII) or not os.path.exists(ATLAS_LH_GII):
        print("错误: 找不到文件。请检查以下路径是否正确：")
        print(FALFF_LH_GII)
        print(ATLAS_LH_GII)
    else:
        # 运行提取函数
        lh_falff_features = extract_surface_falff(
            FALFF_LH_GII,
            ATLAS_LH_GII,
            LH_NUM_ROIS,
            hemi_prefix='lh'
        )

        # 打印并保存结果
        roi_names = [f'Schaefer_{i}' for i in range(1, LH_NUM_ROIS + 1)]
        results_df = pd.DataFrame(lh_falff_features, index=roi_names, columns=[f'fALFF_{SUBJECT_ID}'])
        
        print("\n--- 左脑皮层 fALFF 提取结果 (部分) ---")
        print(results_df.head())
        
        # 保存为 CSV 文件
        output_filename = f'{SUBJECT_ID}_LH_fALFF_Schaefer200.csv'
        results_df.to_csv(output_filename)

        print(f"\n结果已保存至 {output_filename}")
