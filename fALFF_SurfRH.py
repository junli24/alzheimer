import nibabel as nib
import numpy as np
import os
import pandas as pd

def extract_surface_falff(falff_gii_path, atlas_gii_path, num_rois, hemi_prefix='rh', start_label=1):
    """
    使用 NiBabel 从 fALFF 表面 GIIfi 文件中，根据 Schaefer 表面图谱提取 ROI 平均值。

    Args:
        falff_gii_path (str): fALFF 数据的 GIIfi 文件路径。
        atlas_gii_path (str): Schaefer 图谱标签的 GIIfi 文件路径。
        num_rois (int): 右半球 Schaefer ROI 的数量（例如 200）。
        hemi_prefix (str): 半球前缀，用于命名输出。
        start_label (int): 图谱中 ROI 标签的起始值 (通常为 1)。
        
    Returns:
        numpy.ndarray: 包含 ROI 平均 fALFF 值的向量 (num_rois, 1)。
    """
    
    print(f"--- 正在处理 {hemi_prefix.upper()} 皮层 fALFF ---")

    # 1. 加载 fALFF 数据
    try:
        falff_gii = nib.load(falff_gii_path)
        falff_data = falff_gii.darrays[0].data
        print(f"成功加载 fALFF 数据，维度: {falff_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载 fALFF 文件 {falff_gii_path}. 确保文件存在且格式正确。")
        raise e

    # 2. 加载 Schaefer 图谱标签
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
    
    # 循环从起始标签 (1) 到 结束标签 (200)
    for i in range(start_label, start_label + num_rois): 
        # 找到所有标签等于当前 ROI 索引 i 的顶点
        vertex_indices = np.where(atlas_labels == i)[0]
        
        # 标签 0 代表背景，如果标签从 1 开始，则不需要额外处理 0 标签
        # 但我们仍然检查 ROI 是否为空，以防万一
        if len(vertex_indices) == 0:
            # 警告：如果 ROI 标签从 1 到 200 都是有效的，那么这里不应该触发
            print(f"警告: ROI {i} 没有找到顶点。该 ROI 平均值设置为 0。")
            continue

        # 提取这些顶点对应的 fALFF 值并计算平均值
        roi_falff_data = falff_data[vertex_indices]
        mean_falff = np.mean(roi_falff_data)
        
        # 存储结果 (使用 i - start_label 作为 Python 索引 0 到 199)
        mean_falff_values[i - start_label] = mean_falff
        
        # 简单进度输出
        if i % 50 == 0:
            print(f"已处理 ROI {i}/{start_label + num_rois - 1}")

    print("--- 提取完成 ---")
    return mean_falff_values.reshape(-1, 1) # 返回 (200, 1) 的列向量


if __name__ == '__main__':
    # --- 替换为您的实际文件路径 ---
    SUBJECT_ID = 'sub-002S0295'
    
    # 请根据实际路径修改
    FALFF_RH_GII = f'fALFF_rh_{SUBJECT_ID}.func.gii'
    ATLAS_RH_GII = 'fsaverage5_rh_Schaefer2018_400Parcels_7Networks_order.label.gii'
    RH_NUM_ROIS = 200     # 右半球 ROI 数量
    RH_START_LABEL = 1    # RH 图谱文件中的标签从 1 开始

    # 检查文件是否存在
    if not os.path.exists(FALFF_RH_GII) or not os.path.exists(ATLAS_RH_GII):
        print("错误: 找不到文件。请检查以下路径是否正确：")
        print(FALFF_RH_GII)
        print(ATLAS_RH_GII)
    else:
        # 运行提取函数
        rh_falff_features = extract_surface_falff(
            FALFF_RH_GII,
            ATLAS_RH_GII,
            RH_NUM_ROIS,
            hemi_prefix='rh',
            start_label=RH_START_LABEL
        )

        # 打印并保存结果
        # 注意：这里的 ROI 命名是相对于 RH 文件内部的编号 (1-200)
        roi_names = [f'RH_Schaefer_{i}' for i in range(RH_START_LABEL, RH_START_LABEL + RH_NUM_ROIS)]
        results_df = pd.DataFrame(rh_falff_features, index=roi_names, columns=[f'fALFF_{SUBJECT_ID}'])
        
        print("\n--- 右脑皮层 fALFF 提取结果 (部分) ---")
        print(results_df.head())
        
        # 保存为 CSV 文件
        output_filename = f'{SUBJECT_ID}_RH_fALFF_Schaefer200.csv'
        results_df.to_csv(output_filename)
        print(f"\n结果已保存至 {output_filename}")