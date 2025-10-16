import nibabel as nib
import numpy as np
import os
import pandas as pd

def extract_surface_sulc(sulc_gii_path, atlas_gii_path, num_rois, hemi_prefix='lh', start_label=1):
    """
    使用 NiBabel 从皮层脑沟深度 GIIfi 文件中，根据 Schaefer 表面图谱提取 ROI 平均脑沟深度 (Mean Sulcal Depth)。

    Args:
        sulc_gii_path (str): 皮层脑沟深度数据的 GIIfi 文件路径。
        atlas_gii_path (str): Schaefer 图谱标签的 GIIfi 文件路径。
        num_rois (int): 左半球 Schaefer ROI 的数量（200）。
        hemi_prefix (str): 半球前缀，用于命名输出。
        start_label (int): 图谱中 ROI 标签的起始值 (通常为 1)。
        
    Returns:
        numpy.ndarray: 包含 ROI 平均脑沟深度的向量 (num_rois, 1)。
    """
    
    METRIC_NAME = 'Mean_Sulcal_Depth'
    print(f"--- 正在处理 {hemi_prefix.upper()} 皮层 {METRIC_NAME} (求平均) ---")

    # 1. 加载 脑沟深度数据
    try:
        sulc_gii = nib.load(sulc_gii_path)
        # 脑沟深度数据是每个顶点对应的数值
        sulc_data = sulc_gii.darrays[0].data
        print(f"成功加载 Sulcal Depth 数据，维度: {sulc_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载脑沟深度文件 {sulc_gii_path}. 确保文件存在且格式正确。")
        raise e

    # 2. 加载 Schaefer 图谱标签
    try:
        atlas_gii = nib.load(atlas_gii_path)
        atlas_labels = atlas_gii.darrays[0].data
        print(f"成功加载 Atlas 标签，维度: {atlas_labels.shape}")
    except Exception as e:
        print(f"错误: 无法加载 Atlas 文件 {atlas_gii_path}. 确保文件存在且格式正确。")
        raise e

    if sulc_data.shape[0] != atlas_labels.shape[0]:
        raise ValueError(
            "脑沟深度数据和 Atlas 标签的顶点数量不一致。请检查文件是否来自同一 fsaverage5 表面。"
        )

    # 3. 提取 ROI 平均脑沟深度
    mean_sulc_values = np.zeros(num_rois)
    end_label = start_label + num_rois - 1
    
    # 循环从起始标签 (1) 到 结束标签 (200)
    for i in range(start_label, end_label + 1): 
        # 找到所有标签等于当前 ROI 索引 i 的顶点
        vertex_indices = np.where(atlas_labels == i)[0]
        
        # 排除标签 0（背景）
        if len(vertex_indices) == 0:
            continue

        # 提取这些顶点对应的脑沟深度值
        roi_sulc_data = sulc_data[vertex_indices]
        
        # ***** 关键操作：这里使用 np.mean() 求平均 *****
        mean_sulc = np.mean(roi_sulc_data)
        
        # 存储结果 (使用 i - start_label 作为 Python 索引 0 到 199)
        mean_sulc_values[i - start_label] = mean_sulc
        
        if (i - start_label + 1) % 50 == 0:
            print(f"已处理 ROI {i - start_label + 1}/{num_rois}")

    print("--- 提取完成 ---")
    return mean_sulc_values.reshape(-1, 1) # 返回 (200, 1) 的列向量


if __name__ == '__main__':
    # --- 替换为您的实际文件路径 ---
    SUBJECT_ID = 'sub-002S0295'
    
    # 假设文件路径
    SULC_LH_GII = f'{SUBJECT_ID}_space-fsaverage5_hemi-L.sulc.gii'
    ATLAS_LH_GII = 'fsaverage5_lh_Schaefer2018_400Parcels_7Networks_order.label.gii'
    LH_NUM_ROIS = 200     # 左半球 ROI 数量
    LH_START_LABEL = 1    # LH 图谱文件中的标签从 1 开始

    # 检查文件是否存在
    if not os.path.exists(SULC_LH_GII) or not os.path.exists(ATLAS_LH_GII):
        print("错误: 找不到文件。请检查以下路径是否正确：")
        print(SULC_LH_GII)
        print(ATLAS_LH_GII)
    else:
        # 运行提取函数
        lh_sulc_features = extract_surface_sulc(
            SULC_LH_GII,
            ATLAS_LH_GII,
            LH_NUM_ROIS,
            hemi_prefix='lh',
            start_label=LH_START_LABEL
        )

        # 打印并保存结果
        roi_names = [f'LH_Schaefer_{i}' for i in range(LH_START_LABEL, LH_START_LABEL + LH_NUM_ROIS)]
        results_df = pd.DataFrame(
            lh_sulc_features, 
            index=roi_names, 
            columns=[f'MeanSulc_{SUBJECT_ID}']
        )
        
        print("\n--- 左脑皮层平均脑沟深度提取结果 (部分) ---")
        print(results_df.head())
        
        # 保存为 CSV 文件 (F3)
        output_filename = f'{SUBJECT_ID}_LH_MeanSulc_Schaefer200.csv'
        results_df.to_csv(output_filename, header=True, index=True)
        print(f"\n结果已保存至 {output_filename}")