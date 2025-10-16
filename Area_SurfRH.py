import nibabel as nib
import numpy as np
import os
import pandas as pd

def extract_surface_area(area_gii_path, atlas_gii_path, num_rois, hemi_prefix='rh', start_label=1):
    """
    使用 NiBabel 从皮层表面积 GIIfi 文件中，根据 Schaefer 表面图谱提取 ROI 总表面积 (Total Area)。

    Args:
        area_gii_path (str): 皮层表面积数据的 GIIfi 文件路径。
        atlas_gii_path (str): Schaefer 图谱标签的 GIIfi 文件路径。
        num_rois (int): 右半球 Schaefer ROI 的数量（200）。
        hemi_prefix (str): 半球前缀，用于命名输出。
        start_label (int): 图谱中 ROI 标签的起始值 (通常为 1)。
        
    Returns:
        numpy.ndarray: 包含 ROI 总表面积的向量 (num_rois, 1)。
    """
    
    METRIC_NAME = 'Cortical_Surface_Area'
    print(f"--- 正在处理 {hemi_prefix.upper()} 皮层 {METRIC_NAME} (求和) ---")

    # 1. 加载 表面积数据
    try:
        area_gii = nib.load(area_gii_path)
        area_data = area_gii.darrays[0].data
        print(f"成功加载 Area 数据，维度: {area_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载面积文件 {area_gii_path}. 确保文件存在且格式正确。")
        raise e

    # 2. 加载 Schaefer 图谱标签
    try:
        atlas_gii = nib.load(atlas_gii_path)
        atlas_labels = atlas_gii.darrays[0].data
        print(f"成功加载 Atlas 标签，维度: {atlas_labels.shape}")
    except Exception as e:
        print(f"错误: 无法加载 Atlas 文件 {atlas_gii_path}. 确保文件存在且格式正确。")
        raise e

    if area_data.shape[0] != atlas_labels.shape[0]:
        raise ValueError(
            "面积数据和 Atlas 标签的顶点数量不一致。请检查文件是否来自同一 fsaverage5 表面。"
        )

    # 3. 提取 ROI 总表面积
    total_area_values = np.zeros(num_rois)
    end_label = start_label + num_rois - 1
    
    # 循环从起始标签 (1) 到 结束标签 (200)
    for i in range(start_label, end_label + 1): 
        # 找到所有标签等于当前 ROI 索引 i 的顶点
        vertex_indices = np.where(atlas_labels == i)[0]
        
        # 排除标签 0（背景）
        if len(vertex_indices) == 0:
            continue

        # 提取这些顶点对应的面积值
        roi_area_data = area_data[vertex_indices]
        
        # ***** 关键操作：求和 (Sum) *****
        total_area = np.sum(roi_area_data)
        
        # 存储结果 (使用 i - start_label 作为 Python 索引 0 到 199)
        total_area_values[i - start_label] = total_area
        
        if (i - start_label + 1) % 50 == 0:
            print(f"已处理 ROI {i - start_label + 1}/{num_rois}")

    print("--- 提取完成 ---")
    return total_area_values.reshape(-1, 1) # 返回 (200, 1) 的列向量


if __name__ == '__main__':
    # --- 替换为您的实际文件路径 ---
    SUBJECT_ID = 'sub-002S0295'
    
    # 假设文件路径
    AREA_RH_GII = f'{SUBJECT_ID}_space-fsaverage5_hemi-R.area.gii'
    ATLAS_RH_GII = 'fsaverage5_rh_Schaefer2018_400Parcels_7Networks_order.label.gii'
    RH_NUM_ROIS = 200     # 右半球 ROI 数量
    RH_START_LABEL = 1    # RH 图谱文件中的标签从 1 开始

    # 检查文件是否存在
    if not os.path.exists(AREA_RH_GII) or not os.path.exists(ATLAS_RH_GII):
        print("错误: 找不到文件。请检查以下路径是否正确：")
        print(AREA_RH_GII)
        print(ATLAS_RH_GII)
    else:
        # 运行提取函数
        rh_area_features = extract_surface_area(
            AREA_RH_GII,
            ATLAS_RH_GII,
            RH_NUM_ROIS,
            hemi_prefix='rh',
            start_label=RH_START_LABEL
        )

        # 打印并保存结果
        roi_names = [f'RH_Schaefer_{i}' for i in range(RH_START_LABEL, RH_START_LABEL + RH_NUM_ROIS)]
        results_df = pd.DataFrame(
            rh_area_features, 
            index=roi_names, 
            columns=[f'Area_{SUBJECT_ID}']
        )
        
        print("\n--- 右脑皮层表面积提取结果 (部分) ---")
        print(results_df.head())
        
        # 保存为 CSV 文件 (F1)
        output_filename = f'{SUBJECT_ID}_RH_Area_Schaefer200.csv'
        results_df.to_csv(output_filename, header=True, index=True)
        print(f"\n结果已保存至 {output_filename}")