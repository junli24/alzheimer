import nibabel as nib
import numpy as np
import os
import pandas as pd

def extract_surface_metric(metric_gii_path, atlas_gii_path, num_rois, metric_name, hemi_prefix='lh', start_label=1):
    """
    使用 NiBabel 从表面 GIIfi 文件中，根据 Schaefer 表面图谱提取 ROI 平均值。
    
    Args:
        metric_gii_path (str): 待提取指标（ReHo）数据的 GIIfi 文件路径。
        atlas_gii_path (str): Schaefer 图谱标签的 GIIfi 文件路径。
        num_rois (int): 左半球 Schaefer ROI 的数量（200）。
        metric_name (str): 指标名称，用于输出（例如 'ReHo'）。
        hemi_prefix (str): 半球前缀，用于命名输出。
        start_label (int): 图谱中 ROI 标签的起始值 (通常为 1)。
        
    Returns:
        numpy.ndarray: 包含 ROI 平均指标值的向量 (num_rois, 1)。
    """
    
    print(f"--- 正在处理 {hemi_prefix.upper()} 皮层 {metric_name} ---")

    # 1. 加载 指标数据 (ReHo)
    try:
        metric_gii = nib.load(metric_gii_path)
        metric_data = metric_gii.darrays[0].data
        print(f"成功加载 {metric_name} 数据，维度: {metric_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载指标文件 {metric_gii_path}. 确保文件存在且格式正确。")
        raise e

    # 2. 加载 Schaefer 图谱标签
    try:
        atlas_gii = nib.load(atlas_gii_path)
        atlas_labels = atlas_gii.darrays[0].data
        print(f"成功加载 Atlas 标签，维度: {atlas_labels.shape}")
    except Exception as e:
        print(f"错误: 无法加载 Atlas 文件 {atlas_gii_path}. 确保文件存在且格式正确。")
        raise e

    if metric_data.shape[0] != atlas_labels.shape[0]:
        raise ValueError(
            "数据和 Atlas 标签的顶点数量不一致。请检查文件是否来自同一 fsaverage5 表面。"
        )

    # 3. 提取 ROI 平均值
    mean_metric_values = np.zeros(num_rois)
    end_label = start_label + num_rois - 1
    
    # 循环从起始标签 (1) 到 结束标签 (200)
    for i in range(start_label, end_label + 1): 
        # 找到所有标签等于当前 ROI 索引 i 的顶点
        vertex_indices = np.where(atlas_labels == i)[0]
        
        # 排除标签 0（背景）
        if len(vertex_indices) == 0:
            # 这种情况通常只发生在标签 0 处，或图谱有缺陷时
            continue

        # 提取这些顶点对应的 ReHo 值
        roi_metric_data = metric_data[vertex_indices]
        
        # 计算 ROI 内所有顶点的平均值
        mean_metric = np.mean(roi_metric_data)
        
        # 存储结果 (使用 i - start_label 作为 Python 索引 0 到 199)
        mean_metric_values[i - start_label] = mean_metric
        
        if (i - start_label + 1) % 50 == 0:
            print(f"已处理 ROI {i - start_label + 1}/{num_rois}")

    print("--- 提取完成 ---")
    return mean_metric_values.reshape(-1, 1) # 返回 (200, 1) 的列向量


if __name__ == '__main__':
    # --- 替换为您的实际文件路径 ---
    SUBJECT_ID = 'sub-002S0295'
    METRIC_NAME = 'ReHo'
    
    # 假设文件路径
    REHO_LH_GII = f'ReHo_lh_{SUBJECT_ID}.func.gii'
    ATLAS_LH_GII = 'fsaverage5_lh_Schaefer2018_400Parcels_7Networks_order.label.gii'
    LH_NUM_ROIS = 200     # 左半球 ROI 数量
    LH_START_LABEL = 1    # LH 图谱文件中的标签从 1 开始

    # 检查文件是否存在
    if not os.path.exists(REHO_LH_GII) or not os.path.exists(ATLAS_LH_GII):
        print("错误: 找不到文件。请检查以下路径是否正确：")
        print(REHO_LH_GII)
        print(ATLAS_LH_GII)
    else:
        # 运行提取函数
        lh_reho_features = extract_surface_metric(
            REHO_LH_GII,
            ATLAS_LH_GII,
            LH_NUM_ROIS,
            METRIC_NAME,
            hemi_prefix='lh',
            start_label=LH_START_LABEL
        )

        # 打印并保存结果
        roi_names = [f'LH_Schaefer_{i}' for i in range(LH_START_LABEL, LH_START_LABEL + LH_NUM_ROIS)]
        results_df = pd.DataFrame(
            lh_reho_features, 
            index=roi_names, 
            columns=[f'{METRIC_NAME}_{SUBJECT_ID}']
        )
        
        print(f"\n--- 左脑皮层 {METRIC_NAME} 提取结果 (部分) ---")
        print(results_df.head())
        
        # 保存为 CSV 文件
        output_filename = f'{SUBJECT_ID}_LH_{METRIC_NAME}_Schaefer200.csv'
        results_df.to_csv(output_filename, header=True, index=True)
        print(f"\n结果已保存至 {output_filename}")