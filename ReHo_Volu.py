import nibabel as nib
import numpy as np
import os
import pandas as pd

def extract_volumetric_metric(metric_nii_path, atlas_nii_path, num_rois, metric_name, start_label=1):
    """
    使用 NiBabel 从指标体素 NIfTI 文件中，根据 Tian 54 图谱提取 ROI 平均值。

    Args:
        metric_nii_path (str): 待提取指标（ReHo）数据的 NIfTI 文件路径。
        atlas_nii_path (str): Tian 54 图谱标签的 NIfTI 文件路径。
        num_rois (int): 皮层下 ROI 的数量（54）。
        metric_name (str): 指标名称，用于输出（例如 'ReHo'）。
        start_label (int): 图谱中 ROI 标签的起始值 (默认为 1)。
        
    Returns:
        numpy.ndarray: 包含 ROI 平均指标值的向量 (num_rois, 1)。
    """
    
    print(f"--- 正在处理皮层下 {num_rois} 个 ROI 的 {metric_name} (体素) ---")

    # 1. 加载 指标数据 (ReHo)
    try:
        metric_img = nib.load(metric_nii_path)
        metric_data = metric_img.get_fdata()  # 获取三维体素数据
        print(f"成功加载 {metric_name} 数据，维度: {metric_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载指标文件 {metric_nii_path}. 确保文件存在且格式正确。")
        raise e

    # 2. 加载 Tian 54 图谱标签
    try:
        atlas_img = nib.load(atlas_nii_path)
        atlas_labels = atlas_img.get_fdata() # 获取三维体素标签数据
        print(f"成功加载 Tian Atlas 标签，维度: {atlas_labels.shape}")
    except Exception as e:
        print(f"错误: 无法加载 Tian Atlas 文件 {atlas_nii_path}. 确保文件存在且格式正确。")
        raise e

    # 检查体素维度是否一致（必须在同空间、同分辨率）
    if metric_data.shape != atlas_labels.shape:
        raise ValueError(
            "数据和 Atlas 标签的体素维度不一致。请检查它们是否都位于 MNI 2mm 空间。"
        )

    # 3. 提取 ROI 平均值
    mean_metric_values = np.zeros(num_rois)
    
    end_label = start_label + num_rois - 1
    
    for i in range(start_label, end_label + 1):
        # 找到所有标签等于当前 ROI 索引 i 的体素坐标
        voxel_coords = np.where(atlas_labels == i)
        
        # 标签 0 (背景) 已经被排除在循环之外
        if len(voxel_coords[0]) == 0:
            print(f"警告: ROI {i} ({i - start_label + 1}/{num_rois}) 没有找到体素。该 ROI 平均值设置为 0。")
            continue

        # 提取这些体素对应的 ReHo 值
        roi_metric_data = metric_data[voxel_coords]
        
        # 计算 ROI 内所有体素的平均值
        mean_metric = np.mean(roi_metric_data)
        
        # 存储结果 (使用 i - start_label 作为 Python 索引 0 到 53)
        mean_metric_values[i - start_label] = mean_metric
        
        if (i - start_label + 1) % 10 == 0:
            print(f"已处理 ROI {i - start_label + 1}/{num_rois}")

    print("--- 提取完成 ---")
    return mean_metric_values.reshape(-1, 1) # 返回 (54, 1) 的列向量


if __name__ == '__main__':
    # --- 替换为您的实际文件路径 ---
    SUBJECT_ID = 'sub-002S0295'
    METRIC_NAME = 'ReHo'
    
    # 假设文件路径
    REHO_NII = f'ReHo_volu_{SUBJECT_ID}.nii'
    ATLAS_NII = 'Tian_Subcortex_S4_3T_2009cAsym.nii'
    SUBCORTICAL_NUM_ROIS = 54
    # 请根据您的 Tian 图谱实际标签起始值来设置
    TIAN_START_LABEL = 1 

    # 检查文件是否存在
    if not os.path.exists(REHO_NII) or not os.path.exists(ATLAS_NII):
        print("错误: 找不到文件。请检查以下路径是否正确：")
        print(REHO_NII)
        print(ATLAS_NII)
    else:
        # 运行提取函数
        subc_reho_features = extract_volumetric_metric(
            REHO_NII,
            ATLAS_NII,
            SUBCORTICAL_NUM_ROIS,
            METRIC_NAME,
            start_label=TIAN_START_LABEL
        )

        # 打印并保存结果
        roi_names = [f'Tian_{i}' for i in range(1, SUBCORTICAL_NUM_ROIS + 1)]
        results_df = pd.DataFrame(
            subc_reho_features, 
            index=roi_names, 
            columns=[f'{METRIC_NAME}_{SUBJECT_ID}']
        )
        
        print(f"\n--- 皮层下 {METRIC_NAME} 提取结果 (部分) ---")
        print(results_df.head())
        
        # 保存为 CSV 文件
        output_filename = f'{SUBJECT_ID}_Subc_{METRIC_NAME}_Tian54.csv'
        results_df.to_csv(output_filename, header=True, index=True)
        print(f"\n结果已保存至 {output_filename}")