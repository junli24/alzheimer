import nibabel as nib
import numpy as np
import os
import pandas as pd


def extract_volumetric_falff(falff_nii_path, atlas_nii_path, num_rois, start_label=1):
    """
    使用 NiBabel 从 fALFF 体素 NIfTI 文件中，根据 Tian 54 图谱提取 ROI 平均值。

    Args:
        falff_nii_path (str): fALFF 数据的 NIfTI 文件路径。
        atlas_nii_path (str): Tian 54 图谱标签的 NIfTI 文件路径。
        num_rois (int): 皮层下 ROI 的数量（54）。
        start_label (int): 图谱中 ROI 标签的起始值 (默认为 1)。
        
    Returns:
        numpy.ndarray: 包含 ROI 平均 fALFF 值的向量 (num_rois, 1)。
    """
    
    print("--- 正在处理皮层下 54 个 ROI 的 fALFF (体素) ---")

    # 1. 加载 fALFF 数据
    try:
        falff_img = nib.load(falff_nii_path)
        falff_data = falff_img.get_fdata()  # 获取三维体素数据
        print(f"成功加载 fALFF 数据，维度: {falff_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载 fALFF 文件 {falff_nii_path}. 确保文件存在且格式正确。")
        raise e

    # 2. 加载 Tian 54 图谱标签
    try:
        atlas_img = nib.load(atlas_nii_path)
        atlas_labels = atlas_img.get_fdata() # 获取三维体素标签数据
        print(f"成功加载 Tian Atlas 标签，维度: {atlas_labels.shape}")
    except Exception as e:
        print(f"错误: 无法加载 Tian Atlas 文件 {atlas_nii_path}. 确保文件存在且格式正确。")
        raise e

    # 检查两个 NIfTI 文件的体素维度是否一致（必须在同空间、同分辨率）
    if falff_data.shape != atlas_labels.shape:
        raise ValueError(
            "fALFF 数据和 Atlas 标签的体素维度不一致。请检查它们是否都位于 MNI 2mm 空间。"
        )

    # 3. 提取 ROI 平均值
    mean_falff_values = np.zeros(num_rois)
    
    # 循环从起始标签到结束标签
    end_label = start_label + num_rois - 1
    
    for i in range(start_label, end_label + 1):
        # 找到所有标签等于当前 ROI 索引 i 的体素坐标
        # np.where 返回一个包含 (Z, Y, X) 坐标数组的元组
        voxel_coords = np.where(atlas_labels == i)
        
        # 标签 0 (背景) 已经被排除在循环之外
        if len(voxel_coords[0]) == 0:
            print(f"警告: ROI {i} ({i - start_label + 1}/{num_rois}) 没有找到体素。该 ROI 平均值设置为 0。")
            continue

        # 提取这些体素对应的 fALFF 值
        # 使用体素坐标直接索引 fALFF 数据
        roi_falff_data = falff_data[voxel_coords]
        
        # 计算 ROI 内所有体素的平均值
        # 注意：这里计算的是 fALFF 值的平均，而非加权平均，因为我们处理的是 fALFF 密度
        mean_falff = np.mean(roi_falff_data)
        
        # 存储结果 (使用 i - start_label 作为 Python 索引 0 到 53)
        mean_falff_values[i - start_label] = mean_falff
        
        # 简单进度输出
        if (i - start_label + 1) % 10 == 0:
            print(f"已处理 ROI {i - start_label + 1}/{num_rois}")

    print("--- 提取完成 ---")
    return mean_falff_values.reshape(-1, 1) # 返回 (54, 1) 的列向量


if __name__ == '__main__':
    # --- 替换为您的实际文件路径 ---
    SUBJECT_ID = 'sub-002S0295'
    
    # 假设文件路径
    FALFF_NII = f'fALFF_volu_{SUBJECT_ID}.nii'
    ATLAS_NII = 'Tian_Subcortex_S4_3T_2009cAsym.nii'
    SUBCORTICAL_NUM_ROIS = 54
    # 请根据您的 Tian 图谱实际标签起始值来设置
    # 常见的 Tian 图谱标签起始值是 1 或 401
    TIAN_START_LABEL = 1 

    # 检查文件是否存在
    if not os.path.exists(FALFF_NII) or not os.path.exists(ATLAS_NII):
        print("错误: 找不到文件。请检查以下路径是否正确：")
        print(FALFF_NII)
        print(ATLAS_NII)
    else:
        # 运行提取函数
        subc_falff_features = extract_volumetric_falff(
            FALFF_NII,
            ATLAS_NII,
            SUBCORTICAL_NUM_ROIS,
            start_label=TIAN_START_LABEL
        )

        # 打印并保存结果
        roi_names = [f'Tian_{i}' for i in range(1, SUBCORTICAL_NUM_ROIS + 1)]
        results_df = pd.DataFrame(subc_falff_features, index=roi_names, columns=[f'fALFF_{SUBJECT_ID}'])
        
        print("\n--- 皮层下 fALFF 提取结果 (部分) ---")
        print(results_df.head())
        
        # 保存为 CSV 文件
        output_filename = f'{SUBJECT_ID}_Subc_fALFF_Tian54.csv'
        results_df.to_csv(output_filename)
        print(f"\n结果已保存至 {output_filename}")