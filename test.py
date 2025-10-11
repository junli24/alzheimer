import numpy as np
from scipy.io import loadmat

mat_roi_signal = loadmat('ROISignals_sub-002S0295.mat')
roi_signal = mat_roi_signal['Data']

# 提取左脑皮层,右脑皮层,皮下
roi_signal_new = np.hstack((roi_signal[:,180:380],roi_signal[:,560:760],roi_signal[:,2993:3047]))
connectivity_matrix_from_roi = np.corrcoef(roi_signal_new.T)

connectivity_matrix_from_roi_abs = np.abs(connectivity_matrix_from_roi)
degree_centrality = np.sum(connectivity_matrix_from_roi_abs,axis=1)

print(degree_centrality.shape)
print(degree_centrality)

