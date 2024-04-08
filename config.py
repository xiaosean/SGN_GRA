### Experiment Setting ###
# ========================
graph_info = False
# graph_info = True
# ========================
data_volume = 100 # 百分比
# ========================
num_epochs = 200
# ========================
# pre_train = False
pre_train = True
# ========================
clf_layer_num = 1 # 測試 1 ~ 3

# 若實驗要嘗試一次使用多種 transform ，請以"減號"分隔 augmentation 方法，Ex: s_aug = "rot-sh"
# ========================
# Spatial augmentation option:
# rot (rotation)
# sh  (shear)
# GB (Guassian Blur)
# JM (Joint Mask)
# CM (Channel Mask)
# ------------------------
s_aug = "rot"
# s_aug = "rot"
# s_aug = "None"
# ========================
# Temporal augmentation option:
# re (reverse)
# shu (shuffle)
# RM (Random Mask)
# SU (Speed Up)_bil
# ------------------------
t_aug = "RM"
# t_aug = "None"
# ========================
# augmentation probability, 05 = 50%
aug_p = "05"
if s_aug == "None":
    aug_p = "00"
# ========================
if graph_info:
    ckpt_name = f"{s_aug}-{aug_p}-{t_aug}"
else:
    ckpt_name = f"SGN_STRA_06"
    #ckpt_name = "test"
