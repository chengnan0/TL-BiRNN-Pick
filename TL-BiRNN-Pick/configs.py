
h5py_path = rf'F:\2025\P_Picking\train_data_process\第三批数据\microseismic.hdf5'
h5py_csv = rf'F:\2025\P_Picking\train_data_process\第三批数据\microseismic.csv'

### 迁移学习数据划分
"""
    迁移学习数据量：9190 ； 80%/10%/10%
    /训练集/验证集/测试集/
"""
data_for_train = 7300
data_for_val = 900
"""
    三角形标签参数
"""
tri_left = 0
tri_middle = 5
tri_right = 10
"""
 数据长度设置
"""
data_length = 6144  #数据长度
data_cut = 100  #
"""
    数据增强
"""
sca_amp_rate = 0.7
nor_mod ='max'  # 'max';'std'

"""
    路径
"""
"""
    训练参数
"""
epochs = 50
batch_size = 32
lr = 5e-4



