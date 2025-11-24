from sklearn.model_selection import KFold
import os
import pandas as pd
import numpy as np
mode = "survival"  # 1. grading  2. subtyping 3.survival
path_dir = "DATASET/tcga_glioma/labels"

label_csv = os.path.join(path_dir, f'{mode}/{mode}.csv')
data = pd.read_csv(label_csv)


# 创建5折交叉验证的KFold对象
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化一个列表来存储每次交叉验证的准确度
accuracies = []

index=1
# 执行5折交叉验证
for train_index, test_index in kf.split(data):
    # import pdb;pdb.set_trace()
    train_path = os.path.join(path_dir, mode, f"{mode}_train_{index}.csv")
    test_path = os.path.join(path_dir, mode, f"{mode}_test_{index}.csv")
    data_train, data_test = data.loc[train_index].reset_index(drop=True), data.loc[test_index].reset_index(drop=True)
    data_train.to_csv(train_path, index=False)
    data_test.to_csv(test_path, index=False)
    index +=  1

