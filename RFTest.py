import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from evaluation.CreateHeatingImage import create_heating_image
from tools.image2graph import build_mask
from tools.preprocess import preprocessing

if __name__ == '__main__':
    data, _ = preprocessing([
        './mini-datasets/npy/geochemical',
        './mini-datasets/npy/geology'
    ])

    train_mask, val_mask, y = build_mask()
    x_train, x_val = data[train_mask], data[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    predictor = RandomForestClassifier(n_estimators=100, random_state=42)
    predictor.fit(x_train, y_train)

    # 获取预测的概率
    y_pred_proba = predictor.predict_proba(x_val)

    # 设置阈值进行分类
    threshold = 0.5  # 例如设置一个阈值0.5
    y_pred_class = (y_pred_proba[:, 1] >= threshold).astype(int)  # 根据阈值将概率转为标签

    # 打印分类报告
    print(classification_report(y_val, y_pred_class, target_names=['NO', 'YES']))

    # 打印混淆矩阵
    confusion_matrix_model = confusion_matrix(y_val, y_pred_class)
    print(confusion_matrix_model)

    # 打印预测概率
    print(y_pred_proba)

    # 创建加热图像
    mask = np.load('./mini-datasets/npy/label/Mask.npy')
    create_heating_image(predictor.predict_proba(data)[:, 1], mask)
