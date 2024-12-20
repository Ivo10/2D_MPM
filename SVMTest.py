import numpy as np
from sklearn import svm
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

    predictor = svm.SVC(probability=True, kernel='rbf', C=0.1)
    predictor.fit(x_train, y_train)

    y_pred_proba = predictor.predict_proba(x_val)

    threshold = 0.5
    y_pred_class = (y_pred_proba[:, 1] >= threshold).astype(int)
    print(classification_report(y_val, y_pred_class, target_names=['NO', 'YES']))

    confusion_matrix_model = confusion_matrix(y_val, y_pred_class)
    print(confusion_matrix_model)

    mask = np.load('./mini-datasets/npy/label/Mask.npy')
    create_heating_image(predictor.predict_proba(data)[:, 1], mask)
