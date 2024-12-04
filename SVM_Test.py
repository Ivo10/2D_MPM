import matplotlib.pyplot as plt
import numpy as np

from tools.image2graph import build_mask
from tools.preprocess import preprocessing
from sklearn import svm
from evaluation.CreateHeatingImage import create_heating_image
from sklearn.metrics import f1_score, classification_report, confusion_matrix

if __name__ == '__main__':
    data, _ = preprocessing([
        './datasets/npy/geochemical',
        './datasets/npy/geology'
    ])
    train_mask, val_mask, y = build_mask()
    x_train, x_val = data[train_mask], data[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    predictor = svm.SVC(probability=True, kernel='rbf', C=1.8)
    predictor.fit(x_train, y_train)

    y_pred_svm = predictor.predict(x_val)

    print(classification_report(y_val, y_pred_svm, target_names=['NO', 'YES']))
    print('F-score: {0:.3f}'.format(f1_score(y_pred_svm, y_pred_svm, average='micro')))

    mask = np.load('./datasets/npy/label/Mask.npy')
    create_heating_image(y_pred_svm, mask)

    confusion_matrix_model = confusion_matrix(y_val, y_pred_svm)
    print(confusion_matrix_model)

