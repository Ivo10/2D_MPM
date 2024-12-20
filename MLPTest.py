import numpy as np
import torch

from evaluation.ComputeAccuracy import compute_accuracy
from evaluation.CreateEvalutionCurve import create_acc_image
from evaluation.CreateHeatingImage import create_heating_image
from model.MLP import MLP
from tools.image2graph import build_mask
from tools.preprocess import preprocessing


if __name__ == '__main__':
    data, _ = preprocessing([
        './mini-datasets/npy/geochemical',
        './mini-datasets/npy/geology'
    ])
    data = torch.tensor(data, dtype=torch.float32)
    train_mask, val_mask, y = build_mask()
    x_train, x_val = data[train_mask], data[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    model = MLP()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()
    epochs = []
    acces = []

    for epoch in range(2000):
        output = model(x_train)
        optimizer.zero_grad()
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        epochs.append(epoch)

        model.eval()
        with torch.no_grad():
            val_output = model(x_val)
            val_loss = criterion(val_output, y_val)
            val_accuracy = compute_accuracy(val_output, y_val)  # 计算验证集准确率

            print('epoch {}/300, train_loss={:.4f}, val_loss={:.4f}, val_accuracy={:.4f}'.format(
                epoch + 1, loss.item(), val_loss.item(), val_accuracy
            ))
            acces.append(val_accuracy)


    with torch.no_grad():
        output = model(data)
        print(output.shape)
        mask = np.load('./mini-datasets/npy/label/Mask.npy')
        create_heating_image(output, mask)
        create_acc_image(acces, epochs)