import argparse
import os

import numpy as np
import torch.optim
from torch_geometric.data import Data

from evaluation.ComputeAccuracy import compute_accuracy
from evaluation.CreateEvalutionCurve import create_roc_image, create_loss_image, create_acc_image
from evaluation.CreateHeatingImage import create_heating_image
from model.Residual_GAT import Residual_GAT
from tools.image2graph import build_edge, build_mask, scaler, add_noise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=2000)
    parser.add_argument('--degree', type=int, default=4)

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists('./temp/data.pth'):
        data = torch.load('./temp/data.pth').to(device)
    else:
        node_features, edge_index = build_edge()
        train_mask, val_mask, y = build_mask()

        # node_features = scaler(node_features, train_mask, val_mask)
        #
        # node_features = add_noise(node_features, 0.5, 5)
        data = Data(x=node_features, edge_index=edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask)
        torch.save(data, './temp/data.pth')
    print('----------定义图数据为-----------')
    print(data)

    model = Residual_GAT(data).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()
    losses = []
    acces = []
    epochs = []

    for epoch in range(1, args.max_epoch + 1):
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        epochs.append(epoch)

        # # Save model weights of this epoch
        # torch.save(model.state_dict(), '../saved_model/GCN.pth')

        model.eval()
        with torch.no_grad():
            val_output = output[data.val_mask]
            val_loss = criterion(val_output, data.y[data.val_mask])
            losses.append(val_loss.item())
            val_accuracy = compute_accuracy(val_output, data.y[data.val_mask])

            print('epoch {}/{}, train_loss={:.4f}, val_loss={:.4f}, val_accuracy={:.4f}'.format(
                epoch + 1, args.max_epoch, loss.item(), val_loss.item(), val_accuracy
            ))
            acces.append(val_accuracy)

    with torch.no_grad():
        create_loss_image(losses, epochs)
        create_acc_image(acces, epochs)
        output = model(data)
        mask = np.load('./mini-datasets/npy/label/Mask.npy')
        create_heating_image(output, mask)
        pred = data[data.val_mask]
        create_roc_image(data.y[data.val_mask].detach().numpy(), pred)
