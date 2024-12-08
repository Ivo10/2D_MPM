import argparse
import os.path

import numpy as np
import torch.optim

from torch_geometric.data import Data
from evaluation.CreateEvalutionCurve import create_roc_image, create_loss_image, create_acc_image
from evaluation.CreateHeatingImage import create_heating_image
from model.GCN_PCA import GCN
from tools.image2graph import build_edge, build_mask, add_noise, scaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.006)

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists('./temp/data.pth'):
        data = torch.load('./temp/data.pth').to(device)
    else:
        node_features, edge_index = build_edge()
        train_mask, val_mask, y = build_mask()

        node_features = scaler(node_features, train_mask, val_mask)

        node_features = add_noise(node_features, 0.5, 5)
        data = Data(x=node_features, edge_index=edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask)
        torch.save(data, './temp/data.pth')
    print('----------定义图数据为-----------')
    print(data)

    model = GCN(data).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
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
        losses.append(loss.item())
        epochs.append(epoch)

        pred_val = (output[data.train_mask] > 0.5).float()
        correct = (pred_val == data.y[data.train_mask]).sum().item()
        acc = correct / data.train_mask.sum().item()
        acces.append(acc)

        print('epoch {}/{}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, args.max_epoch, loss.item(), acc))

        # # Save model weights of this epoch
        # torch.save(model.state_dict(), './saved_model/GCN.pth')

        # model.eval()
        with torch.no_grad():
            pred_val = (output[data.val_mask] > 0.5).float()
            correct = (pred_val == data.y[data.val_mask]).sum().item()
            acc = correct / data.val_mask.sum().item()

            print('epoch {}/{}, val, loss={:.4f} acc={:.4f}'.format(
                epoch, args.max_epoch, loss.item(), acc))

    with torch.no_grad():
        create_loss_image(losses, epochs)
        create_acc_image(acces, epochs)
        output = model(data)
        mask = np.load('./datasets/npy/label/Mask.npy')
        create_heating_image(output, mask)
        pred_val = data[data.val_mask]
        create_roc_image(data.y[data.val_mask].detach().numpy(), pred_val)
