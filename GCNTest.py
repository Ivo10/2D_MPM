import argparse
import numpy as np
import torch.optim

from torch_geometric.data import Data
from evaluation.CreateEvalutionCurve import create_roc_image, create_loss_image, create_acc_image
from evaluation.CreateHeatingImage import create_heating_image
from model.GCN import GCN
from tools.image2graph import build_edge, build_mask, add_noise, scaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.006)

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node_features, edge_index = build_edge(device)
    train_mask, val_mask, y = build_mask(device)

    node_features = scaler(node_features, train_mask, val_mask)

    node_features = add_noise(node_features, 0.5, 5)
    gcn_input = Data(x=node_features, edge_index=edge_index, y=y,
                     train_mask=train_mask, val_mask=val_mask)
    print('----------定义图数据为-----------')
    print(gcn_input)

    model = GCN(gcn_input).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()
    losses = []
    acces = []
    epochs = []

    for epoch in range(1, args.max_epoch + 1):
        output = model(gcn_input)
        print(output.shape)
        optimizer.zero_grad()
        loss = criterion(output[gcn_input.train_mask], gcn_input.y[gcn_input.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        epochs.append(epoch)

        pred_val = (output[gcn_input.train_mask] > 0.5).float()
        correct = (pred_val == gcn_input.y[gcn_input.train_mask]).sum().item()
        acc = correct / gcn_input.train_mask.sum().item()
        acces.append(acc)

        print('epoch {}/{}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, args.max_epoch, loss.item(), acc))

        # # Save model weights of this epoch
        # torch.save(model.state_dict(), './saved_model/GCN.pth')

        # model.eval()
        with torch.no_grad():
            pred_val = (output[gcn_input.val_mask] > 0.5).float()
            correct = (pred_val == gcn_input.y[gcn_input.val_mask]).sum().item()
            acc = correct / gcn_input.val_mask.sum().item()

            print('epoch {}/{}, val, loss={:.4f} acc={:.4f}'.format(
                epoch, args.max_epoch, loss.item(), acc))

    with torch.no_grad():
        create_loss_image(losses, epochs)
        create_acc_image(acces, epochs)
        output = model(gcn_input)
        mask = np.load('./datasets/npy/label/Mask.npy')
        create_heating_image(output, mask)
        pred_val = gcn_input[gcn_input.val_mask]
        create_roc_image(gcn_input.y[gcn_input.val_mask].detach().numpy(), pred_val)
