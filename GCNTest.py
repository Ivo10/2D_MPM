import argparse
import numpy as np
import torch.optim
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from evaluation.CreateEvalutionCurve import create_roc_image, create_loss_image, create_acc_image
from evaluation.CreateHeatingImage import create_heating_image
from model.GCN import GCN
from tools.image2graph import build_edge, build_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--degree', type=int, default=4)

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node_features, edge_index = build_edge()
    train_mask, val_mask, y = build_mask()

    train_features = node_features[train_mask]
    val_features = node_features[val_mask]
    inference_mask = ~(train_mask | val_mask)
    inference_features = node_features[inference_mask]

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    inference_features_scaled = scaler.transform(inference_features)

    node_features[train_mask] = torch.tensor(train_features_scaled, dtype=torch.float32)
    node_features[val_mask] = torch.tensor(val_features_scaled, dtype=torch.float32)
    node_features[inference_mask] = torch.tensor(inference_features_scaled, dtype=torch.float32)

    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    y = y.to(device)

    gcn_input = Data(x=node_features, edge_index=edge_index, y=y,
                     train_mask=train_mask, val_mask=val_mask)
    print('----------定义图数据为-----------')
    print(gcn_input)

    gcn_model = GCN(gcn_input).to(device)

    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.006, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()
    losses = []
    acces = []
    epochs = []

    for epoch in range(1, args.max_epoch + 1):
        output = gcn_model(gcn_input)
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
        # torch.save(model.state_dict(), '../saved_model/GCN.pth')

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
        output = gcn_model(gcn_input)
        mask = np.load('datasets/npy/label/Mask.npy')
        create_heating_image(output, mask)
        pred_val = gcn_input[gcn_input.val_mask]
        create_roc_image(gcn_input.y[gcn_input.val_mask].detach().numpy(), pred_val)
