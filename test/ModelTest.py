import argparse

import numpy as np
import torch.optim
import glob
import re
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data

from evaluation.CreateEvalutionCurve import create_roc_image, create_loss_image, create_acc_image
from evaluation.CreateHeatingImage import create_heating_image
from model.GCN import GCN
from model.CNN import CNNEmbedding
from model.FusionAndClassfier import FusionAndClassifier
from tools.image2graph import build_node_edge, build_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=100)

    args = parser.parse_args()
    print(args)

    node_features, edge_index = build_node_edge(2220, 1826, 3)
    train_mask, val_mask, y = build_mask()

    gcn_input = Data(x=node_features, edge_index=edge_index, y=y,
                      train_mask=train_mask, val_mask=val_mask)
    print('----------定义图数据为-----------')
    print(gcn_input)

    sub_image_folder = '../datasets/npy/cropped_images/'
    sub_image_paths = glob.glob(sub_image_folder + '*.npy')
    # sub_image_paths中文件名排序
    sub_image_paths.sort(key=lambda x: int(re.search(r'pos_(\d+)', x).group(1)))
    sub_images = [np.load(path) for path in sub_image_paths]
    images = [torch.tensor(img, dtype=torch.float32) for img in sub_images]
    cnn_input = torch.stack(images)

    cnn_model = CNNEmbedding()
    gcn_model = GCN(gcn_input)


    fusion_model = FusionAndClassifier(64, 64)
    # gcn_output = gcn_output[0].unsqueeze(0)
    #
    # fusion_model = FusionAndClassifier(1, 64)
    # combined_output = fusion_model(gcn_output, cnn_output)
    #
    #
    # model = GCN(graph_data)
    optimizer = torch.optim.Adam(list(cnn_model.parameters()) + list(gcn_model.parameters()) + list(fusion_model.parameters()),
                                 lr=0.003, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()
    losses = []
    acces = []
    epochs = []

    for epoch in range(1, args.max_epoch + 1):
        cnn_output = cnn_model(cnn_input)
        gcn_output = gcn_model(gcn_input)

        print('cnn_output', cnn_output.shape)
        print('gcn_output', gcn_output.shape)

        output = fusion_model(gcn_output, cnn_output)
        optimizer.zero_grad()
        loss = criterion(output[gcn_input.train_mask], gcn_input.y[gcn_input.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        epochs.append(epoch)

        pred = (output[gcn_input.train_mask] > 0.5).float()
        acc = accuracy_score(pred, gcn_input.y[gcn_input.train_mask].detach().numpy())
        acces.append(acc)

        print('epoch {}/{}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, args.max_epoch, loss.item(), acc))

        # # Save model weights of this epoch
        # torch.save(model.state_dict(), '../saved_model/GCN.pth')

        # model.eval()
        with torch.no_grad():
            pred = (output[gcn_input.val_mask] > 0.5).float()
            acc = accuracy_score(pred, gcn_input.y[gcn_input.val_mask].detach().numpy())

            print('epoch {}/{}, val, loss={:.4f} acc={:.4f}'.format(
                epoch, args.max_epoch, loss.item(), acc))

    fusion_model.eval()
    with torch.no_grad():
        create_loss_image(losses, epochs)
        create_acc_image(acces, epochs)
        cnn_output = cnn_model(cnn_input)
        gcn_output = gcn_model(gcn_input)

        output = fusion_model(gcn_output, cnn_output)
        mask = np.load('../datasets/npy/Mask.npy')
        create_heating_image(output, mask)
        pred = gcn_input[gcn_input.val_mask]
        create_roc_image(gcn_input.y[gcn_input.val_mask].detach().numpy(), pred)
