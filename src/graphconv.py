import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, BatchNorm
from torch_geometric.utils import add_self_loops


class GraphConvNetwork(torch.nn.Module):
    def __init__(self, conv_layers_num, dense_layers_num, model_dim, hidden_size, input_dim, output_dim,
                 dropout=None, batchnorm=False):
        """
        :param conv_layers_num: number of convolutional layers
        :param dense_layers_num: number of dense layers
        :param model_dim: number of channels in the conv layers
        :param hidden_size: size of the dense layers, dense_layers_num must be two or more
        :param input_dim:
        :param output_dim:
        :param dropout: fraction to drop, dropout after dense layers
        :param: batchnorm: boolean, use batchnorm after graph conv layers and dense layers?
        """
        super(GraphConvNetwork, self).__init__()

        # conv layers
        self.conv_layers_num = conv_layers_num
        self.conv_layers = [GCNConv(in_channels=input_dim, out_channels=model_dim)] + \
                           [GCNConv(in_channels=model_dim, out_channels=model_dim) for _ in range(conv_layers_num - 1)]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # batchnorm layers between graph convolutions
        self.conv_bn = None
        if batchnorm:
            self.conv_bn = [BatchNorm(model_dim) for _ in range(conv_layers_num)]
            self.conv_bn = torch.nn.ModuleList(self.conv_bn)

        # dense layers
        if 1 == dense_layers_num:
            # model_dim -> output_dim
            self.dense_layers = [torch.nn.Linear(model_dim, output_dim), ]

            if dropout is not None and dropout != 0.0:
                print("Warning: dropout was defined as a hyperparameter but is not used (dense_layers_num = 1)")

        else:
            self.hidden_size = hidden_size
            # model_dim -> hidden_size, hidden_size -> hidden_size, hidden_size -> output_dim
            self.dense_layers = [torch.nn.Linear(model_dim, hidden_size), ] + \
                                [torch.nn.Linear(hidden_size, hidden_size) for _ in range(dense_layers_num - 2)] + \
                                [torch.nn.Linear(hidden_size, output_dim), ]

        self.dense_layers_num = dense_layers_num
        self.dense_layers = torch.nn.ModuleList(self.dense_layers)

        # batchnorm layers between dense layers
        self.bn = None
        if batchnorm:
            self.bn = [BatchNorm(hidden_size) for _ in range(dense_layers_num - 1)]
            self.bn = torch.nn.ModuleList(self.bn)

        # dropouts between dense layers
        self.dropouts = None
        if dropout is not None and dropout > 0:
            self.dropouts = [torch.nn.Dropout(p=dropout) for _ in range(dense_layers_num - 1)]
            self.dropouts = torch.nn.ModuleList(self.dropouts)

    def forward(self, data):
        # conv (-> batchnorm) -> relu
        for i in range(self.conv_layers_num):
            data.x = self.conv_layers[i](data.x, data.edge_index)  # TODO: co robi edge index?
            if self.conv_bn is not None:
                data.x = self.conv_bn[i](data.x)
            data.x = F.relu(data.x)

        # pooling
        data.x = global_mean_pool(data.x, data.batch)

        # dense (-> batchnorm) -> relu (-> dropout)
        for i in range(self.dense_layers_num - 1):
            data.x = self.dense_layers[i](data.x)
            if self.bn is not None:
                data.x = self.bn[i](data.x)
            data.x = F.relu(data.x)
            if self.dropouts is not None:
                data.x = self.dropouts[i](data.x)

        # last dense
        x = self.dense_layers[-1](data.x)  # TODO: no dobra, a jak będziemy chcieli softmax?
        return x


def run_epoch(model, loss_function, optimizer, data_loader, device):
    model.train()  # set the model in training mode (changes dropout and batchnorm layers' behaviour)

    cumulative_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data).view(-1)
        loss = loss_function(output, data.y)
        loss.backward()

        cumulative_loss += data.num_graphs * loss.item()
        optimizer.step()

    return cumulative_loss


def predict(model, data_loader, device):
    model.eval()  # set dropout and batchnorm layers to evaluation mode before running inference

    pred_array = []
    true_array = []

    for data in data_loader:
        data = data.to(device)
        output = model(data).view(-1)

        pred_array.append(output.detach().cpu().numpy())
        true_array.append(data.y.cpu().numpy())

    pred_array = np.concatenate(pred_array)
    true_array = np.concatenate(true_array)

    return true_array, pred_array
