import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class DrugGNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels=512, out_channels=64, heads=8, concat=True, edge_dim=7
        )
        self.conv2 = GATv2Conv(
            in_channels=512, out_channels=64, heads=8, concat=True, edge_dim=7
        )
        self.conv3 = GATv2Conv(
            in_channels=512, out_channels=64, heads=8, concat=True, edge_dim=7
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        return global_mean_pool(x, data.batch)


class TargetGNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels=2560, out_channels=320, heads=8, concat=True, edge_dim=1
        )
        self.conv2 = GATv2Conv(
            in_channels=2560, out_channels=320, heads=8, concat=True, edge_dim=1
        )
        self.conv3 = GATv2Conv(
            in_channels=2560, out_channels=320, heads=8, concat=True, edge_dim=1
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        return global_mean_pool(x, data.batch)


class DTANet(nn.Module):

    def __init__(self):
        super().__init__()
        self.drug = DrugGNN()
        self.target = TargetGNN()
        self.out = nn.Sequential(
            nn.Linear(512 + 2560, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(inplace=True),
            nn.Linear(320, 1),
        )

    def forward(self, drug_graph, target_graph):
        return self.out(
            torch.cat((self.drug(drug_graph), self.target(target_graph)), dim=1)
        )
