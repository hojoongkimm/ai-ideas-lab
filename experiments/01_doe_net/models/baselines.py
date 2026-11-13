"""Baseline deep learning models for comparative benchmarking and ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardMLP(nn.Module):
    """Standard multi-layer perceptron without feature selection."""
    def __init__(self, in_features, hidden_dim=64, num_layers=3, out_features=1, task='regression'):
        super().__init__()
        self.task = task
        layers = []
        curr_dim = in_features
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.1))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if self.task == 'classification':
            out = torch.sigmoid(out)
        return out, {}

    def compute_loss(self, y_pred, y_true):
        if self.task == 'regression':
            return {'total_loss': F.mse_loss(y_pred.view(-1), y_true.view(-1))}
        return {'total_loss': F.binary_cross_entropy(y_pred.view(-1), y_true.view(-1).float())}


class L1SparseMLP(nn.Module):
    """MLP with L1 regularized first layer (Lasso-style feature selection)."""
    def __init__(self, in_features, hidden_dim=64, out_features=1, task='regression'):
        super().__init__()
        self.task = task
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))
        out = self.fc3(h)
        if self.task == 'classification':
            out = torch.sigmoid(out)
        return out, {}

    def compute_loss(self, y_pred, y_true, lambda_l1=1e-3):
        if self.task == 'regression':
            task_loss = F.mse_loss(y_pred.view(-1), y_true.view(-1))
        else:
            task_loss = F.binary_cross_entropy(y_pred.view(-1), y_true.view(-1).float())
        l1_loss = torch.norm(self.fc1.weight, p=1)
        total_loss = task_loss + lambda_l1 * l1_loss
        return {'total_loss': total_loss, 'task_loss': task_loss, 'l1_loss': l1_loss}


class GatedMLP_Ablation(nn.Module):
    """Ablation Model: Hard Concrete Gate + Standard MLP.
    Lacks DOE Orthogonality and Functional ANOVA Decomposition.
    """
    def __init__(self, in_features, hidden_dim=64, out_features=1, task='regression'):
        super().__init__()
        from .doe_net import HardConcreteGate
        self.task = task
        self.gate = HardConcreteGate(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        gated_x, gate_val = self.gate(x)
        out = self.mlp(gated_x)
        if self.task == 'classification':
            out = torch.sigmoid(out)
        return out, {'gate_values': gate_val}

    def compute_loss(self, y_pred, y_true, lambda_sparse=1e-3):
        if self.task == 'regression':
            task_loss = F.mse_loss(y_pred.view(-1), y_true.view(-1))
        else:
            task_loss = F.binary_cross_entropy(y_pred.view(-1), y_true.view(-1).float())
        sparse_loss = self.gate.get_l0_loss()
        total_loss = task_loss + lambda_sparse * sparse_loss
        return {'total_loss': total_loss, 'task_loss': task_loss, 'sparse_loss': sparse_loss}
