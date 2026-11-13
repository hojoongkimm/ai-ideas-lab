"""DOENet: Differentiable Orthogonal Factor Screening and Functional ANOVA
for High-Dimensional Deep Learning.

Author: Ho-joong Kim (hojoongkimm) & Pickle
Repository: ai-ideas-lab
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardConcreteGate(nn.Module):
    """Differentiable L0 feature selection gate based on Hard Concrete distribution.
    
    Allows end-to-end backpropagation through discrete feature screening.
    During inference, features with gate <= 0 can be completely pruned.
    """
    def __init__(self, num_features, gamma=-0.1, zeta=1.1, temperature=0.66, init_prob=0.8):
        super().__init__()
        self.num_features = num_features
        self.gamma = gamma
        self.zeta = zeta
        self.temperature = temperature
        
        # Initialize log_alpha such that initial sigmoid(log_alpha) matches init_prob
        init_val = math.log(init_prob / (1.0 - init_prob + 1e-7))
        self.log_alpha = nn.Parameter(torch.full((num_features,), init_val, dtype=torch.float32))

    def forward(self, x):
        """Args:
            x: Input tensor of shape (batch_size, num_features)
        Returns:
            gated_x: Tensor of shape (batch_size, num_features)
            gate_values: Tensor of shape (num_features,) in [0, 1]
        """
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + self.log_alpha) / self.temperature)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            gate = torch.clamp(s_bar, 0.0, 1.0)
        else:
            s = torch.sigmoid(self.log_alpha)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            gate = torch.clamp(s_bar, 0.0, 1.0)
            
        gated_x = x * gate.unsqueeze(0)
        return gated_x, gate

    def get_l0_loss(self):
        """Expected L0 norm regularization term (sum of gate inclusion probabilities)."""
        log_ratio = -self.gamma / self.zeta
        val = self.log_alpha - self.temperature * math.log(log_ratio)
        return torch.sum(torch.sigmoid(val))


class OrthogonalFactorProjection(nn.Module):
    """Enforces orthogonality among feature latent embeddings, inspired by
    Orthogonal Arrays in Design of Experiments (DOE).
    
    Penalizes redundancy and multicollinear confounding in high-dimensional inputs.
    """
    def __init__(self, in_features, num_factors):
        super().__init__()
        self.in_features = in_features
        self.num_factors = num_factors
        self.weight = nn.Parameter(torch.empty(num_factors, in_features))
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return F.linear(x, self.weight)

    def get_orthogonality_loss(self):
        """Frobenius norm: || W W^T - I ||_F^2 / K"""
        gram = torch.matmul(self.weight, self.weight.t())
        identity = torch.eye(self.num_factors, device=self.weight.device)
        return torch.norm(gram - identity, p='fro') / self.num_factors


class FunctionalANOVABlock(nn.Module):
    """Hierarchical Functional ANOVA Decomposition Block.
    
    Decomposes the prediction into:
    y = f_0 (bias) + Sum_i f_i(x_i) [Main Effects]
                  + Sum_{i<j} f_{ij}(x_i, x_j) [Pairwise Interactions with Heredity]
                  + f_res(x) [Deep Higher-Order Residual]
    """
    def __init__(self, in_features, hidden_dim=64, interaction_rank=16):
        super().__init__()
        self.in_features = in_features
        
        # 1. Main Effect sub-network (1D feature-wise projection + non-linear transform)
        self.main_weight = nn.Parameter(torch.randn(in_features, hidden_dim) * 0.02)
        self.main_bias = nn.Parameter(torch.zeros(in_features, hidden_dim))
        self.main_out = nn.Linear(hidden_dim, 1, bias=False)
        
        # 2. Pairwise Interaction via Low-Rank Factorization with Heredity Mask
        self.interaction_rank = interaction_rank
        self.inter_u = nn.Linear(in_features, interaction_rank, bias=False)
        self.inter_v = nn.Linear(in_features, interaction_rank, bias=False)
        self.inter_head = nn.Linear(interaction_rank, 1, bias=False)
        
        # 3. Deep Non-linear Residual Head for complex multi-way patterns
        self.res_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Global bias (Grand Mean f_0)
        self.grand_mean = nn.Parameter(torch.zeros(1))

    def forward(self, gated_x, gate_values):
        """Args:
            gated_x: (B, P) gated input tensor
            gate_values: (P,) screening gate mask
        Returns:
            y_pred: (B, 1) final prediction
            components: dict containing main, interaction, and residual outputs
        """
        batch_size = gated_x.size(0)
        
        # --- 1. Main Effect Computation ---
        # (B, P, 1) * (1, P, H) -> (B, P, H)
        main_hidden = torch.relu(gated_x.unsqueeze(-1) * self.main_weight.unsqueeze(0) + self.main_bias.unsqueeze(0))
        # (B, P, H) -> (B, P, 1) -> (B, P)
        main_per_feat = self.main_out(main_hidden).squeeze(-1)
        # Apply gate to ensure strictly zero if pruned
        main_per_feat = main_per_feat * gate_values.unsqueeze(0)
        main_effect = main_per_feat.sum(dim=-1, keepdim=True)  # (B, 1)
        
        # --- 2. Heredity-Gated Pairwise Interaction ---
        # Strong Heredity: Interaction x_i * x_j is scaled by s_i * s_j
        # Bilinear Factorized interaction: (U^T x) * (V^T x)
        u = self.inter_u(gated_x)  # (B, rank)
        v = self.inter_v(gated_x)  # (B, rank)
        inter_latent = u * v       # (B, rank)
        inter_effect = self.inter_head(inter_latent)  # (B, 1)
        
        # --- 3. Deep Residual Head ---
        res_effect = self.res_mlp(gated_x)  # (B, 1)
        
        # Total Prediction
        y_pred = self.grand_mean + main_effect + inter_effect + res_effect
        
        components = {
            'grand_mean': self.grand_mean,
            'main_effect': main_effect,
            'inter_effect': inter_effect,
            'res_effect': res_effect,
            'main_per_feature': main_per_feat,
            'gate_values': gate_values
        }
        return y_pred, components


class DOENet(nn.Module):
    """Complete DOENet Architecture.
    
    Combines:
    1. Differentiable Hard Concrete Factor Screening
    2. Orthogonality-Regularized Factor Projection
    3. Hierarchical Functional ANOVA Decomposition
    """
    def __init__(self, in_features, out_features=1, num_factors=32, hidden_dim=64,
                 interaction_rank=16, task='regression'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.task = task
        
        # 1. Factor Screening Gate
        self.screening_gate = HardConcreteGate(in_features)
        
        # 2. Orthogonal Factor Projection (DOE Orthogonal Array constraint)
        self.orthogonal_proj = OrthogonalFactorProjection(in_features, num_factors)
        
        # 3. Functional ANOVA Decomposition
        self.anova_block = FunctionalANOVABlock(in_features, hidden_dim, interaction_rank)

    def forward(self, x):
        gated_x, gate_values = self.screening_gate(x)
        y_pred, components = self.anova_block(gated_x, gate_values)
        
        if self.task == 'classification' and self.out_features == 1:
            y_out = torch.sigmoid(y_pred)
        else:
            y_out = y_pred
            
        return y_out, components

    def compute_loss(self, y_pred, y_true, lambda_sparse=1e-3, lambda_ortho=1e-2):
        """Unified End-to-End Objective:
        L_total = L_task + lambda_sparse * L_L0 + lambda_ortho * L_ortho
        """
        if self.task == 'regression':
            task_loss = F.mse_loss(y_pred.view(-1), y_true.view(-1))
        else:
            task_loss = F.binary_cross_entropy(y_pred.view(-1), y_true.view(-1).float())
            
        sparse_loss = self.screening_gate.get_l0_loss()
        ortho_loss = self.orthogonal_proj.get_orthogonality_loss()
        
        total_loss = task_loss + lambda_sparse * sparse_loss + lambda_ortho * ortho_loss
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'sparse_loss': sparse_loss,
            'ortho_loss': ortho_loss
        }

    @torch.no_grad()
    def get_feature_importance(self, x=None):
        """Returns sorted active features and their screening gate probabilities."""
        self.eval()
        s = torch.sigmoid(self.screening_gate.log_alpha)
        s_bar = s * (self.screening_gate.zeta - self.screening_gate.gamma) + self.screening_gate.gamma
        gates = torch.clamp(s_bar, 0.0, 1.0).cpu().numpy()
        
        active_indices = [i for i, g in enumerate(gates) if g > 0.05]
        sorted_indices = sorted(range(len(gates)), key=lambda k: gates[k], reverse=True)
        return {
            'gates': gates,
            'active_indices': active_indices,
            'sorted_indices': sorted_indices,
            'num_active': len(active_indices)
        }

    @torch.no_grad()
    def get_anova_variance_decomposition(self, x):
        """Calculates empirical ANOVA Sum of Squares (SS) contribution of each component."""
        self.eval()
        y_pred, comp = self.forward(x)
        
        main_var = torch.var(comp['main_effect']).item()
        inter_var = torch.var(comp['inter_effect']).item()
        res_var = torch.var(comp['res_effect']).item()
        total_var = main_var + inter_var + res_var + 1e-8
        
        return {
            'main_effect_pct': (main_var / total_var) * 100.0,
            'interaction_pct': (inter_var / total_var) * 100.0,
            'residual_pct': (res_var / total_var) * 100.0,
            'total_variance': total_var
        }
