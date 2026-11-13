"""Benchmark Training Pipeline: DOENet vs. Deep Baselines on High-Dimensional Data.

Compares:
1. DOENet (Ours: Differentiable Orthogonal Screening + Functional ANOVA)
2. GatedMLP (Ablation: Feature Gate without Orthogonality & ANOVA)
3. L1SparseMLP (Baseline: L1 Regularized Linear Skip)
4. StandardMLP (Baseline: Vanilla Deep MLP)

Usage:
    python train_benchmark.py --dim 500 --epochs 80 --batch_size 64
"""

import argparse
import json
import time
import torch
import torch.optim as optim
import numpy as np

from models.doe_net import DOENet
from models.baselines import StandardMLP, L1SparseMLP, GatedMLP_Ablation
from datasets.synthetic_highdim import get_dataloaders
from utils.metrics import compute_regression_metrics, compute_feature_selection_metrics


def train_epoch(model, loader, optimizer, model_type, lambda_sparse=1e-3, lambda_ortho=1e-2, device='cpu'):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        
        y_pred, _ = model(X_b)
        
        if model_type == 'doenet':
            loss_dict = model.compute_loss(y_pred, y_b, lambda_sparse=lambda_sparse, lambda_ortho=lambda_ortho)
        elif model_type == 'gated_mlp':
            loss_dict = model.compute_loss(y_pred, y_b, lambda_sparse=lambda_sparse)
        elif model_type == 'l1_sparse':
            loss_dict = model.compute_loss(y_pred, y_b, lambda_l1=lambda_sparse)
        else:
            loss_dict = model.compute_loss(y_pred, y_b)
            
        loss = loss_dict['total_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_b)
        
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device='cpu'):
    model.eval()
    preds, targets = [], []
    for X_b, y_b in loader:
        X_b = X_b.to(device)
        y_pred, _ = model(X_b)
        preds.append(y_pred.cpu())
        targets.append(y_b)
        
    y_preds = torch.cat(preds, dim=0)
    y_trues = torch.cat(targets, dim=0)
    return compute_regression_metrics(y_trues, y_preds)


def extract_selected_features(model, model_type, n_features, threshold=0.1):
    model.eval()
    if model_type == 'doenet':
        info = model.get_feature_importance()
        selected = [i for i, g in enumerate(info['gates']) if g > threshold]
    elif model_type == 'gated_mlp':
        s = torch.sigmoid(model.gate.log_alpha)
        s_bar = s * (model.gate.zeta - model.gate.gamma) + model.gate.gamma
        gates = torch.clamp(s_bar, 0.0, 1.0).cpu().numpy()
        selected = [i for i, g in enumerate(gates) if g > threshold]
    elif model_type == 'l1_sparse':
        w = model.fc1.weight.abs().sum(dim=0).cpu().numpy()
        # Top-k or dynamic threshold based on mean weight
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
        selected = [i for i, val in enumerate(w_norm) if val > threshold]
    else:
        # Standard MLP does not screen features
        selected = list(range(n_features))
    return selected


def main():
    parser = argparse.ArgumentParser(description="DOENet High-Dimensional Benchmark Experiment")
    parser.add_argument('--dim', type=int, default=500, help='Total number of input features (P)')
    parser.add_argument('--samples', type=int, default=2500, help='Total sample size (N)')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs per model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--lambda_sparse', type=float, default=2e-3, help='Sparsity regularization weight')
    parser.add_argument('--lambda_ortho', type=float, default=1e-2, help='DOE Orthogonality regularization weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f"🚀 DOENet Benchmark Experiment (P={args.dim}, N={args.samples}, Device={device})")
    print("=" * 70)

    # 1. Load High-Dimensional Benchmark Data
    data = get_dataloaders(n_samples=args.samples, n_features=args.dim, batch_size=args.batch_size, seed=args.seed)
    train_loader = data['train']
    val_loader = data['val']
    test_loader = data['test']
    true_indices = data['true_indices']
    X_test_t = data['X_test'].to(device)

    print(f"Ground-truth active causal features: {true_indices} (10 of {args.dim})")
    print(f"Features 10..49 are multicollinear confounding variables.")
    print(f"Features 50..{args.dim-1} are pure noise variables.")
    print("-" * 70)

    models = {
        'DOENet (Ours)': ('doenet', DOENet(in_features=args.dim, num_factors=24, hidden_dim=64, interaction_rank=16).to(device)),
        'GatedMLP (w/o Ortho/ANOVA)': ('gated_mlp', GatedMLP_Ablation(in_features=args.dim, hidden_dim=64).to(device)),
        'L1-Sparse MLP (LassoNet style)': ('l1_sparse', L1SparseMLP(in_features=args.dim, hidden_dim=64).to(device)),
        'Standard MLP (Vanilla)': ('standard_mlp', StandardMLP(in_features=args.dim, hidden_dim=64).to(device))
    }

    benchmark_summary = {}

    for name, (m_type, model) in models.items():
        print(f"\nTraining [{name}]...")
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        
        start_t = time.time()
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, m_type,
                               lambda_sparse=args.lambda_sparse, lambda_ortho=args.lambda_ortho, device=device)
            scheduler.step()
            if epoch % 20 == 0 or epoch == args.epochs:
                val_res = evaluate(model, val_loader, device=device)
                print(f"  Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {loss:.4f} | Val RMSE: {val_res['rmse']:.4f} | Val R²: {val_res['r2']:.4f}")
                
        elapsed = time.time() - start_t
        
        # Test Evaluation
        test_metrics = evaluate(model, test_loader, device=device)
        selected_feats = extract_selected_features(model, m_type, args.dim)
        fs_metrics = compute_feature_selection_metrics(selected_feats, true_indices, args.dim)
        
        # ANOVA Decomposition for DOENet
        anova_info = {}
        if m_type == 'doenet':
            anova_info = model.get_anova_variance_decomposition(X_test_t)
            
        benchmark_summary[name] = {
            'test_rmse': test_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'num_selected': fs_metrics['num_selected'],
            'fs_precision': fs_metrics['precision'],
            'fs_recall': fs_metrics['recall'],
            'fs_f1': fs_metrics['f1_score'],
            'training_time_sec': elapsed,
            'anova': anova_info
        }

    # Print Final Academic Comparison Table
    print("\n" + "=" * 95)
    print("📊 BENCHMARK EXPERIMENT RESULTS (High-Dimensional Tabular P=500)")
    print("=" * 95)
    header = f"{'Model':<30} | {'Test RMSE':<9} | {'Test R²':<8} | {'Selected':<8} | {'Precision':<9} | {'Recall':<7} | {'F1-Score':<8}"
    print(header)
    print("-" * 95)
    for name, res in benchmark_summary.items():
        line = (f"{name:<30} | "
                f"{res['test_rmse']:<9.4f} | "
                f"{res['test_r2']:<8.4f} | "
                f"{res['num_selected']:<8d} | "
                f"{res['fs_precision']:<9.4f} | "
                f"{res['fs_recall']:<7.4f} | "
                f"{res['fs_f1']:<8.4f}")
        print(line)
    print("=" * 95)

    if 'DOENet (Ours)' in benchmark_summary and benchmark_summary['DOENet (Ours)']['anova']:
        anova = benchmark_summary['DOENet (Ours)']['anova']
        print(f"\n🔍 DOENet Functional ANOVA Decomposition on Test Set:")
        print(f"  - Main Effects Variance:     {anova['main_effect_pct']:.2f}%")
        print(f"  - Pairwise Interaction Var:  {anova['interaction_pct']:.2f}%")
        print(f"  - Deep Non-linear Residual:  {anova['residual_pct']:.2f}%")

    with open('benchmark_results.json', 'w') as f:
        json.dump(benchmark_summary, f, indent=2)
    print("\n✅ Results successfully saved to benchmark_results.json")


if __name__ == '__main__':
    main()
