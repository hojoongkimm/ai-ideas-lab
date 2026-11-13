"""Benchmark Training Pipeline with Multi-Seed Cross-Validation & Statistical Significance Testing.

Evaluates DOENet against deep baselines across multiple random seeds, reporting:
- Mean +/- Std on Test RMSE, Test R^2, Feature Selection Precision, Recall, F1-Score, FDP
- Paired Student's t-test (p-value) vs. Baselines to establish statistical significance
- Empirical ANOVA Variance Attribution (Main vs. Interaction vs. Deep Residual)

Usage:
    python train_benchmark.py --dim 500 --epochs 80 --num_seeds 5
"""

import argparse
import json
import time
import math
import numpy as np
import scipy.stats as stats
import torch
import torch.optim as optim

from models.doe_net import DOENet
from models.baselines import StandardMLP, L1SparseMLP, GatedMLP_Ablation
from datasets.synthetic_highdim import get_dataloaders
from utils.metrics import compute_regression_metrics, compute_feature_selection_metrics


def get_temperature(epoch, max_epochs, t_start=0.66, t_min=0.1):
    """Exponential temperature annealing schedule for discrete gate convergence."""
    decay = math.log(t_min / t_start) / max_epochs
    return max(t_min, t_start * math.exp(decay * epoch))


def train_epoch(model, loader, optimizer, model_type, epoch, max_epochs,
                lambda_sparse=2e-3, lambda_ortho=1e-2, device='cpu'):
    model.train()
    total_loss = 0.0
    temp = get_temperature(epoch, max_epochs)
    
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        
        if model_type == 'doenet':
            y_pred, _ = model(X_b, current_temperature=temp)
            loss_dict = model.compute_loss(y_pred, y_b, lambda_sparse=lambda_sparse, lambda_ortho=lambda_ortho)
        elif model_type == 'gated_mlp':
            y_pred, _ = model(X_b)
            loss_dict = model.compute_loss(y_pred, y_b, lambda_sparse=lambda_sparse)
        elif model_type == 'l1_sparse':
            y_pred, _ = model(X_b)
            loss_dict = model.compute_loss(y_pred, y_b, lambda_l1=lambda_sparse)
        else:
            y_pred, _ = model(X_b)
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
        info = model.get_feature_importance(threshold=threshold)
        selected = info['active_indices']
    elif model_type == 'gated_mlp':
        s = torch.sigmoid(model.gate.log_alpha)
        s_bar = s * (model.gate.zeta - model.gate.gamma) + model.gate.gamma
        gates = torch.clamp(s_bar, 0.0, 1.0).cpu().numpy()
        selected = [i for i, g in enumerate(gates) if g > threshold]
    elif model_type == 'l1_sparse':
        w = model.fc1.weight.abs().sum(dim=0).cpu().numpy()
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
        selected = [i for i, val in enumerate(w_norm) if val > threshold]
    else:
        selected = list(range(n_features))
    return selected


def run_single_seed(seed, args, device):
    data = get_dataloaders(n_samples=args.samples, n_features=args.dim,
                           batch_size=args.batch_size, seed=seed)
    train_loader = data['train']
    val_loader = data['val']
    test_loader = data['test']
    true_indices = data['true_indices']
    X_test_t = data['X_test'].to(device)

    models = {
        'DOENet (Ours)': ('doenet', DOENet(in_features=args.dim, num_factors=24, hidden_dim=64, interaction_rank=16).to(device)),
        'GatedMLP (w/o Ortho/ANOVA)': ('gated_mlp', GatedMLP_Ablation(in_features=args.dim, hidden_dim=64).to(device)),
        'L1-Sparse MLP (LassoNet style)': ('l1_sparse', L1SparseMLP(in_features=args.dim, hidden_dim=64).to(device)),
        'Standard MLP (Vanilla)': ('standard_mlp', StandardMLP(in_features=args.dim, hidden_dim=64).to(device))
    }

    seed_results = {}
    for name, (m_type, model) in models.items():
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        
        start_t = time.time()
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, m_type, epoch, args.epochs,
                        lambda_sparse=args.lambda_sparse, lambda_ortho=args.lambda_ortho, device=device)
            scheduler.step()
            
        elapsed = time.time() - start_t
        test_metrics = evaluate(model, test_loader, device=device)
        selected_feats = extract_selected_features(model, m_type, args.dim)
        fs_metrics = compute_feature_selection_metrics(selected_feats, true_indices, args.dim)
        
        anova_info = {}
        if m_type == 'doenet':
            anova_info = model.get_anova_variance_decomposition(X_test_t)
            
        seed_results[name] = {
            'rmse': test_metrics['rmse'],
            'r2': test_metrics['r2'],
            'precision': fs_metrics['precision'],
            'recall': fs_metrics['recall'],
            'f1': fs_metrics['f1_score'],
            'fdp': fs_metrics['fdp'],
            'num_selected': fs_metrics['num_selected'],
            'anova': anova_info,
            'time': elapsed
        }
    return seed_results


def main():
    parser = argparse.ArgumentParser(description="DOENet Multi-Seed Rigorous Benchmark Protocol")
    parser.add_argument('--dim', type=int, default=500, help='Input dimension P')
    parser.add_argument('--samples', type=int, default=2500, help='Sample size N')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--lambda_sparse', type=float, default=2e-3, help='Sparsity regularization')
    parser.add_argument('--lambda_ortho', type=float, default=1e-2, help='Orthogonality regularization')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of independent random seeds for CV')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 85)
    print(f"🚀 DOENet Multi-Seed Cross-Validation Benchmark (P={args.dim}, N={args.samples}, Seeds={args.num_seeds})")
    print(f"   Device: {device}")
    print("=" * 85)

    base_seeds = [42 + i * 17 for i in range(args.num_seeds)]
    model_names = [
        'DOENet (Ours)',
        'GatedMLP (w/o Ortho/ANOVA)',
        'L1-Sparse MLP (LassoNet style)',
        'Standard MLP (Vanilla)'
    ]

    all_seed_runs = {name: {'rmse': [], 'r2': [], 'precision': [], 'recall': [], 'f1': [], 'fdp': [], 'anova': []}
                     for name in model_names}

    for s_idx, seed in enumerate(base_seeds, 1):
        print(f"\n[Run {s_idx}/{args.num_seeds}] Running benchmark with Random Seed {seed}...")
        res = run_single_seed(seed, args, device)
        for name in model_names:
            for metric in ['rmse', 'r2', 'precision', 'recall', 'f1', 'fdp']:
                all_seed_runs[name][metric].append(res[name][metric])
            if res[name]['anova']:
                all_seed_runs[name]['anova'].append(res[name]['anova'])

    # Statistical Aggregation (Mean +/- Std)
    summary = {}
    for name in model_names:
        summary[name] = {}
        for metric in ['rmse', 'r2', 'precision', 'recall', 'f1', 'fdp']:
            vals = np.array(all_seed_runs[name][metric])
            summary[name][f"{metric}_mean"] = float(np.mean(vals))
            summary[name][f"{metric}_std"] = float(np.std(vals))

    # Paired Student's t-test vs DOENet
    doenet_rmse = np.array(all_seed_runs['DOENet (Ours)']['rmse'])
    doenet_f1 = np.array(all_seed_runs['DOENet (Ours)']['f1'])
    t_test_results = {}

    for name in model_names[1:]:
        base_rmse = np.array(all_seed_runs[name]['rmse'])
        base_f1 = np.array(all_seed_runs[name]['f1'])
        _, p_rmse = stats.ttest_rel(doenet_rmse, base_rmse)
        _, p_f1 = stats.ttest_rel(doenet_f1, base_f1)
        t_test_results[name] = {'p_rmse': float(p_rmse), 'p_f1': float(p_f1)}

    # Print Final Academic Comparison Table
    print("\n" + "=" * 105)
    print(f"📊 MULTI-SEED BENCHMARK RESULTS (Mean +/- Std across {args.num_seeds} Seeds, P={args.dim})")
    print("=" * 105)
    header = f"{'Model':<30} | {'Test RMSE':<17} | {'Test R²':<16} | {'Precision':<16} | {'Recall':<15} | {'F1-Score':<16}"
    print(header)
    print("-" * 105)
    for name in model_names:
        s = summary[name]
        rmse_str = f"{s['rmse_mean']:.3f} +/- {s['rmse_std']:.3f}"
        r2_str = f"{s['r2_mean']:.3f} +/- {s['r2_std']:.3f}"
        prec_str = f"{s['precision_mean']:.3f} +/- {s['precision_std']:.3f}"
        rec_str = f"{s['recall_mean']:.3f} +/- {s['recall_std']:.3f}"
        f1_str = f"{s['f1_mean']:.3f} +/- {s['f1_std']:.3f}"
        print(f"{name:<30} | {rmse_str:<17} | {r2_str:<16} | {prec_str:<16} | {rec_str:<15} | {f1_str:<16}")
    print("=" * 105)

    print("\n📌 Statistical Significance (Paired Student's t-test vs. DOENet):")
    for name, p_vals in t_test_results.items():
        sig_rmse = "**" if p_vals['p_rmse'] < 0.01 else ("*" if p_vals['p_rmse'] < 0.05 else "n.s.")
        sig_f1 = "**" if p_vals['p_f1'] < 0.01 else ("*" if p_vals['p_f1'] < 0.05 else "n.s.")
        print(f"  - DOENet vs. {name:<30} | RMSE p-val: {p_vals['p_rmse']:.4e} ({sig_rmse}) | F1 p-val: {p_vals['p_f1']:.4e} ({sig_f1})")

    if all_seed_runs['DOENet (Ours)']['anova']:
        anovas = all_seed_runs['DOENet (Ours)']['anova']
        m_pct = np.mean([a['main_effect_pct'] for a in anovas])
        i_pct = np.mean([a['interaction_pct'] for a in anovas])
        r_pct = np.mean([a['residual_pct'] for a in anovas])
        print(f"\n🔍 DOENet Mean ANOVA Variance Attribution (across {len(anovas)} runs):")
        print(f"  - Main Effects Variance:     {m_pct:.2f}%")
        print(f"  - Pairwise Interaction Var:  {i_pct:.2f}%")
        print(f"  - Deep Non-linear Residual:  {r_pct:.2f}%")

    out_payload = {
        'config': vars(args),
        'summary': summary,
        'significance_tests': t_test_results,
        'raw_runs': all_seed_runs
    }
    with open('benchmark_multiseed_results.json', 'w') as f:
        json.dump(out_payload, f, indent=2)
    print("\n✅ Multi-seed benchmark results saved to benchmark_multiseed_results.json")


if __name__ == '__main__':
    main()
