import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
}

COLORS = {
    'negative': '#2E86AB',  # Blue - clear, professional
    'positive': '#A23B72',  # Magenta - distinct from blue
    'neutral': '#7C7C7C',   # Gray
    'baseline': '#F18F01',  # Orange - for reference lines
    'ablation_top': '#C73E1D',     # Red - danger/impact
    'ablation_random': '#6A994E',  # Green - control
    'gradient_start': '#440154',
    'gradient_end': '#FDE724',
}

COLORBLIND_SAFE = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'cyan': '#56B4E9',
    'magenta': '#CA9161',
    'gray': '#949494',
}

def set_style():
    plt.rcParams.update(CONFIG)
    sns.set_palette("husl")
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("whitegrid", {
        'grid.linestyle': ':',
        'grid.alpha': 0.3,
        'axes.edgecolor': '.15',
        'axes.linewidth': 1.2,
    })

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', 
                         save_path=None, figsize=(7, 5.5)):
    set_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', 
                   vmin=0, vmax=1, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion', rotation=270, labelpad=20)
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            text = f'{count:,}\n({pct:.1f}%)'
            ax.text(j, i, text, ha="center", va="center", 
                   color=text_color, fontsize=8, weight='bold')
    
    ax.set_ylabel('True Label', fontsize=10, weight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10, weight='bold')
    ax.set_title(title, fontsize=13, weight='bold', pad=15)
    
    ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, ax


def plot_distribution_comparison(data1, data2, labels, title='Distribution Comparison',
                                xlabel='Value', save_path=None, figsize=(12, 4.5)):
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    datasets = [data1, data2]
    colors = [COLORBLIND_SAFE['blue'], COLORBLIND_SAFE['red']]
    
    for idx, (data, label, color, ax) in enumerate(zip(datasets, labels, colors, axes)):
        n, bins, patches = ax.hist(data, bins=50, alpha=0.6, color=color, 
                                   edgecolor='black', linewidth=0.8, density=True)
        
        from scipy import stats
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2.5, 
               label=f'KDE', linestyle='-')
        
        mean_val = np.mean(data)
        median_val = np.median(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle=':', linewidth=1.5,
                  alpha=0.7, label=f'Median: {median_val:.3f}')
        
        ax.set_xlabel(xlabel, fontsize=11, weight='bold')
        ax.set_ylabel('Density', fontsize=11, weight='bold')
        ax.set_title(label, fontsize=12, weight='bold', pad=10)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        stats_text = f'μ={mean_val:.3f}\nσ={np.std(data):.3f}\nmin={data.min():.3f}\nmax={data.max():.3f}'
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(title, fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, axes


def plot_pca_grid(coords, labels, layers, positions, evr=None,
                 title='PCA Analysis of MLP Activations',
                 save_path=None, figsize=None, show_silhouette=True):
    set_style()
    
    nL, nP = len(layers), len(positions)
    
    if figsize is None:
        figsize = (4.5 * nP, 3.5 * nL)
    
    fig, axes = plt.subplots(nL, nP, figsize=figsize)
    
    if nL == 1 and nP == 1:
        axes = np.array([[axes]])
    elif nL == 1:
        axes = axes.reshape(1, -1)
    elif nP == 1:
        axes = axes.reshape(-1, 1)
    
    colors = {0: COLORS['negative'], 1: COLORS['positive']}
    
    legend_added = False
    
    for i, layer in enumerate(layers):
        for j, pos in enumerate(positions):
            ax = axes[i, j]
            
            if pos not in coords[layer]:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=11)
                continue
            
            Z = coords[layer][pos]
            if hasattr(Z, "detach"):
                Z = Z.detach().cpu().numpy()
            else:
                Z = np.asarray(Z)
            
            ok = np.isfinite(Z).all(axis=1)
            Z = Z[ok]
            y = labels[ok]
            
            if len(Z) < 10:
                ax.axis('off')
                continue
            
            for class_label in [0, 1]:
                mask = y == class_label
                class_name = 'Negative' if class_label == 0 else 'Positive'
                ax.scatter(Z[mask, 0], Z[mask, 1], 
                          c=colors[class_label], alpha=0.5, s=25,
                          edgecolors='none', label=class_name,
                          rasterized=True)  
            
            title_parts = [f'Layer {layer}, Pos {pos}']
            
            if evr is not None and layer in evr and pos in evr[layer]:
                evr_val = sum(evr[layer][pos])
                title_parts.append(f'EVR: {evr_val:.1%}')
            
            if show_silhouette and len(np.unique(y)) == 2:
                try:
                    sil = silhouette_score(Z, y, sample_size=min(400, len(Z)))
                    title_parts.append(f'Sil: {sil:.3f}')
                except:
                    pass
            
            ax.set_title(' | '.join(title_parts), fontsize=9, pad=8)
            
            ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.6)
            ax.set_xlabel('PC1', fontsize=9)
            ax.set_ylabel('PC2', fontsize=9)
            
            ax.set_aspect('equal', adjustable='box')
            
            if not legend_added and i == 0 and j == nP - 1:
                ax.legend(loc='upper right', framealpha=0.9, 
                         fontsize=8, edgecolor='gray')
                legend_added = True
    
    fig.suptitle(title, fontsize=14, weight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, axes


def plot_layer_accuracy_curves(data_dict, title='Linear Probe Accuracy Across Layers',
                               save_path=None, figsize=(10, 5.5)):
    set_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    position_colors = {
        -3: COLORBLIND_SAFE['blue'],
        -2: COLORBLIND_SAFE['green'],
        -1: COLORBLIND_SAFE['red'],
    }
    
    best_overall = {'acc': 0, 'layer': 0, 'pos': 0}
    
    for pos, df in data_dict.items():
        color = position_colors.get(pos, 'gray')
        layers = df['layer'].values
        accs = df['accuracy'].values
        
        best_idx = np.argmax(accs)
        best_layer = layers[best_idx]
        best_acc = accs[best_idx]
        
        if best_acc > best_overall['acc']:
            best_overall = {'acc': best_acc, 'layer': best_layer, 'pos': pos}
        
        ax.plot(layers, accs, marker='o', linewidth=2.5, markersize=7,
               color=color, label=f'Position {pos}', alpha=0.85,
               markeredgecolor='white', markeredgewidth=1)
        
        ax.scatter([best_layer], [best_acc], s=200, marker='*',
                  color=color, edgecolors='black', linewidths=1.5,
                  zorder=10, alpha=0.9)
    
    ax.set_xlabel('Layer', fontsize=12, weight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=13, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=10)
    
    ax.annotate(f"Best: Layer {best_overall['layer']} (pos {best_overall['pos']})\nAcc: {best_overall['acc']:.4f}",
               xy=(best_overall['layer'], best_overall['acc']),
               xytext=(10, -30), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                             color='black', lw=1.5),
               fontsize=9, weight='bold')
    
    y_min, y_max = ax.get_ylim()
    ax.set_ylim([max(0, y_min - 0.02), min(1, y_max + 0.02)])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, ax


def plot_neuron_distribution(layer_counts, title='Distribution of Top Neurons',
                            save_path=None, figsize=(14, 6), top_n=5):
    set_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    layers = sorted([int(k) for k in layer_counts.keys()])
    counts = [layer_counts[str(l)] for l in layers]
    
    top_layers_data = sorted(layer_counts.items(), key=lambda x: -x[1])[:top_n]
    top_layer_ids = {int(l) for l, _ in top_layers_data}
    
    bar_colors = [COLORS['ablation_top'] if l in top_layer_ids 
                  else COLORBLIND_SAFE['blue'] for l in layers]
    
    bars = ax.bar(layers, counts, color=bar_colors, alpha=0.75, 
                  edgecolor='black', linewidth=1.2, width=0.8)
    
    for layer, count, bar in zip(layers, counts, bars):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=7, weight='bold')
    
    mean_count = np.mean(counts)
    ax.axhline(mean_count, color=COLORS['baseline'], linestyle='--', 
              linewidth=2, label=f'Mean: {mean_count:.1f}', alpha=0.7)
    
    ax.set_xlabel('Layer Index', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Top Neurons', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=13, weight='bold', pad=15)
    ax.grid(True, alpha=0.25, axis='y', linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['ablation_top'], edgecolor='black', 
              label=f'Top {top_n} Layers', alpha=0.75),
        Patch(facecolor=COLORBLIND_SAFE['blue'], edgecolor='black',
              label='Other Layers', alpha=0.75),
    ]
    ax.legend(handles=legend_elements, loc='center', 
             framealpha=0.95, edgecolor='gray', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, ax


def plot_ablation_study(results_top, results_random, baseline_acc,
                       title='Ablation Study: Impact of Neuron Removal',
                       save_path=None, figsize=(10, 5)):
    set_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_top, acc_top = zip(*results_top)
    n_rand, acc_rand = zip(*results_random)
    
    ax.plot(n_top, acc_top, marker='o', linewidth=3, markersize=9,
           color=COLORS['ablation_top'], label='Top Neurons Removed',
           markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    
    ax.plot(n_rand, acc_rand, marker='s', linewidth=3, markersize=9,
           color=COLORS['ablation_random'], label='Random Neurons Removed',
           markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    
    ax.axhline(baseline_acc, color=COLORS['baseline'], linestyle='--',
              linewidth=2.5, label=f'Baseline: {baseline_acc:.4f}', alpha=0.8)
    
    ax.fill_between(n_top, acc_top, acc_rand, alpha=0.15, 
                    color=COLORS['ablation_top'],
                    label='Performance Gap')
    
    max_removal_idx = len(n_top) - 1
    final_gap = acc_rand[max_removal_idx] - acc_top[max_removal_idx]
    
    ax.annotate(f'Final Gap: {final_gap:.4f}',
               xy=(n_top[max_removal_idx], acc_top[max_removal_idx]),
               xytext=(-50, 30), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
               fontsize=10, weight='bold')
    
    ax.set_xlabel('Number of Neurons Removed', fontsize=12, weight='bold')
    ax.set_ylabel('Model Accuracy', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=13, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=10)
    
    ax.set_ylim([min(min(acc_top), min(acc_rand)) - 0.02, baseline_acc + 0.01])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, ax

