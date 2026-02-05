import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_run(prefix):
    with open(prefix + ".json", "r") as f:
        data = json.load(f)

    layer_counts = {int(k): int(v) for k, v in data["layer_counts"].items()}
    top_neurons = data["top_neurons"]
    meta = data.get("meta", {})

    abs_coef = np.array([abs(n["coef"]) for n in top_neurons], dtype=np.float32)
    importance = np.array([n.get("importance", abs(n["coef"])) for n in top_neurons], dtype=np.float32)

    return layer_counts, top_neurons, abs_coef, importance, meta


def plot_pca_2d(X, y, layer, position, title=None, save_path=None):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[y==0, 0], X_2d[y==0, 1], alpha=0.5, label="Negative", s=30)
    plt.scatter(X_2d[y==1, 0], X_2d[y==1, 1], alpha=0.5, label="Positive", s=30)
    
    if title is None:
        title = f"PCA: Layer {layer}, Position {position}"
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_layer_accuracy(layer_acc, title="Linear Probe Accuracy by Layer", save_path=None):
    layers = sorted(layer_acc.keys())
    accs = [layer_acc[l] for l in layers]
    
    plt.figure(figsize=(10, 5))
    plt.plot(layers, accs, marker='o', linewidth=2, markersize=6)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    best_layer = max(layer_acc, key=layer_acc.get)
    plt.axvline(best_layer, color='r', linestyle='--', alpha=0.5, 
                label=f'Best: Layer {best_layer} ({layer_acc[best_layer]:.3f})')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_neuron_distribution(layer_counts, title="Top Neurons Distribution by Layer", save_path=None):
    layers = sorted(layer_counts.keys())
    counts = [layer_counts[l] for l in layers]
    
    plt.figure(figsize=(12, 5))
    plt.bar(layers, counts, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Number of Top Neurons", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ablation_comparison(results_top, results_random, baseline_acc, title="Ablation Study: Top vs Random Neurons", save_path=None):
    n_removed_top, acc_top = zip(*results_top)
    n_removed_rand, acc_rand = zip(*results_random)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_removed_top, acc_top, marker='o', linewidth=2, 
             label='Remove Top Neurons', color='red')
    plt.plot(n_removed_rand, acc_rand, marker='s', linewidth=2, 
             label='Remove Random Neurons', color='blue')
    plt.axhline(baseline_acc, color='green', linestyle='--', 
                label=f'Baseline ({baseline_acc:.3f})')
    
    plt.xlabel("Number of Neurons Removed", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pca_grid_from_coords(
    coords,
    labels,
    layers=None,
    positions=None,
    evr=None,
    title="PCA of MLP Activations (Layer × Position)",
    save_path=None,
    show_silhouette=False,   
    point_size=12,
    alpha=0.35,
    share_limits=True,
    pad=0.05,
    col_width=4.0,
    row_height=1.25,
    show=True,
    close=False,
):

    available_layers = sorted(coords.keys())
    if layers is None:
        layers = available_layers
    else:
        missing_layers = [l for l in layers if l not in coords]
        layers = [l for l in layers if l in coords]
        if missing_layers:
            print("Warning: requested layers missing:", missing_layers)

    if len(layers) == 0:
        raise ValueError("No valid layers to plot.")

    available_positions = set()
    for l in layers:
        available_positions.update(coords[l].keys())
    available_positions = sorted(list(available_positions))

    if positions is None:
        positions = available_positions
    else:
        missing_pos = [p for p in positions if p not in available_positions]
        positions = [p for p in positions if p in available_positions]
        if missing_pos:
            print("Warning: requested positions missing:", missing_pos)

    if len(positions) == 0:
        raise ValueError("No valid positions to plot.")

    nL, nP = len(layers), len(positions)
    fig, axes = plt.subplots(nL, nP, figsize=(col_width * nP, row_height * nL))

    if nL == 1 and nP == 1:
        axes = np.array([[axes]])
    elif nL == 1:
        axes = axes.reshape(1, -1)
    elif nP == 1:
        axes = axes.reshape(-1, 1)

    col_limits = None
    if share_limits:
        col_limits = {}
        for pos in positions:
            xs, ys = [], []
            for layer in layers:
                if pos not in coords[layer]:
                    continue
                Z = coords[layer][pos]
                if hasattr(Z, "detach"):
                    Z = Z.detach().cpu().numpy()
                else:
                    Z = np.asarray(Z)

                ok = np.isfinite(Z).all(axis=1)
                Z = Z[ok]
                if len(Z) == 0:
                    continue
                xs.append(Z[:, 0])
                ys.append(Z[:, 1])

            if len(xs) == 0:
                col_limits[pos] = None
                continue

            x_all = np.concatenate(xs)
            y_all = np.concatenate(ys)

            xpad = pad * (x_all.max() - x_all.min() + 1e-9)
            ypad = pad * (y_all.max() - y_all.min() + 1e-9)
            col_limits[pos] = (
                (x_all.min() - xpad, x_all.max() + xpad),
                (y_all.min() - ypad, y_all.max() + ypad),
            )

    if show_silhouette:
        from sklearn.metrics import silhouette_score

    legend_added = False

    for i, layer in enumerate(layers):
        for j, pos in enumerate(positions):
            ax = axes[i, j]

            if pos not in coords[layer]:
                ax.axis("off")
                ax.set_title(f"L{layer} P{pos}\n(missing)", fontsize=9)
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
                ax.axis("off")
                ax.set_title(f"L{layer} P{pos}\n(insufficient)", fontsize=9)
                continue

            ax.scatter(Z[y == 0, 0], Z[y == 0, 1], alpha=alpha, s=point_size, label="Negative")
            ax.scatter(Z[y == 1, 0], Z[y == 1, 1], alpha=alpha, s=point_size, label="Positive")

            title_parts = [f"L{layer} P{pos}"]

            if evr is not None and layer in evr and pos in evr[layer]:
                title_parts.append(f"EVR={sum(evr[layer][pos]):.2%}")

            if show_silhouette:
                uniq = np.unique(y)
                if len(uniq) == 2 and min((y == 0).sum(), (y == 1).sum()) >= 2:
                    try:
                        title_parts.append(f"Sil={silhouette_score(Z, y):.3f}")
                    except Exception:
                        pass

            ax.set_title(", ".join(title_parts), fontsize=9)
            ax.grid(True, alpha=0.25)

            if share_limits and col_limits is not None:
                lim = col_limits.get(pos, None)
                if lim is not None:
                    ax.set_xlim(lim[0])
                    ax.set_ylim(lim[1])
                else:
                    ax.autoscale(enable=True, axis="both", tight=True)
            else:
                xmin, xmax = Z[:, 0].min(), Z[:, 0].max()
                ymin, ymax = Z[:, 1].min(), Z[:, 1].max()
                xpad = pad * (xmax - xmin + 1e-9)
                ypad = pad * (ymax - ymin + 1e-9)
                ax.set_xlim(xmin - xpad, xmax + xpad)
                ax.set_ylim(ymin - ypad, ymax + ypad)

            if i != nL - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

            if not legend_added:
                handles, lbls = ax.get_legend_handles_labels()
                if handles:
                    fig.legend(handles, lbls, loc="upper right", fontsize=10)
                    legend_added = True

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.98, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=250, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return fig
