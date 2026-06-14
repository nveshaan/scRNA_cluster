import warnings
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.metrics import adjusted_rand_score
import torch
import numpy as np

from src.seurat import (
    seurat_like_clustering,
    seurat_scvi_clustering,
    seurat_linear_scvi_clustering,
)

warnings.filterwarnings("ignore")

OmegaConf.register_new_resolver(
    "device",
    lambda: "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu",
    replace=True,
)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    for path in cfg.paths.values():
        (run_dir / path).mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.run_dir = str(run_dir)

    if cfg.dataset == "pbmc":
        print("Loading PBMC 3k dataset...")
        adata = sc.datasets.pbmc3k()
        adata_processed = sc.datasets.pbmc3k_processed()
        common_cells = adata.obs_names.intersection(adata_processed.obs_names)
        adata = adata[common_cells].copy()
        adata.obs["ground_truth"] = adata_processed.obs.loc[common_cells, "louvain"]

    elif cfg.dataset == "bmmc":
        print("Loading BMMC dataset...")
        adata = sc.read_h5ad("data/bmmc.h5ad")
        adata.obs["ground_truth"] = adata.obs[
            "cell_type"
        ]  # Assuming 'leiden' column has the original clusters
        
    else:
        print("Loading simulated dataset...")
        adata = sc.datasets.blobs(
            n_cells=3000, n_genes=2000, centers=5, cluster_std=0.5, random_state=42
        )
        adata.obs["ground_truth"] = adata.obs[
            "leiden"
        ]  # 'leiden' is the default cluster label for blobs

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata.layers["counts"] = adata.X.copy()

    def save_embedding_only(adata, basis, label_key, run_dir, cfg, ari):
        plt.style.use("fivethirtyeight")
        emb = adata.obsm.get(basis)
        if emb is None:
            raise KeyError(f"Embedding basis '{basis}' not found in adata.obsm")
        # labels -> categorical codes for coloring; no legend drawn
        if label_key in adata.obs.columns:
            labels = adata.obs[label_key].astype(str)
        else:
            labels = adata.obs.iloc[:, 0].astype(str)
        codes = labels.astype("category").cat.codes.values
        n_clusters = int(labels.nunique())
        # use seaborn palette for categorical coloring
        palette = sns.color_palette("husl", n_clusters)
        colors = [palette[int(c)] for c in codes]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=8, alpha=0.9)
        # remove ticks and labels, keep the box (spines)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        for spine in ax.spines.values():
            spine.set_visible(True)
        fname = f"{cfg.dataset}_{cfg.method}_ari{ari:.3f}_nclust{n_clusters}.png"
        outpath = Path(run_dir) / "viz" / fname
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight", dpi=150, pad_inches=0.1)
        plt.close(fig)

    if cfg.method == "pca":
        print("Running PCA...")
        adata, trajectory = seurat_like_clustering(adata, cfg)
        ari = adjusted_rand_score(adata.obs["seurat_leiden"], adata.obs["ground_truth"])
        save_embedding_only(
            adata,
            basis="X_umap_pca",
            label_key="seurat_leiden",
            run_dir=run_dir,
            cfg=cfg,
            ari=ari,
        )
    elif cfg.method == "scvi":
        print("Running scVI...")
        adata, model, trajectory = seurat_scvi_clustering(adata, cfg)
        ari = adjusted_rand_score(
            adata.obs["seurat_scvi_leiden"], adata.obs["ground_truth"]
        )
        save_embedding_only(
            adata,
            basis="X_umap_scvi",
            label_key="seurat_scvi_leiden",
            run_dir=run_dir,
            cfg=cfg,
            ari=ari,
        )
        model.save(run_dir / "checkpoints" / "scvi_model.pt")
    elif cfg.method == "linear_scvi":
        print("Running linear scVI...")
        adata, model, trajectory = seurat_linear_scvi_clustering(adata, cfg)
        ari = adjusted_rand_score(
            adata.obs["seurat_linear_scvi_leiden"], adata.obs["ground_truth"]
        )
        save_embedding_only(
            adata,
            basis="X_umap_linear_scvi",
            label_key="seurat_linear_scvi_leiden",
            run_dir=run_dir,
            cfg=cfg,
            ari=ari,
        )
        model.save(run_dir / "checkpoints" / "linear_scvi_model.pt")
    else:
        raise ValueError(f"Unknown method: {cfg.method}")
    
    if trajectory is not None:
        np.save(Path(run_dir) / "trajectory.npy", trajectory)

    print(f"Finished {cfg.method} on {cfg.dataset} with ARI ({cfg.shift}): {ari:.4f}")


if __name__ == "__main__":
    main()
