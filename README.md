# Clustering scRNA-seq

## Setup

```bash
git clone https://github.com/nveshaan/scRNA_cluster.git
cd scRNA_cluster
uv sync
```

## Run

```bash
python main.py --multirun dataset=pbmc,bmmc method=pca,scvi,linear_scvi 'latent_dim=range(10,110,10)'
```

## Acknowledgements

This project builds on ideas and implementations from prior tools and publications, including:

- Stuart, T., Butler, A., Hoffman, P., Hafemeister, C., Papalexi, E., Mauck, W. M., Hao, Y., Stoeckius, M., Smibert, P., & Satija, R. (2019). Comprehensive Integration of Single-Cell Data. Cell. https://doi.org/10.1016/j.cell.2019.05.031
- Gayoso, A., et al. (2022). scvi-tools: a library for deep probabilistic analysis of single-cell omics data. Nature Biotechnology. https://doi.org/10.1038/s41587-022-01270-x

These projects and their documentation were essential references when implementing the Seurat-like workflows and integrating scVI/LinearSCVI models.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.