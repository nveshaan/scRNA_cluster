import scanpy as sc
import scvi

def seurat_like_clustering(adata, cfg):
    """
    Simulates Seurat's graph-based clustering using Scanpy (Leiden on PCA KNN graph).
    """
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, svd_solver='arpack', n_comps=cfg.latent_dim)
    
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=cfg.latent_dim)
    
    sc.tl.umap(adata)
    adata.obsm['X_umap_pca'] = adata.obsm['X_umap'].copy()
    
    sc.tl.leiden(adata, resolution=0.8, key_added='seurat_leiden')
    
    return adata

def seurat_scvi_clustering(adata, cfg):
    """
    Modified Seurat workflow that substitutes PCA with scVI latent representations.
    Assumes adata.layers['counts'] contains unnormalized raw counts.
    """
        
    print("Setting up scVI on 'counts' layer...")
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    
    print("Training scVI model...")
    model = scvi.model.SCVI(adata, n_latent=cfg.latent_dim)
    model.train(max_epochs=cfg.epochs, accelerator=cfg.device, devices=1, train_size=0.9, validation_size=0.1, check_val_every_n_epoch=1) 
    
    adata.obsm['X_scvi'] = model.get_latent_representation()
    
    print("Running Leiden clustering on scVI latent space...")
    sc.pp.neighbors(adata, use_rep='X_scvi', n_neighbors=10)
    
    sc.tl.umap(adata)
    adata.obsm['X_umap_scvi'] = adata.obsm['X_umap'].copy()
    
    sc.tl.leiden(adata, resolution=0.8, key_added='seurat_scvi_leiden')
    
    return adata, model

def seurat_linear_scvi_clustering(adata, cfg):
    """
    Modified Seurat workflow that substitutes PCA with LinearSCVI representations.
    LinearSCVI ensures the decoder maps linearly back to gene space, allowing interpretability.
    """
        
    print("Setting up LinearSCVI on 'counts' layer...")
    scvi.model.LinearSCVI.setup_anndata(adata, layer="counts")
    
    print("Training LinearSCVI model...")
    model = scvi.model.LinearSCVI(adata, n_latent=cfg.latent_dim)
    model.train(max_epochs=cfg.epochs, accelerator=cfg.device, devices=1, train_size=0.9, validation_size=0.1, check_val_every_n_epoch=1) 
    
    adata.obsm['X_linear_scvi'] = model.get_latent_representation()
    
    print("Running Leiden clustering on LinearSCVI latent space...")
    sc.pp.neighbors(adata, use_rep='X_linear_scvi', n_neighbors=10)
    
    sc.tl.umap(adata)
    adata.obsm['X_umap_linear_scvi'] = adata.obsm['X_umap'].copy()
    
    sc.tl.leiden(adata, resolution=0.8, key_added='seurat_linear_scvi_leiden')
    
    return adata, model