import os
import sys
import anndata
import pandas as pd
import scanpy as sc
import dgl
import scipy
import torch
import collections
from scipy.sparse import csr_matrix
from pathlib import Path
import numpy as np
from torch.fx.experimental.unification.multipledispatch.dispatcher import source
from torch_geometric.data import Data


def get_id_2_gene(species_data_path):
    data_path = Path(species_data_path)
    data_files = data_path.glob('*matrix*.csv')
    genes = None
    for file in data_files:
        data = pd.read_csv(file, dtype=str, header=0).values[:, 0]

        if genes is None:
            genes = set(data)
        else:
            genes = genes | set(data)

    id2gene = list(genes)
    id2gene.sort()
    return id2gene


def dgl_to_pyg(dgl_graph):
    edge_index = torch.stack(dgl_graph.edges(), dim=0)

    x = dgl_graph.ndata['features']
    y = dgl_graph.ndata['label']

    edge_index = edge_index.to(torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def get_id_2_label_and_label_statistics(disease_data_path):
    data_path = Path(disease_data_path)
    cell_files = list(data_path.glob('*label*.csv'))
    cell_types = set()
    cell_type_list = []

    for file in cell_files:
        df = pd.read_csv(file, dtype=str)
        df['celltype'] = df['Celltype (major-lineage)'].map(str.strip)  # 去除细胞类型名称前后的空格
        cell_types.update(df['celltype'])
        cell_type_list.extend(df['celltype'].tolist())

    id2label = sorted(cell_types)
    label_statistics = dict(collections.Counter(cell_type_list))

    return id2label, label_statistics


def load_expression_data(data_path, label2id):
    matrices = []
    all_labels = []
    num_cells = 0
    adatas = []
    data_path = Path(data_path)
    matrix_files = list(data_path.glob('*matrix*.csv'))
    for matrix_file in matrix_files:
        label_file_name = matrix_file.name.replace('matrix', 'label')
        type_file = data_path / label_file_name
        cell2type = pd.read_csv(type_file, dtype=str)
        cell2type['celltype'] = cell2type['Celltype (major-lineage)'].map(str.strip)
        cell2type['id'] = cell2type['celltype'].map(label2id)
        all_labels += cell2type['id'].tolist()
        df = pd.read_csv(matrix_file, index_col=0)
        df = df.transpose(copy=True)

        adata = anndata.AnnData(X=df)
        matrices.append(df)
        adatas.append(adata)

        num_cells += len(df)

    return matrices, all_labels, num_cells, adatas


def select_pathway_gene(pathway_name):
    gene_set = set()

    parts = pathway_name.split('_')
    field_name = '_'.join(parts[2:])
    database_name = parts[0]

    with open(f'../Pathway/split pathway/{database_name} Pathway Split Package/{database_name}_Human_{field_name}.txt', 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            genes = columns[2:]
            gene_set.update(genes)

    return gene_set


def select_gene(adata, params):
    if params.selected_gene == 'hvg':
        sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable]
    else:
        gene_set = select_pathway_gene(params.selected_gene)
        adata.var_names = adata.var_names.str.lower()
        genes_all = adata.var_names
        genes_keep = genes_all[genes_all.isin(gene_set)]
        adata = adata[:, genes_keep]

    return adata


def csv_load_data_together(self,params):
    dense_dim = params.dense_dim

    source_disease = params.source_disease
    target_disease = params.target_disease
    device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    train_disease_data_path = '../Datasets/' + source_disease
    target_disease_data_path = '../Datasets/' + target_disease

    if not os.path.exists(train_disease_data_path):
        sys.exit(f"Source data path '{train_disease_data_path}' does not exist.")
    if not os.path.exists(target_disease_data_path):
        sys.exit(f"Target data path '{target_disease_data_path}' does not exist.")


    source_id2gene = get_id_2_gene(train_disease_data_path)
    source_id2label, label_statistics = get_id_2_label_and_label_statistics(train_disease_data_path)

    target_id2gene = get_id_2_gene(target_disease_data_path)
    target_id2label, target_label_statistics = get_id_2_label_and_label_statistics(target_disease_data_path)

    source_num_genes = len(source_id2gene)
    target_num_genes = len(target_id2gene)
    source_num_labels = len(source_id2label)
    target_num_labels = len(target_id2label)

    label2id = {label: idx for idx, label in enumerate(source_id2label)}

    print(f"The build graph contains {source_num_genes} genes with {source_num_labels} labels supported.")
    print(f"The build graph contains {target_num_genes} genes with {target_num_labels} labels supported.")

    source_matrices, source_all_labels, source_num_cells, adatas_source = load_expression_data(
        train_disease_data_path, label2id)
    target_matrices, target_all_labels, target_num_cells, adatas_target = load_expression_data(
        target_disease_data_path,label2id)

    for adata in adatas_source:
        adata = select_gene(adata, params)
        adata = adata.copy()
        sc.pp.normalize_total(adata, exclude_highly_expressed=True, target_sum=10000)
        sc.pp.log1p(adata)

    for adata in adatas_target:
        adata = select_gene(adata, params)
        adata = adata.copy()
        sc.pp.normalize_total(adata, exclude_highly_expressed=True, target_sum=10000)

    adata_all = sc.concat(adatas_source + adatas_target, join='inner', label='batch', keys=['source', 'target'],
                          index_unique=None)

    adata_all.obs_names_make_unique(join="-")

    if params.use_pca:
        sc.pp.pca(adata_all, n_comps=dense_dim)

    source_cell_feat = adata_all[adata_all.obs["batch"] == "source"].obsm['X_pca']
    target_cell_feat = adata_all[adata_all.obs["batch"] == "target"].obsm['X_pca']

    if scipy.sparse.issparse(source_cell_feat):
        source_cell_feat = source_cell_feat.toarray()

    if scipy.sparse.issparse(target_cell_feat):
        target_cell_feat = target_cell_feat.toarray()

    source_cell_feat = torch.from_numpy(source_cell_feat).float()
    target_cell_feat = torch.from_numpy(target_cell_feat).float()

    k = params.k
    source_k = max(1, int(source_cell_feat.shape[0] * k))
    target_k = max(1, int(target_cell_feat.shape[0] * k))

    source_graph = dgl.knn_graph(x=source_cell_feat, k=source_k, algorithm='kd-tree', dist='cosine')
    source_graph.ndata['features'] = source_cell_feat.float()
    source_graph.ndata['label'] = torch.tensor(source_all_labels)
    source_data = dgl_to_pyg(source_graph)

    target_graph = dgl.knn_graph(x=target_cell_feat, k=target_k, algorithm='kd-tree', dist='cosine')
    target_graph.ndata['features'] = target_cell_feat.float()
    target_graph.ndata['label'] = torch.tensor(target_all_labels)
    target_data = dgl_to_pyg(target_graph)

    source_data.to(device)
    target_data.to(device)

    return source_num_cells, source_num_genes, source_num_labels, source_data, target_num_cells, target_num_genes, target_num_labels, target_data, label2id