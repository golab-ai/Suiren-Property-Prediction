"""
Molecular Property Prediction Model using Suiren Pre-trained and Fine-tuned GNN

This module implements a transfer learning approach for molecular property prediction
by combining pre-trained GNN embeddings with fine-tuned layers. The architecture
supports both regression and classification tasks.

Author: JunyiAn
Date: 2026-02-28
"""

import torch
import torch.nn as nn
from models.graph_NN import GNN
from torch_geometric.nn import global_add_pool


class PredictModel2D(torch.nn.Module):
    """
    2D Molecular Property Predictor with Transfer Learning
    
    This model employs a two-stage approach:
    1. Pre-trained GNN: Extracts general molecular representations
    2. Fine-tuned GNN: Refines representations for specific property prediction
    
    The overall pipeline:
    - Extract node embeddings from pre-trained model (all layers)
    - Process through fine-tuned model with pre-trained embedding conditioning
    - Project node embeddings to latent space
    - Aggregate to molecular level (graph pooling)
    - Generate final prediction (regression or classification)
    
    Args:
        pretrain_num_layer (int): Number of layers in pre-trained model
        finetune_num_layer (int): Number of layers in fine-tuned model
        pretrain_embed_dim (int): Embedding dimension of pre-trained model
        finetune_embed_dim (int): Embedding dimension of fine-tuned model
        drop_ratio (float): Dropout ratio for regularization (default: 0.1)
        d_proj (int): Projection dimension for latent space (default: 256)
        class_num (int): Number of classes for classification (default: 2)
        class_flag (bool): Whether to perform classification (True) or regression (False)
                          (default: False)
    
    Attributes:
        pretrain_model (GNN): Pre-trained GNN model (returns all layer outputs)
        finetune_model (GNN): Fine-tune GNN model
        proj_2d (Sequential): Projection network for node embeddings to latent space
        proj_2d_glob (Sequential): Global predictor from molecular representation
    """
    
    def __init__(self, 
                 pretrain_num_layer,
                 finetune_num_layer,
                 pretrain_embed_dim,
                 finetune_embed_dim,
                 drop_ratio=0.1,
                 d_proj=256, 
                 class_num=2, 
                 class_flag=False):
        
        super().__init__()

        # Average number of atoms used for normalization
        self.avg_atom = 35.2160
        
        # ========================================================================
        # Pre-trained and Fine-tuned Models
        # ========================================================================
        # Pre-trained model extracts embeddings from all layers
        self.pretrain_model = GNN(
            num_layer=pretrain_num_layer,
            emb_dim=pretrain_embed_dim,
            drop_ratio=0.0,
            output_type="layers"  # Return outputs from all layers
        )
        
        # Fine-tuned model refines representations with conditioning from pre-trained model
        self.finetune_model = GNN(
            num_layer=finetune_num_layer,
            emb_dim=finetune_embed_dim,
            drop_ratio=drop_ratio,
            model_mode="finetune",
            pretrain_emb_dim=pretrain_embed_dim,
            pretrain_num_layer=pretrain_num_layer
        )
        
        # ========================================================================
        # Projection Networks
        # ========================================================================
        # Node embedding projection to latent space
        self.proj_2d = nn.Sequential(
            nn.Linear(finetune_embed_dim, d_proj),
            nn.SiLU(),
            nn.Linear(d_proj, d_proj),
            nn.SiLU(),
            nn.Linear(d_proj, d_proj),
        )

        # Global prediction head (molecular level)
        if class_flag:
            # Classification task
            self.proj_2d_glob = nn.Sequential(
                nn.Linear(d_proj, d_proj),
                nn.SiLU(),
                nn.Linear(d_proj, class_num)
            )
        else:
            # Regression task
            self.proj_2d_glob = nn.Sequential(
                nn.Linear(d_proj, d_proj),
                nn.SiLU(),
                nn.Linear(d_proj, 1)
            )

    def forward(self, data):
        """
        Forward pass of the model.
        
        Args:
            data: PyTorch Geometric Data object with attributes:
                - x (Tensor): Node features
                - edge_index (LongTensor): Local graph edges
                - edge_index_all (LongTensor): Full-connect graph edges
                - edge_attr (Tensor): Edge attributes
                - batch (LongTensor): Batch assignment for nodes
        
        Returns:
            Tensor: Molecular level predictions
                - Shape [batch_size, class_num] for classification
                - Shape [batch_size, 1] for regression
        """
        
        # ========================================================================
        # Stage 1: Extract Pre-trained Embeddings
        # ========================================================================
        # Get embeddings from all layers of pre-trained model
        reference_2d = self.pretrain_model(
            node_atom=data.x,
            edge_index=data.edge_index,
            edge_index_all=data.edge_index_all,
            edge_attr=data.edge_attr,
            batch=data.batch
        )
        
        # ========================================================================
        # Stage 2: Fine-tune with Pre-trained Embeddings
        # ========================================================================
        # Process through fine-tuned model with pre-trained embedding conditioning
        outputs_2d = self.finetune_model(
            node_atom=data.x,
            edge_index=data.edge_index,
            edge_index_all=data.edge_index_all,
            edge_attr=data.edge_attr,
            batch=data.batch,
            extra_embedding=reference_2d
        )
        
        # ========================================================================
        # Stage 3: Node-level Projection
        # ========================================================================
        # Project node embeddings to latent space
        outputs_2d = self.proj_2d(outputs_2d)
        
        # ========================================================================
        # Stage 4: Graph-level Pooling and Normalization
        # ========================================================================
        # Aggregate node representations to molecular level using scatter mean
        outputs_2d = global_add_pool(outputs_2d, data.batch) / self.avg_atom
        
        # ========================================================================
        # Stage 5: Molecular Prediction
        # ========================================================================
        # Generate final predictions
        outputs_2d_final = self.proj_2d_glob(outputs_2d)
        
        return outputs_2d_final


def standard_finetune(class_num=2, class_flag=False):
    """
    Factory function to create a standard fine-tuned model with predefined architecture.
    
    This function returns a PredictModel2D with commonly used hyperparameters
    optimized for molecular property prediction.
    
    Args:
        class_num (int): Number of classes for classification task (default: 2)
        class_flag (bool): Whether to perform classification (True) or regression (False)
                          (default: False)
    
    Returns:
        PredictModel2D: Initialized model ready for fine-tuning
    
    Model Architecture:
        - Pre-trained: 12 layers with 256-dim embeddings
        - Fine-tune: 16 layers with 256-dim embeddings
        - Projection: 256-dim latent space
        - Dropout: 0.1
    """
    return PredictModel2D(
        pretrain_num_layer=12,
        finetune_num_layer=12,
        pretrain_embed_dim=256,
        finetune_embed_dim=256,
        drop_ratio=0.1,
        d_proj=256,
        class_num=class_num,
        class_flag=class_flag,
    )
