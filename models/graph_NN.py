"""
Molecular Graph Neural Network (GNN) Models for 2D Property Prediction

This module provides implementation of Graph Attention Network (GAT) based GNN
for molecular property prediction tasks. It includes pre-training and fine-tuning
capabilities with multi-head attention mechanisms.

Author: JunyiAn
Date: 2026-02-28
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F

# ============================================================================
# Global Configuration Constants for Molecular Features
# ============================================================================
NUM_ATOM_TYPE = 100  # Including extra mask tokens for pre-training
NUM_CHIRALITY_TAG = 12  # Chirality information tags

NUM_BOND_TYPE = 28  # Including aromatic and self-loop edges, and extra masked tokens
NUM_BOND_DIRECTION = 10  # Bond direction types 

class GATConv(nn.Module):
    """
    Multi-head Graph Attention Convolution Layer with Full-Connect Graph Processing
    
    This layer implements a full-connect graph attention mechanism that:
    1. Processes messages through local graph edges
    2. Applies attention on the full-connect graph
    3. Aggregates messages with residual connections
    4. Uses layer normalization and activation functions for stability
    
    Args:
        emb_dim (int): Embedding dimension of node features
        heads (int): Number of attention heads (default: 2)
        drop_ratio (float): Dropout ratio for regularization (default: 0.0)
        negative_slope (float): Negative slope for LeakyReLU in attention (default: 0.2)
        reduce_ratio (int): Channel reduction ratio for dimensionality reduction (default: 16)
    """
    
    def __init__(self, emb_dim, heads=2, drop_ratio=0.0, negative_slope=0.2, reduce_ratio=16):
        super(GATConv, self).__init__()

        # Store configuration parameters
        self.emb_dim = emb_dim
        self.heads = heads
        self.reduce_ratio = reduce_ratio
        self.negative_slope = negative_slope

        # Activation function and normalization layers
        self.act = nn.SiLU()
        self.norm1 = nn.LayerNorm(emb_dim)  # Normalization for local message computation
        self.norm2 = nn.LayerNorm(emb_dim)  # Normalization for full-connect message computation
        self.norm3 = nn.LayerNorm(emb_dim)  # Normalization for attention mechanism
        self.norm4 = nn.LayerNorm(emb_dim)  # Normalization before feed-forward network

        # Linear projections for local graph message passing
        self.weight_linear1 = torch.nn.Linear(emb_dim, heads * emb_dim)  # Source node projection
        self.weight_linear2 = torch.nn.Linear(emb_dim, heads * emb_dim)  # Target node projection

        # Dimensionality reduction for full-connect graph processing
        self.reduce_channel = emb_dim // self.reduce_ratio
        self.weight_linear3 = torch.nn.Linear(emb_dim, heads * self.reduce_channel)  # Source node in FC graph
        self.weight_linear4 = torch.nn.Linear(emb_dim, heads * self.reduce_channel)  # Target node in FC graph
        self.weight_linear5 = torch.nn.Linear(self.reduce_channel, emb_dim)  # Projection back to full dimension

        # Attention mechanism projections for full-connect graph
        self.weight_linear_a1 = torch.nn.Linear(emb_dim, heads)  # Source attention weights
        self.weight_linear_a2 = torch.nn.Linear(emb_dim, heads)  # Target attention weights
        self.fc_attn = torch.nn.Linear(heads, heads)  # Attention aggregation

        # Edge attribute embeddings
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, heads * emb_dim)  # Bond type embedding
        self.edge_embedding2 = torch.nn.Embedding(5, heads * emb_dim)  # Bond stereo embedding
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_DIRECTION, heads * emb_dim)  # Bond direction embedding

        # Feed-forward network for residual update
        self.FFN = nn.Sequential(nn.Linear(emb_dim, 4 * emb_dim),
                                nn.SiLU(),
                                nn.Linear(emb_dim * 4, emb_dim))
        
        # Scaling factor for full-connect graph contribution
        self.scale = 0.01

        # Dropout layer for regularization
        self.dropout = torch.nn.Dropout(drop_ratio)

        # Initialize embedding weights with Xavier uniform distribution
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

    def forward(self, x, edge_index, edge_index_all, edge_attr):
        """
        Forward pass of GAT convolution with full-connect graph processing.
        
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, emb_dim]
            edge_index (LongTensor): Edge indices for local graph of shape [2, num_edges]
            edge_index_all (LongTensor): Edge indices for full-connect graph of shape [2, num_edges_full]
            edge_attr (Tensor): Edge attributes of shape [num_edges, num_attr_features]
        
        Returns:
            Tensor: Updated node representations of shape [num_nodes, emb_dim]
        """
        # Get tensor dimensions
        num_node, n_dim = x.shape
        shortcut1 = x  # Store input for residual connection

        # ========================================================================
        # Stage 1: Local Graph Message Passing
        # ========================================================================
        x_norm = self.norm1(x)

        # Add self-loops to the edge index
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # Create edge attributes for self-loop edges
        self_loop_attr = torch.zeros(x.size(0), 3, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 25  # Bond type ID 25 is reserved for self-loops
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Embed edge attributes using multiple embedding layers
        edge_emb = (self.edge_embedding1(edge_attr[:, 0].long()) + 
                   self.edge_embedding2(edge_attr[:, 1].long()) + 
                   self.edge_embedding3(edge_attr[:, 2].long()))
        edge_attr_embedded = edge_emb.view(-1, self.heads, self.emb_dim)

        # Compute source and target node representations for messages
        x_src = self.weight_linear1(x_norm).view(-1, self.heads, self.emb_dim)
        x_tgt = self.weight_linear2(x_norm).view(-1, self.heads, self.emb_dim)
        
        x_i = x_src[edge_index[0]]  # Source nodes for all edges
        x_j = x_tgt[edge_index[1]]  # Target nodes for all edges
        
        # Compute messages with edge embeddings and activation
        message = self.act(x_i + x_j + edge_attr_embedded)
        
        # Combine multi-head messages
        message_combined = message.mean(dim=1)

        # Aggregate messages to target nodes
        new_x = torch.zeros_like(x)
        new_x = new_x.index_add_(0, edge_index[1], message_combined)

        # ========================================================================
        # Stage 2: Full-Connect Graph Processing with Attention
        # ========================================================================
        x_norm2 = self.norm2(x)
        
        # Compute dimensionality-reduced representations
        x_src_fc = self.weight_linear3(x_norm2).view(-1, self.heads, self.reduce_channel)[edge_index_all[0]]
        x_tgt_fc = self.weight_linear4(x_norm2).view(-1, self.heads, self.reduce_channel)[edge_index_all[1]]
        
        # Compute full-connect graph messages
        fc_message = self.act(x_src_fc + x_tgt_fc)
        fc_message = self.weight_linear5(fc_message)

        # ========================================================================
        # Stage 3: Attention Mechanism on Full-Connect Graph
        # ========================================================================
        x_norm3 = self.norm3(x)
        
        # Compute attention logits
        attn_src = self.weight_linear_a1(x_norm3)[edge_index_all[0]]
        attn_tgt = self.weight_linear_a2(x_norm3)[edge_index_all[1]]
        fc_attn_logits = F.leaky_relu(attn_src + attn_tgt, self.negative_slope)
        fc_attn_logits = self.fc_attn(fc_attn_logits)
        
        # Normalize attention weights using softmax
        attn_weights = softmax(fc_attn_logits, edge_index_all[1]).view(-1, self.heads, 1)

        # Apply attention weights to messages
        fc_message_weighted = fc_message * attn_weights
        fc_message_final = fc_message_weighted.mean(dim=1)
        
        # Aggregate attention-weighted messages to target nodes
        new_x_fc = torch.zeros(num_node, self.emb_dim, dtype=x.dtype, device=x.device)
        new_x_fc = new_x_fc.index_add_(0, edge_index_all[1], fc_message_final)

        # ========================================================================
        # Stage 4: Residual Connection and Feed-Forward Network
        # ========================================================================
        # Combine local and full-connect contributions with residual connection
        new_x = new_x + self.scale * new_x_fc + shortcut1
        shortcut2 = new_x
        
        # Apply normalization and feed-forward network
        new_x = self.norm4(new_x)
        new_x = self.FFN(new_x)
        new_x = self.dropout(new_x)

        # Return output with residual connection
        return new_x + shortcut2


class GNN(torch.nn.Module):
    """
    Graph Neural Network for Molecular Property Prediction
    
    A multi-layer GNN architecture that supports both pre-training and fine-tuning modes.
    In fine-tuning mode, it can leverage pre-trained embeddings from a larger pre-trained model.
    
    Args:
        num_layer (int): Number of GAT layers in the network
        emb_dim (int): Embedding dimension for node features
        drop_ratio (float): Dropout rate for regularization (default: 0)
        output_type (str): Output mode - "last" returns final layer output, 
                          "layers" returns outputs from all layers (default: "last")
        model_mode (str): Training mode - "pretrain" or "finetune" (default: "pretrain")
        pretrain_emb_dim (int): Embedding dimension of pre-trained model (default: 126)
        pretrain_num_layer (int): Number of layers in pre-trained model (default: 12)
    
    Attributes:
        gnns (ModuleList): List of GAT convolution layers
        x_embedding1 (Embedding): Atom type embedding
        x_embedding2 (Embedding): Atom chirality embedding
        x_embedding3-5 (Embedding): Additional atom feature embeddings
        fc_condition (ModuleList): Linear layers for conditioning in fine-tuning mode
    """
    
    def __init__(self, num_layer, emb_dim, drop_ratio=0, output_type="last", 
                 model_mode="pretrain", pretrain_emb_dim=126, pretrain_num_layer=12):
        super(GNN, self).__init__()
        
        # Store configuration parameters
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.pretrain_num_layer = pretrain_num_layer
        self.model_mode = model_mode
        self.output_type = output_type

        # ========================================================================
        # Input Validation
        # ========================================================================
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than or equal to 2.")
        
        if self.model_mode not in ["pretrain", "finetune"]:
            raise ValueError(f"model_mode must be 'pretrain' or 'finetune', got {self.model_mode}")
        
        if self.model_mode == "finetune" and self.num_layer < self.pretrain_num_layer:
            raise ValueError(
                f"In finetune mode, num_layer ({self.num_layer}) must be >= "
                f"pretrain_num_layer ({self.pretrain_num_layer})"
            )

        # ========================================================================
        # Node Feature Embeddings
        # ========================================================================
        # Primary atom feature embeddings
        self.x_embedding1 = torch.nn.Embedding(NUM_ATOM_TYPE, emb_dim)  # Atom type
        self.x_embedding2 = torch.nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)  # Atom chirality
        
        # Auxiliary atom feature embeddings
        self.x_embedding3 = torch.nn.Embedding(20, emb_dim)  # Additional feature 1
        self.x_embedding4 = torch.nn.Embedding(10, emb_dim)  # Additional feature 2
        self.x_embedding5 = torch.nn.Embedding(10, emb_dim)  # Additional feature 3
        
        # Feed-forward network for combining auxiliary embeddings
        self.fc_embedding = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Initialize embedding weights with Xavier uniform distribution
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)

        # ========================================================================
        # Graph Attention Convolution Layers
        # ========================================================================
        self.gnns = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GATConv(emb_dim, drop_ratio=self.drop_ratio))
        
        # ========================================================================
        # Fine-tuning Mode Specific Components
        # ========================================================================
        if self.output_type == "last" and self.model_mode == "finetune":
            # Linear layers for conditioning the fine-tune model with pre-trained embeddings
            self.fc_condition = nn.ModuleList()
            for _ in range(pretrain_num_layer):
                condition_layer = nn.Linear(pretrain_emb_dim, emb_dim, bias=False) if pretrain_emb_dim != emb_dim else nn.Identity()
                self.fc_condition.append(
                    condition_layer
                )

    def forward(self, node_atom, edge_index, edge_index_all, edge_attr, batch, extra_embedding=None):
        """
        Forward pass through the GNN layers.
        
        Args:
            node_atom (Tensor): Node atom features of shape [num_nodes, num_features]
                               Expected format: [atom_type, chirality, feature1, feature2, feature3]
            edge_index (LongTensor): Edge indices for local graph [2, num_edges]
            edge_index_all (LongTensor): Edge indices for full-connect graph [2, num_edges_full]
            edge_attr (Tensor): Edge attributes [num_edges, 3]
            batch (LongTensor): Batch assignment for each node [num_nodes]
            extra_embedding (list of Tensor, optional): Pre-trained embeddings for fine-tuning.
                                                       List of layer outputs from pre-trained model.
        
        Returns:
            Tensor or List[Tensor]: 
                - If output_type == "last": Node representations after final layer [num_nodes, emb_dim]
                - If output_type == "layers": List of node representations from all layers
        """
        
        # ========================================================================
        # Stage 1: Node Feature Embedding
        # ========================================================================
        # Combine primary atom features (atom type + chirality)
        x = self.x_embedding1(node_atom[:, 0]) + self.x_embedding2(node_atom[:, 1])
        
        # Process auxiliary atom features
        extra_features = (self.x_embedding3(node_atom[:, 2]) + 
                         self.x_embedding4(node_atom[:, 3]) + 
                         self.x_embedding5(node_atom[:, 4]))
        extra_features = self.fc_embedding(extra_features)
        
        # Combine all embeddings
        x = x + extra_features
        
        # Initialize list for storing layer outputs if needed
        if self.output_type == "layers":
            h = []

        # ========================================================================
        # Stage 2: Multi-layer Graph Convolution
        # ========================================================================
        for layer_idx in range(self.num_layer):
            # In fine-tuning mode, condition the network with pre-trained embeddings
            if self.model_mode == "finetune":
                if extra_embedding is not None and layer_idx < self.pretrain_num_layer:
                    # Project pre-trained embedding to current dimension and add as residual
                    condition_feature = self.fc_condition[layer_idx](extra_embedding[layer_idx])
                    x = x + condition_feature
            
            # Apply graph attention convolution
            x = self.gnns[layer_idx](x, edge_index=edge_index, 
                                     edge_index_all=edge_index_all, 
                                     edge_attr=edge_attr)
            
            # Store layer output if returning all layers
            if self.output_type == "layers":
                h.append(x)

        # ========================================================================
        # Return Results
        # ========================================================================
        if self.output_type == "layers":
            return h
        return x

if __name__ == "__main__":
    """
    Main entry point for testing the GNN models.
    """
    pass
