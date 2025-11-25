import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Add project root to path and import debug config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.gnn_backbones.modules import TimeEncoder, MergeLayer, MultiHeadAttention
from utils.utils import NeighborSampler


class TGAT(nn.Module):
    """
    A clean, refactored TGAT model that uses a standardized dual-stream time encoder.
    The specific time encoder is injected during initialization.
    """
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_encoder: nn.Module, time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, 
                 dropout: float = 0.1, device: str = 'cpu', sort_neighbors_by_time: bool = False):
        super(TGAT, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.time_encoder = time_encoder
        self.sort_neighbors_by_time = sort_neighbors_by_time

        self.temporal_conv_layers = nn.ModuleList([MultiHeadAttention(
            node_feat_dim=self.node_feat_dim,
            edge_feat_dim=self.edge_feat_dim,
            time_feat_dim=self.time_feat_dim,
            num_heads=self.num_heads,
            dropout=self.dropout) for _ in range(num_layers)])
            
        self.merge_layers = nn.ModuleList([MergeLayer(
            input_dim1=self.node_feat_dim + self.time_feat_dim, 
            input_dim2=self.node_feat_dim,
            hidden_dim=self.node_feat_dim, 
            output_dim=self.node_feat_dim) for _ in range(num_layers)])

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times, num_neighbors=20):
        src_node_embeddings = self.compute_node_temporal_embeddings(src_node_ids, node_interact_times, self.num_layers, num_neighbors)
        dst_node_embeddings = self.compute_node_temporal_embeddings(dst_node_ids, node_interact_times, self.num_layers, num_neighbors)
        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids, node_interact_times, current_layer_num, num_neighbors=20):
        assert current_layer_num >= 0
        node_raw_features = self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_raw_features
        else:
            node_conv_features = self.compute_node_temporal_embeddings(node_ids, node_interact_times, current_layer_num - 1, num_neighbors)
            
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids, node_interact_times, num_neighbors)
            
            # ===== FIX: Sort neighbors chronologically for Mamba-based encoders =====
            if self.sort_neighbors_by_time:
                # Sort neighbors by time (oldest to newest) to create proper sequences for Mamba
                sorted_indices = np.argsort(neighbor_times, axis=1)
                
                # Reorder all neighbor arrays
                neighbor_times = np.take_along_axis(neighbor_times, sorted_indices, axis=1)
                neighbor_node_ids = np.take_along_axis(neighbor_node_ids, sorted_indices, axis=1)
                neighbor_edge_ids = np.take_along_axis(neighbor_edge_ids, sorted_indices, axis=1)
            # ===== END FIX =====
            
            neighbor_conv_features = self.compute_node_temporal_embeddings(neighbor_node_ids.flatten(), neighbor_times.flatten(), current_layer_num - 1, num_neighbors)
            neighbor_conv_features = neighbor_conv_features.reshape(len(node_ids), num_neighbors, self.node_feat_dim)

            # --- Cleaned Time Logic ---
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times
            
            neighbor_t_abs = torch.from_numpy(neighbor_times).float().to(self.device).unsqueeze(-1)
            neighbor_t_rel = torch.from_numpy(neighbor_delta_times).float().to(self.device).unsqueeze(-1)
            
            # Check if neighbor times are sorted before encoder
            if neighbor_times.size > 0:
                print(f"\n[TGAT] Neighbor Time Sorting Debug:")
                print(f"   Layer: {current_layer_num}, sort_neighbors_by_time: {self.sort_neighbors_by_time}")
                print(f"   Batch size: {neighbor_times.shape[0]}, Num neighbors: {neighbor_times.shape[1]}")
                
                # Check first row for sorting
                first_neighbor_times = neighbor_times[0]
                is_sorted = all(first_neighbor_times[i] <= first_neighbor_times[i+1] for i in range(len(first_neighbor_times)-1))
                print(f"   First row neighbor times: {first_neighbor_times[:5]}...")
                print(f"   Is chronologically sorted: {is_sorted}")
                
                # Check t_abs and t_rel values
                print(f"   neighbor_t_abs sample: {neighbor_t_abs[0, :5, 0].cpu().numpy()}")
                print(f"   neighbor_t_rel sample: {neighbor_t_rel[0, :5, 0].cpu().numpy()}")
                print(f"   Time computation: current_time - neighbor_time")
                
                self._debug_printed_sorting = True
            
            # This single call works for both KAN-MAMMOTE and the wrapped TimeEncoder
            neighbor_time_features = self.time_encoder(t_abs=neighbor_t_abs, t_rel=neighbor_t_rel)

            source_t_abs = torch.from_numpy(node_interact_times).float().to(self.device).unsqueeze(-1).unsqueeze(-1)
            source_t_rel = torch.zeros_like(source_t_abs)
            node_time_features = self.time_encoder(t_abs=source_t_abs, t_rel=source_t_rel).squeeze(1)
            
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
            
            '''# DIMENSION FIXING: Ensure proper dimensions before attention
            print(f"DEBUG TGAT: Tensor shapes before attention:")
            print(f"  - node_conv_features: {node_conv_features.shape}")
            print(f"  - node_time_features: {node_time_features.shape}")
            print(f"  - neighbor_conv_features: {neighbor_conv_features.shape}")
            print(f"  - neighbor_time_features: {neighbor_time_features.shape}")
            print(f"  - neighbor_edge_features: {neighbor_edge_features.shape}")'''
            
            # Fix node_time_features dimensions - should be (batch_size, 1, time_feat_dim)
            if node_time_features.dim() == 2:  # (batch_size, time_feat_dim)
                node_time_features = node_time_features.unsqueeze(1)  # (batch_size, 1, time_feat_dim)
            elif node_time_features.dim() == 3 and node_time_features.shape[1] != 1:
                node_time_features = node_time_features.mean(dim=1, keepdim=True)  # Average sequence to single timestep
            
            # Fix neighbor_time_features dimensions - should be (batch_size, num_neighbors, time_feat_dim)
            if neighbor_time_features.dim() == 2:  # (batch_size, time_feat_dim)
                # Expand to match num_neighbors
                neighbor_time_features = neighbor_time_features.unsqueeze(1).expand(-1, num_neighbors, -1)
            elif neighbor_time_features.dim() == 4:  # (batch_size, num_neighbors, 1, time_feat_dim)
                neighbor_time_features = neighbor_time_features.squeeze(2)
            elif neighbor_time_features.dim() == 3:
                # Check if we have the right number of neighbors
                if neighbor_time_features.shape[1] != num_neighbors:
                    if neighbor_time_features.shape[1] > num_neighbors:
                        neighbor_time_features = neighbor_time_features[:, :num_neighbors, :]
                    else:
                        # Pad with zeros or repeat last
                        pad_size = num_neighbors - neighbor_time_features.shape[1]
                        padding = neighbor_time_features[:, -1:, :].expand(-1, pad_size, -1)
                        neighbor_time_features = torch.cat([neighbor_time_features, padding], dim=1)
            
            ''' print(f"DEBUG TGAT: Tensor shapes after fixing:")
            print(f"  - node_conv_features: {node_conv_features.shape}")
            print(f"  - node_time_features: {node_time_features.shape}")
            print(f"  - neighbor_conv_features: {neighbor_conv_features.shape}")
            print(f"  - neighbor_time_features: {neighbor_time_features.shape}")
            print(f"  - neighbor_edge_features: {neighbor_edge_features.shape}")'''
            
            output, _ = self.temporal_conv_layers[current_layer_num - 1](
                node_features=node_conv_features,
                node_time_features=node_time_features,
                neighbor_node_features=neighbor_conv_features,
                neighbor_node_time_features=neighbor_time_features,
                neighbor_node_edge_features=neighbor_edge_features,
                neighbor_masks=neighbor_node_ids
            )
            output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_raw_features)
            return output

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
