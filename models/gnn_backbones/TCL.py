import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Add project root to path and import debug config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from debug_config import should_debug_model  # 🔍 Global debug control
from models.gnn_backbones.modules import TimeEncoder, TransformerEncoder
from utils.utils import NeighborSampler


class TCL(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, num_depths: int = 20, dropout: float = 0.1, device: str = 'cpu', time_encoder: nn.Module = None):
        """
        TCL model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param num_depths: int, number of depths, identical to the number of sampled neighbors plus 1 (involving the target node)
        :param dropout: float, dropout rate
        :param device: str, device
        :param time_encoder: nn.Module, optional custom time encoder (if None, uses default TimeEncoder)
        """
        super(TCL, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_depths = num_depths
        self.dropout = dropout
        self.device = device

        # Use provided time encoder or create default one
        if time_encoder is not None:
            self.time_encoder = time_encoder
            print(f"TCL: Using custom time encoder: {type(time_encoder).__name__}")
        else:
            self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
            print(f"TCL: Using default TimeEncoder")
            
        self.depth_embedding = nn.Embedding(num_embeddings=num_depths, embedding_dim=self.node_feat_dim)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.node_feat_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.node_feat_dim, bias=True)
        })

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.node_feat_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # get temporal neighbors of source nodes, including neighbor ids, edge ids and time information
        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors)
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # get temporal neighbors of destination nodes, including neighbor ids, edge ids and time information
        # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors)
        dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_node_ids = np.concatenate((src_node_ids[:, np.newaxis], src_neighbor_node_ids), axis=1)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_edge_ids = np.concatenate((np.zeros((len(src_node_ids), 1)).astype(np.longlong), src_neighbor_edge_ids), axis=1)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], src_neighbor_times), axis=1)

        # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_node_ids = np.concatenate((dst_node_ids[:, np.newaxis], dst_neighbor_node_ids), axis=1)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_edge_ids = np.concatenate((np.zeros((len(dst_node_ids), 1)).astype(np.longlong), dst_neighbor_edge_ids), axis=1)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], dst_neighbor_times), axis=1)

        # pad the features of the sequence of source and destination nodes
        # src_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # src_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # src_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # src_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, node_feat_dim)
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_nodes_neighbor_depth_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=src_neighbor_node_ids,
                              nodes_edge_ids=src_neighbor_edge_ids, nodes_neighbor_times=src_neighbor_times, time_encoder=self.time_encoder)

        # dst_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # dst_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # dst_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # dst_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, node_feat_dim)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, dst_nodes_neighbor_depth_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=dst_neighbor_node_ids,
                              nodes_edge_ids=dst_neighbor_edge_ids, nodes_neighbor_times=dst_neighbor_times, time_encoder=self.time_encoder)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_nodes_neighbor_node_raw_features)
        dst_nodes_edge_raw_features = self.projection_layer['edge'](dst_nodes_edge_raw_features)
        dst_nodes_neighbor_time_features = self.projection_layer['time'](dst_nodes_neighbor_time_features)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        src_node_features = src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features + src_nodes_neighbor_depth_features
        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        dst_node_features = dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features + dst_nodes_neighbor_depth_features

        for transformer in self.transformers:
            # self-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_features = transformer(inputs_query=src_node_features, inputs_key=src_node_features,
                                            inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            dst_node_features = transformer(inputs_query=dst_node_features, inputs_key=dst_node_features,
                                            inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            # cross-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_embeddings = transformer(inputs_query=src_node_features, inputs_key=dst_node_features,
                                              inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            dst_node_embeddings = transformer(inputs_query=dst_node_features, inputs_key=src_node_features,
                                              inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)

            src_node_features, dst_node_features = src_node_embeddings, dst_node_embeddings

        # retrieve the embedding of the corresponding target node, which is at the first position of the sequence
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_node_embeddings[:, 0, :])
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_node_embeddings[:, 0, :])

        return src_node_embeddings, dst_node_embeddings

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge, time and depth features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_edge_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_neighbor_times: ndarray, shape (batch_size, num_neighbors + 1)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        # Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        # Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        
        # ✅ ENHANCED: Support KAN_MAMMOTE dual-stream interface
        time_encoder_name = getattr(time_encoder, '__class__', type(time_encoder)).__name__
        
        if hasattr(time_encoder, 'encoder') and hasattr(time_encoder.encoder, '__class__'):
            # Check if wrapped encoder is KAN_MAMMOTE variant
            wrapped_encoder_name = time_encoder.encoder.__class__.__name__
            is_kan_mammote = wrapped_encoder_name in ['KAN_MAMMOTE', 'KAN_MAMMOTE_Lite']
        else:
            # Direct encoder check
            is_kan_mammote = time_encoder_name in ['KAN_MAMMOTE', 'KAN_MAMMOTE_Lite']
        
        # 🔍 COMPREHENSIVE DEBUG OUTPUT (controlled by args.debug_encoder)
        debug_enabled = hasattr(self, '_debug_encoder') and self._debug_encoder
        if debug_enabled:
            print(f"\n🏷️  TCL TIME ENCODER DEBUG:")
            print(f"   Time encoder: {time_encoder_name}")
            print(f"   KAN_MAMMOTE variant: {is_kan_mammote}")
            print(f"   Batch size: {node_interact_times.shape[0]}")
            print(f"   Sequence length: {nodes_neighbor_times.shape[1] if len(nodes_neighbor_times.shape) > 1 else 1}")
            print(f"   node_interact_times range: [{node_interact_times.min():.1f}, {node_interact_times.max():.1f}]")
            print(f"   nodes_neighbor_times range: [{nodes_neighbor_times.min():.1f}, {nodes_neighbor_times.max():.1f}]")
        
        if is_kan_mammote:
            # KAN_MAMMOTE dual-stream: Pass both absolute and relative time
            t_abs = torch.from_numpy(nodes_neighbor_times).float().to(self.device)  # Absolute neighbor times
            t_rel = torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device)  # Relative time differences
            
            # 🔍 DEBUG: Check if neighbor times are sorted before encoder
            if should_debug_model() and not hasattr(self, '_debug_printed_sorting') and nodes_neighbor_times.size > 0:
                print(f"\n🔍 [TCL] Neighbor Time Sorting Debug:")
                print(f"   Batch size: {nodes_neighbor_times.shape[0]}, Neighbors+self: {nodes_neighbor_times.shape[1]}")
                
                # Check first row for sorting (remember: [current_node, neighbor1, neighbor2, ...])
                first_neighbor_times = nodes_neighbor_times[0]
                # Skip first element (current node) and check if neighbors are sorted
                if len(first_neighbor_times) > 1:
                    neighbor_only = first_neighbor_times[1:]  # Skip current node
                    is_sorted = all(neighbor_only[i] <= neighbor_only[i+1] for i in range(len(neighbor_only)-1)) if len(neighbor_only) > 1 else True
                    print(f"   First row times: {first_neighbor_times[:5]}...")
                    print(f"   Neighbors only sorted: {is_sorted}")
                
                # Check t_abs and t_rel values
                print(f"   t_abs sample: {t_abs[0, :5].cpu().numpy()}")
                print(f"   t_rel sample: {t_rel[0, :5].cpu().numpy()}")
                print(f"   Time computation: current_time - neighbor_time")
                
                self._debug_printed_sorting = True
            
            if debug_enabled:
                print(f"   🎯 DUAL-STREAM INPUT:")
                print(f"      t_abs shape: {t_abs.shape}")
                print(f"      t_rel shape: {t_rel.shape}")
                print(f"      t_abs sample: {t_abs.flatten()[:5].cpu().numpy()}")
                print(f"      t_rel sample: {t_rel.flatten()[:5].cpu().numpy()}")
                print(f"   🔍 NON-ZERO ANALYSIS:")
                
                print(f"      t_abs non-zero count: {(t_abs != 0).sum().item()} / {t_abs.numel()}")
                print(f"      t_rel non-zero count: {(t_rel != 0).sum().item()} / {t_rel.numel()}")
                print(f"      t_abs unique values: {torch.unique(t_abs)[:10]}")
                print(f"      t_rel unique values: {torch.unique(t_rel)[:10]}")
                # Enable debug for time encoder if it supports it
                if hasattr(time_encoder, 'enable_debug_mode'):
                    time_encoder.enable_debug_mode()
                    nodes_neighbor_time_features = time_encoder(t_abs=t_abs, t_rel=t_rel)
                    time_encoder.disable_debug_mode()
                else:
                    nodes_neighbor_time_features = time_encoder(t_abs=t_abs, t_rel=t_rel)
            else:
                nodes_neighbor_time_features = time_encoder(t_abs=t_abs, t_rel=t_rel)
        else:
            # Standard time encoders: Use relative time only (backward compatibility)
            timestamps = torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device)
            
            if debug_enabled:
                print(f"   🎯 SINGLE-STREAM INPUT:")
                print(f"      timestamps shape: {timestamps.shape}")
                print(f"      timestamps sample: {timestamps.flatten()[:5].cpu().numpy()}")
            
            nodes_neighbor_time_features = time_encoder(timestamps=timestamps)
        
        if debug_enabled:
            print(f"   📤 OUTPUT:")
            print(f"      nodes_neighbor_time_features shape: {nodes_neighbor_time_features.shape}")
            print(f"      Output sample: {nodes_neighbor_time_features.flatten()[:5].detach().cpu().numpy()}")
        
        assert nodes_neighbor_ids.shape[1] == self.depth_embedding.weight.shape[0]
        # Tensor, shape (num_neighbors + 1, node_feat_dim)
        nodes_neighbor_depth_features = self.depth_embedding(torch.tensor(range(nodes_neighbor_ids.shape[1])).to(self.device))

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_depth_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
