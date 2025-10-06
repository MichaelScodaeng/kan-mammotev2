import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

# Add parent directory to Python path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_backbones.TGAT import TGAT
from models.gnn_backbones.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.gnn_backbones.CAWN import CAWN
from models.gnn_backbones.TCL import TCL
from models.gnn_backbones.GraphMixer import GraphMixer
from models.gnn_backbones.DyGFormer import DyGFormer
from models.gnn_backbones.DyGMamba import DyGMamba
from models.gnn_backbones.modules import MergeLayer, MergeLayerTD
from models.time_encoders.factory import create_time_encoder
from utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.metrics_logger import create_metrics_logger
from experiments.evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from models.time_encoders import KAN_MAMMOTE, KAN_MAMMOTE_Lite, TimeEncoderWrapper
if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # ===== DEBUG: Print configuration before data loading =====
    print(f"\n🔍 DEBUGGING DATA LOADING:")
    print(f"   Time encoder: {args.time_encoder_type}")
    print(f"   Dataset: {args.dataset_name}")
    print(f"   Val ratio: {args.val_ratio}")
    print(f"   Test ratio: {args.test_ratio}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Load best configs: {args.load_best_configs}")
    print(f"   Data ratio: {args.data_ratio}")
    print(f"   Seed: {args.seed}")
    # ===== END DEBUG =====

    # get data for training, validation and testing
    # ✅ NEW: Pass data_ratio and seed to data loader (applies BEFORE splitting)
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(
            dataset_name=args.dataset_name, 
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,
            seed=args.seed,  # ✅ Fixed seed for reproducibility
            data_ratio=args.data_ratio  # ✅ Applied BEFORE splitting
        )

    # ===== DEBUG: Print actual data sizes =====
    print(f"\n📊 DATA SPLIT SIZES (after data_ratio={args.data_ratio}):")
    print(f"   Full data: {len(full_data.src_node_ids):,} edges")
    print(f"   Train data: {len(train_data.src_node_ids):,} edges ({len(train_data.src_node_ids)/len(full_data.src_node_ids)*100:.1f}%)")
    print(f"   Val data: {len(val_data.src_node_ids):,} edges ({len(val_data.src_node_ids)/len(full_data.src_node_ids)*100:.1f}%)")
    print(f"   Test data: {len(test_data.src_node_ids):,} edges ({len(test_data.src_node_ids)/len(full_data.src_node_ids)*100:.1f}%)")
    print(f"   Expected batches: {(len(train_data.src_node_ids) + args.batch_size - 1) // args.batch_size}")
    print(f"   ✅ All splits proportionally scaled with ratio {len(train_data.src_node_ids)/len(full_data.src_node_ids):.2f}:{len(val_data.src_node_ids)/len(full_data.src_node_ids):.2f}:{len(test_data.src_node_ids)/len(full_data.src_node_ids):.2f}")
    # ===== END DEBUG =====

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                             sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                             time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data,
                                                            sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                            time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so

    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        # Include time encoder in save name so logs/models/results are encoder-specific
        base_save_name = f'{args.model_name}_{args.time_encoder_type}_seed{args.seed}'
        
        # Add suffix if specified (for ablation studies)
        if hasattr(args, 'save_model_name_suffix') and args.save_model_name_suffix:
            args.save_model_name = f'{base_save_name}_{args.save_model_name_suffix}'
        else:
            args.save_model_name = base_save_name

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        # Try to set up file logging, but fall back to console-only if file system has issues
        try:
            os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
            # create file handler that logs debug and higher level messages
            fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except (OSError, IOError) as e:
            print(f"Warning: Could not set up file logging due to: {e}. Continuing with console logging only.")
        
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # Initialize metrics logger for this run
        metrics_logger = create_metrics_logger(args, run_id=run)
        logger.info(f'Metrics will be saved to: {metrics_logger.metrics_dir}')

        # --- MODIFICATION START ---
        # create time encoder using the factory
        # This now correctly passes the specific args for KAN-MAMMOTE
        time_encoder = create_time_encoder(
            encoder_type=args.time_encoder_type,
            time_dim=args.time_feat_dim,
            device=args.device,
            # These are passed into **kwargs for the factory
            expert_dim=args.expert_dim, 
            num_mixtures=args.num_mixtures
        )
        logger.info(f'time encoder type: {args.time_encoder_type}')

        if args.model_name == 'TGAT':
        
            # Use the factory to create the correct time encoder
            time_encoder = create_time_encoder(
                encoder_type=args.time_encoder_type,
                time_dim=args.time_feat_dim,
                train_data=train_data,
                train_neighbor_sampler=train_neighbor_sampler,
                args=args,  # Pass args here - factory will extract needed parameters
                device=args.device
            )

            # Inject the created encoder into the single, flexible TGAT backbone
            dynamic_backbone = TGAT(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features, 
                neighbor_sampler=train_neighbor_sampler,
                time_encoder=time_encoder, # <<<<<< INJECT THE CHOSEN ENCODER
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                device=args.device,
                sort_neighbors_by_time=args.sort_neighbors_by_time  # Enable for Mamba-based encoders
            )
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)

        elif args.model_name == 'DyGMamba':
            dynamic_backbone = DyGMamba(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,gamma=args.gamma,
                                         max_input_sequence_length=args.max_input_sequence_length, max_interaction_times=args.max_interaction_times,device=args.device)

        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        if args.model_name == 'DyGMamba':
            link_predictor = MergeLayerTD(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1], input_dim3=node_raw_features.shape[1],
                                        hidden_dim=node_raw_features.shape[1], output_dim=1)
        else:
            link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                        hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        # ===== WARM UP KAN-MAMMOTE (if applicable) =====
        # Warm up CUDA kernels for Mamba2 to avoid 5-40 second compilation delays
        if args.model_name == 'TGAT' and hasattr(time_encoder, 'encoder'):
            # Unwrap the TimeEncoderWrapper to access the actual encoder
            actual_encoder = time_encoder.encoder
            if isinstance(actual_encoder, (KAN_MAMMOTE, KAN_MAMMOTE_Lite)):
                logger.info(f"Warming up {actual_encoder.__class__.__name__}...")
                actual_encoder.warmup(device=args.device, num_iterations=3)
        elif args.time_encoder_type in ['kan_mammote', 'kan_mammote_lite']:
            # Direct usage without wrapper (for other models)
            if hasattr(time_encoder, 'warmup'):
                logger.info(f"Warming up time encoder...")
                time_encoder.warmup(device=args.device, num_iterations=3)
        # ===== END WARM UP =====

        # Use ablation_dir if provided, otherwise default to ./saved_models
        if hasattr(args, 'ablation_dir') and args.ablation_dir:
            save_model_folder = f"{args.ablation_dir}/saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}"
        else:
            save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}"
        
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()
        best_acc = 0
        for epoch in range(args.num_epochs):

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGMamba']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            print("batch size: ", args.batch_size)
            print("num batches: ", len(train_idx_data_loader))
            print("num training interactions: ", len(train_data.src_node_ids))
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, dynamic_ncols=True, leave=False)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):

                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                elif args.model_name in ['DyGFormer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                elif args.model_name in ['DyGMamba']:
                    # get temporal embedding of source , destination nodes and time difference
                    # three Tensors, with shape (batch_size, node_feat_dim)

                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_time_diff_emb = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    # get temporal embedding of negative source , destination nodes and time difference
                    # three Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_time_diff_emb = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                if args.model_name in ['DyGMamba']:
                    positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings, input_3=batch_time_diff_emb).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings, input_3=batch_neg_time_diff_emb).squeeze(dim=-1).sigmoid()
                else:
                    positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings,
                                                    input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                train_losses.append(loss.item())
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                 # Update progress bar less frequently to avoid conflicts
                if batch_idx % 5 == 0 or batch_idx == len(train_idx_data_loader_tqdm) - 1:
                    train_idx_data_loader_tqdm.set_description(
                        f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}/{len(train_idx_data_loader_tqdm)}, Loss: {loss.item():.4f}'
                    )

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()


            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after training so it can be used for new validation nodes
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)
            


            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                # reload training memory bank for new validation nodes
                model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)



            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reload validation memory bank for testing nodes or saving models
                # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            # ===== LOG METRICS TO CSV =====
            # Log training metrics
            train_metrics_avg = {k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0].keys()}
            metrics_logger.log_epoch_metrics(
                epoch=epoch + 1,
                phase='train',
                metrics=train_metrics_avg,
                loss=np.mean(train_losses)
            )
            
            # Log validation metrics
            val_metrics_avg = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()}
            metrics_logger.log_epoch_metrics(
                epoch=epoch + 1,
                phase='val',
                metrics=val_metrics_avg,
                loss=np.mean(val_losses)
            )
            
            # Log new node validation metrics
            new_node_val_metrics_avg = {k: np.mean([m[k] for m in new_node_val_metrics]) for k in new_node_val_metrics[0].keys()}
            metrics_logger.log_epoch_metrics(
                epoch=epoch + 1,
                phase='new_node_val',
                metrics=new_node_val_metrics_avg,
                loss=np.mean(new_node_val_losses)
            )
            # ===== END METRICS LOGGING =====

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)


                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for new testing nodes
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap)


                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                    if metric_name == 'average_precision':
                        if np.mean([test_metric[metric_name] for test_metric in test_metrics]) > best_acc:
                            best_acc = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                        logger.info(
                            f'best test average_precision: {best_acc:.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

                # ===== LOG TEST METRICS TO CSV EVERY test_interval_epochs =====
                # Log test metrics (periodic during training)
                test_metrics_avg = {k: np.mean([m[k] for m in test_metrics]) for k in test_metrics[0].keys()}
                metrics_logger.log_epoch_metrics(
                    epoch=epoch + 1,
                    phase='test_periodic',
                    metrics=test_metrics_avg,
                    loss=np.mean(test_losses)
                )
                
                # Log new node test metrics (periodic during training)
                new_node_test_metrics_avg = {k: np.mean([m[k] for m in new_node_test_metrics]) for k in new_node_test_metrics[0].keys()}
                metrics_logger.log_epoch_metrics(
                    epoch=epoch + 1,
                    phase='new_node_test_periodic',
                    metrics=new_node_test_metrics_avg,
                    loss=np.mean(new_node_test_losses)
                )
                # ===== END TEST METRICS LOGGING =====

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)
        
            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # reload validation memory bank for new testing nodes
            model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric
        
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
                new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        # ===== LOG FINAL TEST METRICS =====
        test_metrics_avg = {k: np.mean([m[k] for m in test_metrics]) for k in test_metrics[0].keys()}
        metrics_logger.log_epoch_metrics(
            epoch=args.num_epochs,
            phase='test',
            metrics=test_metrics_avg,
            loss=np.mean(test_losses)
        )
        
        # Log final new node test metrics
        new_node_test_metrics_avg = {k: np.mean([m[k] for m in new_node_test_metrics]) for k in new_node_test_metrics[0].keys()}
        metrics_logger.log_epoch_metrics(
            epoch=args.num_epochs,
            phase='new_node_test',
            metrics=new_node_test_metrics_avg,
            loss=np.mean(new_node_test_losses)
        )
        
        # Save summary of all metrics
        metrics_logger.save_summary()
        logger.info(f'✅ Metrics saved to: {metrics_logger.metrics_dir}')
        # ===== END TEST METRICS LOGGING =====

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
            new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "time_encoder_type": args.time_encoder_type,
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        else:
            result_json = {
                "time_encoder_type": args.time_encoder_type,
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }

        result_json = json.dumps(result_json, indent=4)


        # Use ablation_dir if provided, otherwise default to ./saved_results
        if hasattr(args, 'ablation_dir') and args.ablation_dir:
            save_result_folder = f"{args.ablation_dir}/saved_results/{args.model_name}/{args.dataset_name}"
        else:
            save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        
        os.makedirs(save_result_folder, exist_ok=True)

        timestamp = str(time.time())
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}_{timestamp}.json")


        while os.path.exists(save_result_path):
            timestamp = str(time.time())
            save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}_{timestamp}.json")
    
        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')
    
        for metric_name in new_node_val_metric_all_runs[0].keys():
            logger.info(f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
            logger.info(f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
                        f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
