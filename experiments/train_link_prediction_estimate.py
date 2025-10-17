import logging
import time
import sys
import os
import glob
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

# Optional Mamba-based imports
try:
    from models.gnn_backbones.DyGMamba import DyGMamba
    from models.time_encoders import KAN_MAMMOTE, KAN_MAMMOTE_Lite, TimeEncoderWrapper
    MAMBA_AVAILABLE = True
except ImportError as e:
    DyGMamba = None
    KAN_MAMMOTE = None
    KAN_MAMMOTE_Lite = None
    TimeEncoderWrapper = None
    MAMBA_AVAILABLE = False
    print(f"Warning: Mamba-based models not available: {e}")

from models.gnn_backbones.modules import MergeLayer, MergeLayerTD
from models.time_encoders.factory import create_time_encoder
from utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.load_configs import get_link_prediction_args
from datetime import datetime
import numpy as np
import json

def get_available_models():
    """Return list of available model types"""
    base_models = ['TGAT', 'JODIE', 'DyRep', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']
    if MAMBA_AVAILABLE:
        base_models.append('DyGMamba')
    return base_models

def get_available_encoders():
    """Return list of available encoder types"""
    base_encoders = ['original', 'lete', 'mercer', 'bochner', 'time2vec']
    if MAMBA_AVAILABLE:
        base_encoders.extend(['kan_mammote', 'kan_mammote_dual_kmote', 'kan_mammote_lite'])
    return base_encoders

def find_best_checkpoint(checkpoint_dir, logger, validate=True):
    """
    Find the best available checkpoint, trying newest first with fallback to older ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        logger: Logger instance
        validate: Whether to validate checkpoint integrity
    
    Returns:
        tuple: (checkpoint_path, epoch) or (None, None) if no valid checkpoint found
    """
    try:
        import glob
        
        # Find all checkpoint files
        checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            logger.info(f"No checkpoints found in {checkpoint_dir}")
            return None, None
        
        # Sort by epoch number (newest first)
        def extract_epoch(filepath):
            try:
                filename = os.path.basename(filepath)
                epoch_str = filename.split('checkpoint_epoch_')[1].split('.pth')[0]
                return int(epoch_str)
            except (IndexError, ValueError):
                return -1
        
        checkpoint_files.sort(key=extract_epoch, reverse=True)
        
        logger.info(f"Found {len(checkpoint_files)} checkpoints in {checkpoint_dir}")
        
        # Try each checkpoint from newest to oldest
        for checkpoint_path in checkpoint_files:
            epoch = extract_epoch(checkpoint_path)
            logger.info(f"Trying checkpoint: {checkpoint_path} (epoch {epoch})")
            
            # Validate checkpoint if requested
            if validate:
                if validate_checkpoint(checkpoint_path, logger):
                    logger.info(f"✅ Using valid checkpoint: {checkpoint_path} (epoch {epoch})")
                    return checkpoint_path, epoch
                else:
                    logger.warning(f"❌ Checkpoint corrupted, trying next: {checkpoint_path}")
                    continue
            else:
                # Skip validation, use directly
                logger.info(f"Using checkpoint (no validation): {checkpoint_path} (epoch {epoch})")
                return checkpoint_path, epoch
        
        # No valid checkpoint found
        logger.warning(f"No valid checkpoints found in {checkpoint_dir}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error finding checkpoints in {checkpoint_dir}: {e}")
        return None, None

def cleanup_old_checkpoints(save_model_folder, max_to_keep, logger):
    """
    Keep only the most recent N checkpoints to save disk space
    
    Args:
        save_model_folder: Directory containing checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        logger: Logger instance
    """
    try:
        import glob
        
        # Find all checkpoint files
        checkpoint_pattern = os.path.join(save_model_folder, 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= max_to_keep:
            return  # No need to cleanup
        
        # Sort by epoch number (extract from filename)
        def extract_epoch(filepath):
            try:
                filename = os.path.basename(filepath)
                epoch_str = filename.split('checkpoint_epoch_')[1].split('.pth')[0]
                return int(epoch_str)
            except (IndexError, ValueError):
                return 0
        
        checkpoint_files.sort(key=extract_epoch)
        
        # Remove oldest checkpoints (keep newest max_to_keep)
        files_to_remove = checkpoint_files[:-max_to_keep]
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                epoch_num = extract_epoch(file_path)
                logger.info(f'Removed old checkpoint: epoch {epoch_num}')
            except OSError as e:
                logger.warning(f'Could not remove old checkpoint {file_path}: {e}')
                
    except Exception as e:
        logger.warning(f'Error during checkpoint cleanup: {e}')

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

    # Support seed-level resuming: start from specified seed instead of always 0
    start_seed = getattr(args, 'start_from_seed', 0)
    for run in range(start_seed, args.num_runs):

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

        # --- MODIFICATION START ---
        # create time encoder using the factory - this will be used by ALL models now
        time_encoder = create_time_encoder(
            encoder_type=args.time_encoder_type,
            time_dim=args.time_feat_dim,
            train_data=train_data,
            train_neighbor_sampler=train_neighbor_sampler,
            args=args,  # Pass args here - factory will extract needed parameters
            device=args.device
        )
        logger.info(f'time encoder type: {args.time_encoder_type}')
        logger.info(f'Time encoder will be injected into {args.model_name} model')

        # Check if trying to use unavailable models
        if args.model_name == 'DyGMamba' and not MAMBA_AVAILABLE:
            raise RuntimeError(f"Cannot use DyGMamba: Mamba libraries not installed. "
                             f"Available models: {get_available_models()}")

        if args.model_name == 'TGAT':
            # Inject the created encoder into TGAT backbone
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
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, 
                                           device=args.device, time_encoder=time_encoder)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device, time_encoder=time_encoder)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device, time_encoder=time_encoder)
            
            # 🔍 Enable debug mode for TCL if debug_encoder is set
            if args.debug_encoder:
                dynamic_backbone._debug_encoder = True
                logger.info(f"🔍 Debug mode ENABLED for TCL time encoder calls")
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device, time_encoder=time_encoder)

        elif args.model_name == 'DyGMamba':
            if not MAMBA_AVAILABLE:
                raise RuntimeError("DyGMamba requires Mamba libraries which are not installed")
            dynamic_backbone = DyGMamba(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,gamma=args.gamma,
                                         max_input_sequence_length=args.max_input_sequence_length, max_interaction_times=args.max_interaction_times,device=args.device, time_encoder=time_encoder)

        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device, time_encoder=time_encoder)
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
        # Only attempt warmup if Mamba is available
        if MAMBA_AVAILABLE:
            # Warm up CUDA kernels for Mamba2 to avoid 5-40 second compilation delays
            if args.model_name == 'TGAT' and hasattr(time_encoder, 'encoder'):
                # Unwrap the TimeEncoderWrapper to access the actual encoder
                actual_encoder = time_encoder.encoder
                if isinstance(actual_encoder, (KAN_MAMMOTE, KAN_MAMMOTE_Lite)):
                    logger.info(f"Warming up {actual_encoder.__class__.__name__}...")
                    actual_encoder.warmup(device=args.device, num_iterations=3)
            elif args.time_encoder_type in ['kan_mammote', 'kan_mammote_dual_kmote', 'kan_mammote_lite']:
                # Direct usage without wrapper (for other models)
                if hasattr(time_encoder, 'warmup'):
                    logger.info(f"Warming up time encoder...")
                    time_encoder.warmup(device=args.device, num_iterations=3)
        # ===== END WARM UP =====

        loss_func = nn.BCELoss()
        # Only run first epoch for timing estimation
        for epoch in range(0, 1):

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGMamba']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            """ print("batch size: ", args.batch_size)
            print("num batches: ", len(train_idx_data_loader))
            print("num training interactions: ", len(train_data.src_node_ids)) """
            
            # ===== IMPROVED TRAINING TIME ESTIMATION =====
            logger.info("🕒 Starting improved training time estimation...")
            
            # Take more batches for better accuracy, especially for Mamba models
            timing_batches = min(10, len(train_idx_data_loader))  # Increased from 3 to 10
            batch_times = []
            
            # Add warm-up batches for Mamba models to exclude compilation overhead
            warmup_batches = 2 if args.time_encoder_type in ['kan_mammote', 'kan_mammote_dual_kmote', 'kan_mammote_lite'] else 0
            
            logger.info(f"   Warm-up batches: {warmup_batches}")
            logger.info(f"   Timing batches: {timing_batches}")
            
            train_idx_data_loader_iter = iter(train_idx_data_loader)
            
            # Warm-up phase (exclude from timing)
            for warmup_idx in range(warmup_batches):
                try:
                    train_data_indices = next(train_idx_data_loader_iter)
                    logger.info(f"   Warm-up batch {warmup_idx + 1}/{warmup_batches}...")
                    
                    # Run the same forward pass but don't time it
                    train_data_indices = train_data_indices.numpy()
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                        train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                    _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                    batch_neg_src_node_ids = batch_src_node_ids

                    # Run forward pass (same as timing phase)
                    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                num_neighbors=args.num_neighbors)
                        
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                num_neighbors=args.num_neighbors)
                                
                    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=None,
                                edges_are_positive=False,
                                num_neighbors=args.num_neighbors)
                        
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=batch_edge_ids,
                                edges_are_positive=True,
                                num_neighbors=args.num_neighbors)
                                
                    elif args.model_name in ['GraphMixer']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                num_neighbors=args.num_neighbors,
                                time_gap=args.time_gap)
                        
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                num_neighbors=args.num_neighbors,
                                time_gap=args.time_gap)
                                
                    elif args.model_name in ['DyGFormer']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times)
                        
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times)
                                
                    elif args.model_name in ['DyGMamba']:
                        batch_src_node_embeddings, batch_dst_node_embeddings, batch_time_diff_emb = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times)
                        
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_time_diff_emb = \
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times)
                    
                    # Run prediction and loss calculation
                    if args.model_name in ['DyGMamba']:
                        positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings, input_3=batch_time_diff_emb).squeeze(dim=-1).sigmoid()
                        negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings, input_3=batch_neg_time_diff_emb).squeeze(dim=-1).sigmoid()
                    else:
                        positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                        negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    
                    predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                    labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                    
                    loss = loss_func(input=predicts, target=labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        model[0].memory_bank.detach_memory_bank()
                        
                except StopIteration:
                    break

            if warmup_batches > 0:
                logger.info("   Warm-up complete, starting timing...")
            
            # Actual timing phase
            for batch_idx in range(timing_batches):
                try:
                    train_data_indices = next(train_idx_data_loader_iter)
                except StopIteration:
                    break
                    
                batch_start_time = time.time()
                
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # Run forward pass with the same logic as training
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors)
                    
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors)
                            
                elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            edge_ids=None,
                            edges_are_positive=False,
                            num_neighbors=args.num_neighbors)
                    
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            edge_ids=batch_edge_ids,
                            edges_are_positive=True,
                            num_neighbors=args.num_neighbors)
                            
                elif args.model_name in ['GraphMixer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap)
                    
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap)
                            
                elif args.model_name in ['DyGFormer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times)
                    
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times)
                            
                elif args.model_name in ['DyGMamba']:
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_time_diff_emb = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times)
                    
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_time_diff_emb = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times)
                            
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                # Run prediction and loss calculation
                if args.model_name in ['DyGMamba']:
                    positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings, input_3=batch_time_diff_emb).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings, input_3=batch_neg_time_diff_emb).squeeze(dim=-1).sigmoid()
                else:
                    positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                
                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                
                loss = loss_func(input=predicts, target=labels)
                
                # Simulate backward pass timing (includes optimizer step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    model[0].memory_bank.detach_memory_bank()
                
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)
                
                logger.info(f"  Timing batch {batch_idx + 1}/{timing_batches}: {batch_time:.3f}s")
            
            # Calculate estimates with improved statistics
            if batch_times:
                # Remove outliers (highest and lowest) for better accuracy
                if len(batch_times) >= 5:
                    sorted_times = sorted(batch_times)
                    batch_times_cleaned = sorted_times[1:-1]  # Remove highest and lowest
                    avg_batch_time = np.mean(batch_times_cleaned)
                    logger.info(f"   Using {len(batch_times_cleaned)}/{len(batch_times)} batches (outliers removed)")
                    logger.info(f"   Removed: min={sorted_times[0]:.3f}s, max={sorted_times[-1]:.3f}s")
                else:
                    avg_batch_time = np.mean(batch_times)
                    logger.info(f"   Using all {len(batch_times)} batches (too few to remove outliers)")
                
                # Calculate batch time statistics
                batch_std = np.std(batch_times)
                batch_min = np.min(batch_times)
                batch_max = np.max(batch_times)
                
                total_batches = len(train_idx_data_loader)
                estimated_epoch_time = avg_batch_time * total_batches
                estimated_total_time = estimated_epoch_time * args.num_epochs
                
                logger.info(f"\n📊 IMPROVED TRAINING TIME ESTIMATION:")
                logger.info(f"   Average batch time: {avg_batch_time:.3f}s (std: {batch_std:.3f}s)")
                logger.info(f"   Batch time range: {batch_min:.3f}s - {batch_max:.3f}s")
                logger.info(f"   Warm-up batches: {warmup_batches}")
                logger.info(f"   Timing batches: {len(batch_times)}")
                logger.info(f"   Total batches per epoch: {total_batches}")
                logger.info(f"   Estimated time per epoch: {estimated_epoch_time/60:.1f} minutes")
                logger.info(f"   Estimated total training time: {estimated_total_time/3600:.1f} hours")
                logger.info(f"   Dataset: {args.dataset_name}")
                logger.info(f"   Model: {args.model_name}")
                logger.info(f"   Time encoder: {args.time_encoder_type}")
                logger.info(f"   Batch size: {args.batch_size}")
                logger.info(f"   Data ratio: {args.data_ratio}")
                
                # Save estimation to file with improved metrics
                estimation_data = {
                    "dataset": args.dataset_name,
                    "model": args.model_name,
                    "time_encoder": args.time_encoder_type,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "data_ratio": args.data_ratio,
                    "total_batches": total_batches,
                    "warmup_batches": warmup_batches,
                    "timing_batches": len(batch_times),
                    "sample_batch_times_seconds": batch_times,
                    "avg_batch_time_seconds": avg_batch_time,
                    "batch_time_std_seconds": batch_std,
                    "batch_time_min_seconds": batch_min,
                    "batch_time_max_seconds": batch_max,
                    "estimated_epoch_time_minutes": estimated_epoch_time / 60,
                    "estimated_total_time_hours": estimated_total_time / 3600,
                    "estimated_total_time_days": estimated_total_time / (3600 * 24),
                    "timestamp": datetime.now().isoformat(),
                    "training_data_size": len(train_data.src_node_ids),
                    "full_data_size": len(full_data.src_node_ids),
                    "estimation_improved": True,  # Flag to indicate this uses improved estimation
                    "outliers_removed": len(batch_times) >= 5  # Whether outliers were removed
                }
                
                os.makedirs("./time_estimates", exist_ok=True)
                estimate_file = f"./time_estimates/{args.model_name}_{args.time_encoder_type}_{args.dataset_name}_dr{args.data_ratio}_estimate.json"
                
                with open(estimate_file, 'w') as f:
                    json.dump(estimation_data, f, indent=2)
                
                logger.info(f"💾 Time estimation saved to: {estimate_file}")
                
                # Display improved summary table
                print(f"\n" + "="*70)
                print(f"{'IMPROVED TRAINING TIME ESTIMATE SUMMARY':^70}")
                print(f"="*70)
                print(f"Dataset: {args.dataset_name}")
                print(f"Model: {args.model_name}")  
                print(f"Time Encoder: {args.time_encoder_type}")
                print(f"Data Ratio: {args.data_ratio}")
                print(f"Batch Size: {args.batch_size}")
                print(f"Total Epochs: {args.num_epochs}")
                print(f"-"*70)
                print(f"Training Data Size: {len(train_data.src_node_ids):,} edges")
                print(f"Batches per Epoch: {total_batches:,}")
                print(f"Warm-up Batches: {warmup_batches}")
                print(f"Timing Batches: {len(batch_times)}")
                print(f"-"*70)
                print(f"Average Batch Time: {avg_batch_time:.3f}s ± {batch_std:.3f}s")
                print(f"Batch Time Range: {batch_min:.3f}s - {batch_max:.3f}s")
                print(f"-"*70)
                print(f"Estimated Time per Epoch: {estimated_epoch_time/60:.1f} minutes")
                print(f"Estimated Total Time: {estimated_total_time/3600:.1f} hours")
                print(f"                      ({estimated_total_time/(3600*24):.1f} days)")
                
                # Add warning for very high estimates
                if estimated_total_time > 7 * 24 * 3600:  # More than 7 days
                    print(f"⚠️  WARNING: Very long training time estimated!")
                    print(f"⚠️  Consider reducing epochs, batch size, or data ratio.")
                elif estimated_total_time > 24 * 3600:  # More than 1 day
                    print(f"⚠️  Note: Long training time - plan accordingly.")
                else:
                    print(f"✅ Reasonable training time estimated.")
                    
                print(f"="*70)
                
            logger.info("🔚 Improved estimation complete, exiting")
            sys.exit(0)
            # ===== END IMPROVED TRAINING TIME ESTIMATION =====
