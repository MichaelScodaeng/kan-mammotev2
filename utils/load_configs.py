import argparse
import sys
import torch


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
                                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGMamba', help='name of the model',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGMamba'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='uniform', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    
    # Time encoder arguments
    parser.add_argument('--time_encoder_type', type=str, default='original', 
                        choices=['original', 'lete', 'KMM', 'time2vec', 'kan_mammote'], 
                        help='type of time encoder to use')
    parser.add_argument('--expert_dim', type=int, default=128, 
                        help='dimension of each expert in K-MOTE (for KMM encoder)')
    parser.add_argument('--num_mixtures', type=int, default=16, 
                        help='number of mixture components in SM-Kernel (for KMM encoder)')
    parser.add_argument('--sort_neighbors_by_time', action='store_true', default=True,
                        help='Sort sampled neighbors chronologically for Mamba-based time encoders (recommended for KMM)')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate for GNN backbone')
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='dropout rate for time encoder (e.g., KMM)')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='AdamW8bit', choices=['SGD', 'Adam', 'RMSprop',"AdamW","Adam8bit","AdamW8bit"], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum gradient norm for clipping (0 disables clipping)')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=3, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=5, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--max_interaction_times', type=int, default=10,
                        help='max interactions for src and dst to consider')
    parser.add_argument('--load_best_configs', action='store_true', default=True, help='whether to load the best configurations')
    
    # Data and experiment control arguments
    parser.add_argument('--data_ratio', type=float, default=1.0, 
                        help='ratio of data to use for experiments (e.g., 0.1 for 10%% of data)')
    parser.add_argument('--train_only_ratio', action='store_true', default=False,
                        help='Apply data_ratio only to training set, keep val/test at 100%% (recommended for hyperparameter tuning)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducible data splits and sampling (default: 42)')
    parser.add_argument('--save_model_name_suffix', type=str, default='', 
                        help='suffix to add to saved model names for ablation studies')
    parser.add_argument('--ablation_dir', type=str, default='', 
                        help='directory to save all ablation study outputs (models, metrics, results)')
    
    # Mamba-specific arguments for KAN-MAMMOTE
    parser.add_argument('--mamba_d_state', type=int, default=128, 
                        help='Mamba state dimension (for KMM encoder)')
    parser.add_argument('--mamba_d_conv', type=int, default=4, 
                        help='Mamba convolution dimension (for KMM encoder)')
    parser.add_argument('--mamba_expand', type=int, default=4, 
                        help='Mamba expansion factor (for KMM encoder)')
    parser.add_argument('--mamba_headdim', type=int, default=64, 
                        help='Mamba head dimension (for KMM encoder)')
    
    # Arguments specific to KAN-MAMMOTE
    
    # Debug arguments
    parser.add_argument('--debug_encoder', action='store_true', default=False,
                        help='Enable comprehensive debugging for time encoders (especially KAN-MAMMOTE)')
    
    # Checkpoint and resuming arguments
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='path to checkpoint file to resume training from')
    parser.add_argument('--save_checkpoints', action='store_true', default=False,
                        help='save training checkpoints every few epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='save checkpoint every N epochs (default: 5)')
    parser.add_argument('--start_from_seed', type=int, default=0,
                        help='start training from specific seed/run (for resuming)')
    parser.add_argument('--max_checkpoints_to_keep', type=int, default=3,
                        help='maximum number of recent checkpoints to keep (default: 3)')
    parser.add_argument('--checkpoint_strategy', type=str, default='smart', 
                        choices=['frequent', 'smart', 'minimal'],
                        help='checkpoint frequency strategy')
    parser.add_argument('--validate_checkpoints', action='store_true', default=True,
                        help='validate checkpoint integrity before loading (recommended)')
    parser.add_argument('--disable_progress_bar', action='store_true', default=False,
                        help='disable tqdm progress bars (useful for logging to files in batch jobs)')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        print(f'Using device: {args.device}')
    except:
        parser.print_help()
        sys.exit()

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['enron']:
            args.dropout = 0.2
        elif args.dataset_name in ['CanParl']:
            args.dropout = 0.2
        elif args.dataset_name in ['Contacts']:
            args.dropout = 0.1
        elif args.dataset_name in ['Flights']:
            args.dropout = 0.1
        elif args.dataset_name in ['UNtrade']:
            args.dropout = 0.1
        elif args.dataset_name in ['UNvote']:
            args.dropout = 0.2
        elif args.dataset_name in ['USLegis']:
            args.dropout = 0.1
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['CanParl']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['UNtrade']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['UNvote']:
            args.sample_neighbor_strategy = 'recent'
        elif args.dataset_name in ['Contacts', 'Flights', 'USLegis']:
            args.sample_neighbor_strategy = 'recent'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        args.num_neighbors = 10
        args.num_layers = 1
        if args.model_name == 'JODIE':
            if args.dataset_name in ['mooc']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm']:
                args.dropout = 0.3
            elif args.dataset_name in ['uci']:
                args.dropout = 0.4
            elif args.dataset_name in ['CanParl']:
                args.dropout = 0.0
            elif args.dataset_name in ['Contacts']:
                args.dropout = 0.1
            elif args.dataset_name in ['Flights']:
                args.dropout = 0.1
            elif args.dataset_name in ['UNtrade']:
                args.dropout = 0.4
            elif args.dataset_name in ['UNvote']:
                args.dropout = 0.1
            elif args.dataset_name in ['USLegis']:
                args.dropout = 0.2
            else:
                args.dropout = 0.1
        elif args.model_name == 'DyRep':
            if args.dataset_name in ['mooc', 'lastfm', 'enron', 'uci']:
                args.dropout = 0.0
            elif args.dataset_name in ['CanParl']:
                args.dropout = 0.0
            elif args.dataset_name in ['Contacts']:
                args.dropout = 0.0
            elif args.dataset_name in ['Flights']:
                args.dropout = 0.1
            elif args.dataset_name in ['UNtrade']:
                args.dropout = 0.1
            elif args.dataset_name in ['UNvote']:
                args.dropout = 0.1
            elif args.dataset_name in ['USLegis']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        else:
            assert args.model_name == 'TGN'
            if args.dataset_name in ['mooc']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm']:
                args.dropout = 0.3
            elif args.dataset_name in ['enron', 'SocialEvo']:
                args.dropout = 0.0
            elif args.dataset_name in ['CanParl']:
                args.dropout = 0.3
            elif args.dataset_name in ['Contacts']:
                args.dropout = 0.1
            elif args.dataset_name in ['Flights']:
                args.dropout = 0.1
            elif args.dataset_name in ['UNtrade']:
                args.dropout = 0.2
            elif args.dataset_name in ['UNvote']:
                args.dropout = 0.1
            elif args.dataset_name in ['USLegis']:
                args.dropout = 0.1
            else:
                args.dropout = 0.1
        if args.model_name in ['TGN', 'DyRep']:
            if args.dataset_name in ['CanParl']:
                args.sample_neighbor_strategy = 'uniform'
            elif args.dataset_name in ['UNtrade']:
                args.sample_neighbor_strategy = 'uniform'
            elif args.dataset_name in ['UNvote']:
                args.sample_neighbor_strategy = 'uniform'
            else:
                args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'CAWN':
        args.time_scaling_factor = 1e-6
        if args.dataset_name in ['mooc', 'SocialEvo', 'uci']:
            args.num_neighbors = 64
        elif args.dataset_name in ['lastfm']:
            args.num_neighbors = 128
        elif args.dataset_name in ['CanParl']:
            args.num_neighbors = 128
        elif args.dataset_name in ['Contacts']:
            args.num_neighbors = 64
        elif args.dataset_name in ['Flights']:
            args.num_neighbors = 64
        elif args.dataset_name in ['UNtrade']:
            args.num_neighbors = 64
        elif args.dataset_name in ['UNvote']:
            args.num_neighbors = 64
        elif args.dataset_name in ['USLegis']:
            args.num_neighbors = 32
        else:
            args.num_neighbors = 32
        if args.dataset_name in ['CanParl']:
            args.dropout = 0.0
        elif args.dataset_name in ['Contacts']:
            args.dropout = 0.1
        elif args.dataset_name in ['Flights']:
            args.dropout = 0.1
        elif args.dataset_name in ['UNtrade']:
            args.dropout = 0.1
        elif args.dataset_name in ['UNvote']:
            args.dropout = 0.1
        elif args.dataset_name in ['USLegis']:
            args.dropout = 0.1
        else:
            args.dropout = 0.1
        args.sample_neighbor_strategy = 'time_interval_aware'
    elif args.model_name == 'TCL':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['SocialEvo', 'uci']:
            args.dropout = 0.0
        elif args.dataset_name in ['CanParl']:
            args.dropout = 0.2
        elif args.dataset_name in ['Contacts']:
            args.dropout = 0.0
        elif args.dataset_name in ['Flights']:
            args.dropout = 0.1
        elif args.dataset_name in ['UNtrade']:
            args.dropout = 0.0
        elif args.dataset_name in ['UNvote']:
            args.dropout = 0.0
        elif args.dataset_name in ['USLegis']:
            args.dropout = 0.3
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['CanParl']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['USLegis']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['UNtrade']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['UNvote']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['Contacts', 'Flights']:
            args.sample_neighbor_strategy = 'recent'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        if args.dataset_name in ['wikipedia']:
            args.num_neighbors = 30
        elif args.dataset_name in ['reddit', 'lastfm']:
            args.num_neighbors = 10
        elif args.dataset_name in ['CanParl']:
            args.num_neighbors = 20
        elif args.dataset_name in ['Contacts']:
            args.num_neighbors = 20
        elif args.dataset_name in ['Flights']:
            args.num_neighbors = 20
        elif args.dataset_name in ['UNtrade']:
            args.num_neighbors = 20
        elif args.dataset_name in ['UNvote']:
            args.num_neighbors = 20
        elif args.dataset_name in ['USLegis']:
            args.num_neighbors = 20
        else:
            args.num_neighbors = 20
        if args.dataset_name in ['wikipedia', 'reddit', 'enron']:
            args.dropout = 0.5
        elif args.dataset_name in ['mooc', 'uci']:
            args.dropout = 0.4
        elif args.dataset_name in ['lastfm']:
            args.dropout = 0.0
        elif args.dataset_name in ['SocialEvo']:
            args.dropout = 0.3
        elif args.dataset_name in ['CanParl']:
            args.dropout = 0.2
        elif args.dataset_name in ['Contacts']:
            args.dropout = 0.1
        elif args.dataset_name in ['Flights']:
            args.dropout = 0.2
        elif args.dataset_name in ['UNtrade']:
            args.dropout = 0.1
        elif args.dataset_name in ['UNvote']:
            args.dropout = 0.0
        elif args.dataset_name in ['USLegis']:
            args.dropout = 0.4
        else:
            args.dropout = 0.1
        if args.dataset_name in ['CanParl']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['UNtrade']:
            args.sample_neighbor_strategy = 'uniform'
        elif args.dataset_name in ['UNvote']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ['mooc', 'enron']:
            args.max_input_sequence_length = 256
            args.patch_size = 4
        elif args.dataset_name in ['lastfm']:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ['CanParl']:
            args.max_input_sequence_length = 2048
            args.patch_size = 64
        elif args.dataset_name in ['Contacts']:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        elif args.dataset_name in ['Flights']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ['UNtrade']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ['UNvote']:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        elif args.dataset_name in ['USLegis']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        else:
            args.max_input_sequence_length = 32 
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        elif args.dataset_name in ['enron']:
            args.dropout = 0.0
        elif args.dataset_name in ['CanParl']:
            args.dropout = 0.1
        elif args.dataset_name in ['Contacts']:
            args.dropout = 0.0
        elif args.dataset_name in ['Flights']:
            args.dropout = 0.1
        elif args.dataset_name in ['UNtrade']:
            args.dropout = 0.0
        elif args.dataset_name in ['UNvote']:
            args.dropout = 0.2
        elif args.dataset_name in ['USLegis']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
    #need hyperparams tuning on new datasets
    elif args.model_name == 'DyGMamba':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ['mooc']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif  args.dataset_name in ['enron']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ['lastfm']:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ['CanParl']:
            args.max_input_sequence_length = 2048
            args.patch_size = 64
        elif args.dataset_name in ['Contacts']:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        elif args.dataset_name in ['Flights']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ['UNtrade']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ['UNvote']:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        elif args.dataset_name in ['USLegis']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        else:
            args.max_input_sequence_length = 32 
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        elif args.dataset_name in ['enron']:
            args.dropout = 0.0
        elif args.dataset_name in ['CanParl']:
            args.dropout = 0.1
        elif args.dataset_name in ['Contacts']:
            args.dropout = 0.0
        elif args.dataset_name in ['Flights']:
            args.dropout = 0.1
        elif args.dataset_name in ['UNtrade']:
            args.dropout = 0.0
        elif args.dataset_name in ['UNvote']:
            args.dropout = 0.2
        elif args.dataset_name in ['USLegis']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        if args.dataset_name in ['enron']:
            args.max_interaction_times = 30
        elif args.dataset_name in ['mooc','lastfm']:
            args.max_interaction_times = 10
        elif args.dataset_name in ['CanParl']:
            args.max_interaction_times = 100
        elif args.dataset_name in ['Contacts']:
            args.max_interaction_times = 5
        elif args.dataset_name in ['Flights']:
            args.max_interaction_times = 30
        elif args.dataset_name in ['UNtrade']:
            args.max_interaction_times = 30
        elif args.dataset_name in ['UNvote']:
            args.max_interaction_times = 10
        elif args.dataset_name in ['USLegis']:
            args.max_interaction_times = 30
        else:
            args.max_interaction_times = 5
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")

