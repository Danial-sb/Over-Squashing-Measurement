import argparse
from attrdict import AttrDict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MUTAG", help='Dataset name')
    parser.add_argument('--layer_type', type=str, default='GIN', help='Layer type')
    parser.add_argument('--rewiring', type=str, default='fosr', choices=["rw", "none", "sdrf", "digl", "fosr", "borf"], help='type of rewiring to be performed'),
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64, 64, 64], help='Hidden layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

    parser.add_argument('--cuda_device', type=int, default=2, help="CUDA device to use e.g. 0")
    parser.add_argument('--train_fraction', type=float, default=0.8, help='Fraction of training data')
    parser.add_argument('--validation_fraction', type=float, default=0.1, help='Fraction of validation data')
    parser.add_argument('--test_fraction', type=float, default=0.1, help='Fraction of test data')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--input_dim', type=int, default=None, help='input dimension')
    parser.add_argument('--batch_size', type=int, default=16, choices=[16, 32, 64], help='Batch size') # MUTAG:16, ENZYMES:32, PROTEINS:64
    parser.add_argument('--batch_norm', type=bool, default=False, choices=[True, False], help='Batch normalization')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--stopping_criterion', type=str, default='validation', help='Stopping criterion')
    parser.add_argument('--eval_every', type=int, default=1, help='calculate validation/test accuracy every X epochs')
    parser.add_argument('--stopping_threshold', type=float, default=1.01, help='Stopping threshold')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--num_relations', type=int, default=2, help='num_relations')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--display', action='store_true', help='Display print statements')
    parser.add_argument('--separate_gnns', type=bool, default=False, choices=[True, False], help='Separate GNNs for each relation')
    parser.add_argument('--last_layer_fa', type=bool, default=False, choices=[True, False], help='Last layer fully adjacent')
    # DIGL args
    parser.add_argument('--alpha', type=float, default=0.05, help='alpha hyperparameter for DIGL')
    parser.add_argument('--k', type=int, default=None, help='k hyperparameter for DIGL')
    parser.add_argument('--eps', type=float, default=0.001, help='epsilon hyperparameter for DIGL')
    # BORF args
    parser.add_argument('--num_iterations', type=int, default=3, help='num_iterations')
    parser.add_argument('--borf_batch_add', default=30, type=int)
    parser.add_argument('--borf_batch_remove', default=30, type=int)
    # wandb
    parser.add_argument('--wandb', default=True, action='store_true', help="flag if logging to wandb")
    parser.add_argument('--wandb_sweep', action='store_true',
                        help="flag if sweeping")  # if not it picks up params in greed_params
    parser.add_argument('--wandb_entity', default="danial-saber", type=str)
    parser.add_argument('--wandb_project', default="Random-Wiring-New", type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--run_track_reports', action='store_true', help="run_track_reports")
    parser.add_argument('--save_wandb_reports', action='store_true', help="save_wandb_reports")
    args = parser.parse_args()
    return AttrDict(vars(args))
