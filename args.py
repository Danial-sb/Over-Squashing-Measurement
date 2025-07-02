import argparse
from attrdict import AttrDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MUTAG", help='Dataset name')
    parser.add_argument('--layer_type', type=str, default='R-GCN', help='Layer type')
    parser.add_argument('--rewiring', type=str, default='digl',
                        choices=["sdrf", "digl", "fosr", "borf"], help='Type of rewiring method')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--display', action='store_true', help='Display config info')

    # DIGL
    parser.add_argument('--alpha', type=float, default=0.05, help='Alpha hyperparameter for DIGL')
    parser.add_argument('--k', type=int, default=None, help='K hyperparameter for DIGL')
    parser.add_argument('--eps', type=float, default=0.001, help='Epsilon hyperparameter for DIGL')

    # BORF
    parser.add_argument('--num_iterations', type=int, default=3, help='Number of iterations for rewiring')
    parser.add_argument('--borf_batch_add', type=int, default=30, help='BORF batch add size')
    parser.add_argument('--borf_batch_remove', type=int, default=30, help='BORF batch remove size')

    args = parser.parse_args()
    return populate_defaults(args)


def get_args():
    return parse_args()


def populate_defaults(args):
    args = AttrDict(vars(args)) if not isinstance(args, AttrDict) else args

    arch = args.layer_type.upper()
    dataset = args.dataset.upper()

    args.layer_type = arch
    args.dataset = dataset

    # -------------------- FoSR --------------------
    FOSR_NUM_ITERATIONS = {
        'GCN': {
            'REDDIT-BINARY': 5, 'IMDB-BINARY': 5, 'MUTAG': 40,
            'ENZYMES': 10, 'PROTEINS': 20, 'COLLAB': 10,
            'CORA': 150, 'CITESEER': 100, 'TEXAS': 50,
            'CORNELL': 125, 'WISCONSIN': 175, 'CHAMELEON': 50
        },
        'R-GCN': {
            'REDDIT-BINARY': 5, 'IMDB-BINARY': 20, 'MUTAG': 40,
            'ENZYMES': 40, 'PROTEINS': 5, 'COLLAB': 5
        },
        'GIN': {
            'REDDIT-BINARY': 10, 'IMDB-BINARY': 20, 'MUTAG': 20,
            'ENZYMES': 5, 'PROTEINS': 10, 'COLLAB': 20,
            'CORA': 50, 'CITESEER': 200, 'TEXAS': 150,
            'CORNELL': 75, 'WISCONSIN': 25, 'CHAMELEON': 25
        },
        'R-GIN': {
            'REDDIT-BINARY': 40, 'IMDB-BINARY': 20, 'MUTAG': 5,
            'ENZYMES': 40, 'PROTEINS': 20, 'COLLAB': 10
        }
    }

    if args.rewiring == "fosr":
        args.num_iterations = FOSR_NUM_ITERATIONS.get(arch, {}).get(dataset, args.num_iterations)

    # -------------------- SDRF --------------------
    SDRF_NUM_ITERATIONS = {
        'GCN': {
            'REDDIT-BINARY': 5, 'IMDB-BINARY': 20, 'MUTAG': 5,
            'ENZYMES': 5, 'PROTEINS': 40, 'COLLAB': 5,
            'CORA': 12, 'CITESEER': 175, 'TEXAS': 87,
            'CORNELL': 100, 'WISCONSIN': 25, 'CHAMELEON': 50
        },
        'R-GCN': {
            'REDDIT-BINARY': 40, 'IMDB-BINARY': 5, 'MUTAG': 40,
            'ENZYMES': 5, 'PROTEINS': 20, 'COLLAB': 20
        },
        'GIN': {
            'REDDIT-BINARY': 5, 'IMDB-BINARY': 10, 'MUTAG': 5,
            'ENZYMES': 5, 'PROTEINS': 20, 'COLLAB': 40,
            'CORA': 50, 'CITESEER': 25, 'TEXAS': 37,
            'CORNELL': 25, 'WISCONSIN': 150, 'CHAMELEON': 87
        },
        'R-GIN': {
            'REDDIT-BINARY': 5, 'IMDB-BINARY': 40, 'MUTAG': 5,
            'ENZYMES': 5, 'PROTEINS': 5, 'COLLAB': 20
        }
    }

    if args.rewiring == "sdrf":
        args.num_iterations = SDRF_NUM_ITERATIONS.get(arch, {}).get(dataset, args.num_iterations)

    # -------------------- DIGL --------------------
    DIGL_ALPHA = {
        'GCN': {'REDDIT-BINARY': 0.15, 'IMDB-BINARY': 0.05, 'MUTAG': 0.15,
                'ENZYMES': 0.15, 'PROTEINS': 0.15, 'COLLAB': 0.05,
                'CORA': 0.0773, 'CITESEER': 0.1076, 'TEXAS': 0.0206,
                'CORNELL': 0.1795, 'WISCONSIN': 0.1246, 'CHAMELEON': 0.0244},
        'R-GCN': {'REDDIT-BINARY': 0.15, 'IMDB-BINARY': 0.05, 'MUTAG': 0.05,
                  'ENZYMES': 0.15, 'PROTEINS': 0.05, 'COLLAB': 0.05},
        'GIN': {'REDDIT-BINARY': 0.05, 'IMDB-BINARY': 0.15, 'MUTAG': 0.05,
                'ENZYMES': 0.15, 'PROTEINS': 0.15, 'COLLAB': 0.15},
        'R-GIN': {'REDDIT-BINARY': 0.05, 'IMDB-BINARY': 0.15, 'MUTAG': 0.05,
                  'ENZYMES': 0.05, 'PROTEINS': 0.05, 'COLLAB': 0.05}
    }

    DIGL_EPSILON = {
        'GCN': {'REDDIT-BINARY': 1e-3, 'IMDB-BINARY': 1e-4, 'MUTAG': 1e-3,
                'ENZYMES': 1e-4, 'PROTEINS': 1e-4, 'COLLAB': 1e-4,
                'CORA': None, 'CITESEER': 0.0008, 'TEXAS': None,
                'CORNELL': None, 'WISCONSIN': 0.0001, 'CHAMELEON': None},
        'R-GCN': {'REDDIT-BINARY': 1e-4, 'IMDB-BINARY': 1e-3, 'MUTAG': 1e-4,
                  'ENZYMES': 1e-3, 'PROTEINS': 1e-3, 'COLLAB': 1e-3},
        'GIN': {'REDDIT-BINARY': 1e-3, 'IMDB-BINARY': 1e-4, 'MUTAG': 1e-4,
                'ENZYMES': 1e-3, 'PROTEINS': 1e-3, 'COLLAB': 1e-4},
        'R-GIN': {'REDDIT-BINARY': 1e-3, 'IMDB-BINARY': 1e-4, 'MUTAG': 1e-3,
                  'ENZYMES': 1e-4, 'PROTEINS': 1e-4, 'COLLAB': 1e-3}
    }

    DIGL_K = {
        'GCN': {'CORA': 128, 'CITESEER': None, 'TEXAS': 32,
                'CORNELL': 64, 'WISCONSIN': None, 'CHAMELEON': 64}
    }

    if args.rewiring == "digl":
        args.alpha = DIGL_ALPHA.get(arch, {}).get(dataset, args.alpha)
        args.eps = DIGL_EPSILON.get(arch, {}).get(dataset, args.eps)
        if arch == 'GCN' and dataset in DIGL_K['GCN']:
            args.k = DIGL_K['GCN'][dataset]

    # -------------------- BORF --------------------
    BORF_PARAMS = {
        'GCN': {
            'CORA': (3, 20, 10), 'CITESEER': (3, 20, 10),
            'TEXAS': (3, 30, 10), 'CORNELL': (3, 20, 30),
            'WISCONSIN': (2, 30, 20), 'CHAMELEON': (3, 20, 20),
            'ENZYMES': (1, 3, 2), 'IMDB': (1, 3, 0),
            'MUTAG': (1, 20, 3), 'PROTEINS': (3, 4, 1)
        },
        'GIN': {
            'CORA': (3, 20, 30), 'CITESEER': (3, 10, 20),
            'TEXAS': (1, 20, 10), 'CORNELL': (3, 10, 20),
            'WISCONSIN': (2, 50, 30), 'CHAMELEON': (3, 30, 30),
            'ENZYMES': (3, 3, 1), 'IMDB': (1, 4, 2),
            'MUTAG': (1, 3, 1), 'PROTEINS': (2, 4, 3)
        }
    }

    if args.rewiring == "borf":
        if arch in BORF_PARAMS and dataset in BORF_PARAMS[arch]:
            args.num_iterations, args.borf_batch_add, args.borf_batch_remove = BORF_PARAMS[arch][dataset]

    if args.display:
        print(f"\n[INFO] Final arguments after populating defaults for ({arch}, {dataset}):")
        for k, v in args.items():
            print(f"  {k}: {v}")

    return args
