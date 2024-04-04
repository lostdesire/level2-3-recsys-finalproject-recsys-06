import torch
import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

## 각종 파라미터 세팅
def argparsing():

    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

    parser.add_argument('--data_dir', type=str, default='/home/wooksbaby/boostcamp6th/CteeEDA/api/app/models/EASE/EASE/data/')
    parser.add_argument('--raw_data', type=str, default='orders_items_total.json',
                        help='raw data')
    parser.add_argument('--data', type=str, default='train.csv',
                        help='preprocessed data')

    parser.add_argument('--heldout_users', type=int, default=3000)

    parser.add_argument('--lr', type=float, default=5*1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    
    parser.add_argument('--model', type=str, default='EASE',
                        help='select model')
    
    parser.add_argument('--encoder_epochs', type=int, default=3,
                        help='num of encoder epochs per decoder epoch')
    parser.add_argument('--beta', type=str2bool, default='False',
                        help='recvae composite prior weight')
    parser.add_argument('--gamma', type=float, default=0.005,
                        help='recvae composite prior rebalancing parameter')
    parser.add_argument('--hidden-dim', type=int, default=600)
    parser.add_argument('--latent-dim', type=int, default=200)

    args = parser.parse_args()
    args.pro_dir = args.data_dir

    args.model_path = os.path.join(args.pro_dir, 'model_files')

    if torch.cuda.is_available():
        args.cuda = True
    args.device = torch.device("cuda" if args.cuda else "cpu")

    args.is_VAE = True if args.model=='MultiVAE' else False

    if args.model=='EASE':
        args.epochs = 1

    return args

# Set the random seed manually for reproductibility.
