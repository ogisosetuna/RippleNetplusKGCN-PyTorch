import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(150)
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='numbers of each ripple set')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--n_heads', type=int, default=1, help='heads of self attention')
parser.add_argument('--feed_f_dim', type=int, default=16, help='dim of feed forward network in transformer')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
'''
# default settings for Book-Crossing
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=40, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--n_heads', type=int, default=1, help='heads of self attention')
parser.add_argument('--feed_f_dim', type=int, default=16, help='dim of feed forward network in transformer')
'''
# default settings for last fm 50

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='numbers of each ripple set')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--n_heads', type=int, default=1, help='heads of self attention')
parser.add_argument('--feed_f_dim', type=int, default=16, help='dim of feed forward network in transformer')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')

args = parser.parse_args()

show_loss = False
data_info = load_data(args)
train(args, data_info, show_loss)
