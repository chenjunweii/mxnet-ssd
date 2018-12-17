import mxnet as mx
from mxnet import nd
from symbol.symbol_factory_pretrain import get_symbol

import argparse

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')

parser.add_argument('-n', '--network', type = str)

parser.add_argument('-s', '--size', type = int)

parser.add_argument('-b', '--batch', type = int)

opt = parser.parse_args()

net = get_symbol(opt.network, opt.size, num_classes = 1000)

args_shape, out_shape, aux_shape = net.infer_shape(data = (opt.batch, 3, opt.size, opt.size))

args = net.list_arguments()

aux = net.list_auxiliary_states()

#print(args)

#print(len(args))

#print(len(args_shape))

nds = dict()

w = mx.initializer.MSRAPrelu()

init = mx.initializer.Initializer()

for node, shape in zip(args, args_shape):

    if node == 'data':

        continue

    nds[node] = nd.zeros(shape)

    if node.endswith('weight'):

        w._init_weight(node, nds[node])
    
    elif node.endswith('bias'):

        w._init_beta(node, nds[node])

    elif node.endswith('gamma') or node.endswith('beta'):

        continue

    else:

        print(node)

        raise ValueError('')

for node, shape in zip(aux, aux_shape):

    if node == 'data':

        continue
    
    nds[node] = nd.zeros(shape)

    if node.endswith('gamma') or node.endswith('mean') or node.endswith('var'):
        
        init._init_one(node, nds[node])
    
    elif node.endswith('beta'):

        init._init_zero(node, nds[node])
    
    elif node.endswith('weight') or node.endswith('bias'):

        continue

    else:

        print(node)

        raise ValueError('')

    #if node.endswith('weight'):

nd.save('{}_init_for_imagenet.params'.format(opt.network), nds)

net.save('{}_init_for_imagenet.json'.format(opt.network))

