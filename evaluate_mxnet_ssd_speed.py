import os
import cv2
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.model import load_checkpoint
from time import time
import argparse

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--network', type = str, default = None, help = 'Network Architecture')
parser.add_argument('--dtype', type = str, default = 'float32', help = 'Data Type')
parser.add_argument('--tensorrt', action = 'store_true', help = 'if use tensorrt')
parser.add_argument('--size', type = int, help = 'Input Size')
parser.add_argument('--epoch', type = int, help = 'Epoch')
args = parser.parse_args()


ctx = mx.gpu(0)

shapes = (1, 3, args.size, args.size)

if args.network == 'mobilenetv2':

    net, arg, aux = load_checkpoint('model/deploy_ssd_mobilenet_v2_680', args.epoch)

elif args.network == 'inceptionv3' : 
    
    net, arg, aux = load_checkpoint('model/deploy_ssd_inceptionv3_512', args.epoch)

elif args.network == 'inceptionv3_fp16' : 
    
    net, arg, aux = load_checkpoint('model/deploy_ssd_inceptionv3_fp16_512', args.epoch)

else:

    raise ValueError("Network {} is not supported".format(args.network))

r, g, b = 123, 117, 104 

inputs = nd.array(np.ones(shapes), ctx)

executor = None

if args.tensorrt:

    os.environ['MXNET_USE_TENSORRT'] = '1'

    all_params = dict()

    all_params.update(arg); all_params.update(aux)
    
    all_params = dict([(k, v.as_in_context(ctx)) for k, v in all_params.items()])
    
    executor = mx.contrib.tensorrt.tensorrt_bind(net,
                                                ctx = ctx,
                                                all_params = all_params,
                                                data = shapes,
                                                grad_req = 'null',
                                                force_rebind = True)

else:

    os.environ['MXNET_USE_TENSORRT'] = '0'

    net = net.get_children()[0]

    executor = net.simple_bind(ctx = ctx,
                               data = shapes,
                               grad_req = 'null',
                               force_rebind = True)

#executor.copy_params_from(arg, aux)

for i in range(10):

    start = time()

    out = executor.forward(is_train = False, data = inputs)

    out[0].asnumpy()

    print('Time Cost : ', time() - start)

print('[!] Evaluation Done')



