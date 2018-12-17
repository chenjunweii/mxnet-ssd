import numpy as np
from mxnet import nd
from gluoncv import model_zoo

scale = 2

nds_p = nd.load('mobilenet_v2-0000.params')

#for n in nds_p.keys():
    
#    print('{} : {}'.format(n, nds_p[n].shape))

mobilenet = model_zoo.get_model('mobilenetv2_1.0', pretrained = True)

mobilenet.save_params('__mobilenet.params')

mobilenet_scale = model_zoo.get_model('mobilenetv2_{:.1f}'.format(min(1, scale)), pretrained = True)

mobilenet_scale.save_params('__mobilenet_scale.params')

nds_zoo = nd.load('__mobilenet.params')

nds_scale = nd.load('model/ssd_mobilenet_v2_680-0239.params')# load from trained params

#nds_scale = nd.load('__mobilenet_scale.params') # load from model zoo scale version

nds_target = dict()

#for n_z in nds_zoo.keys():

#    print('{} => {}'.format(n_z, nds_zoo[n_z].shape))

p2z = dict()

p2z['conv6_3_linear'] = 'linearbottleneck16_conv2'

p2z['conv6_2_expand_weight'] = 'linearbottleneck16_conv0'

p2z['conv6_3_dwise_weight'] = 'linearbottleneck16_conv1_weight'

p2z['bn'] = 'batchnorm'

processed = []

layers = []

for n_p in nds_p.keys():

    n = n_p.split(':')[-1]

    if 'arg' in n_p:

        layer_n, n2 = n.split('_')[:2]

        if 'dwise' in n:

            print('{} => {}'.format(n, nds_p[n_p].shape))

        l = '{}_{}'.format(layer_n, n2) if n2.isdigit() else layer_n

        if 'conv' in layer_n and l not in layers:

            layers.append(l)

        elif 'fc' in layer_n and l not in layers:

            layers.append(l)

    elif 'aux' in n_p:
       
        pass

layers.sort()

bn_n = 0


bn = 'batchnorm'

w = []; gamma = []; beta = []; mean = []; var = []; b = [];

pw = []; pgamma = []; pbeta = []; pmean = []; pvar = []; pb = [];

for l in layers:

    if 'conv1' in l or 'conv6_4' in l:

        d = dict()

        d['conv1'] = '0'

        d['conv6_4'] = '1'
        
        f = 'features'

        w.append('{}_conv{}_weight'.format(f, d[l]))
        gamma.append('{}_{}{}_gamma'.format(f, bn, d[l]))
        beta.append('{}_{}{}_beta'.format(f, bn, d[l]))
        mean.append('{}_{}{}_running_mean'.format(f, bn, d[l]))
        var.append('{}_{}{}_running_var'.format(f, bn, d[l]))

        pw.append('arg:{}_weight'.format(l))
        pbeta.append('arg:{}_bn_beta'.format(l))
        pgamma.append('arg:{}_bn_gamma'.format(l))
        pmean.append('aux:{}_bn_moving_mean'.format(l))
        pvar.append('aux:{}_bn_moving_var'.format(l))


    elif 'fc' in l:

        w.append('output_pred_weight')
        pw.append('arg:{}_weight'.format(l))

    else:

        pblock = ['expand', 'dwise', 'linear']

        args = ['{}_conv{}_weight', '{}_{}{}_gamma', '{}_{}{}_beta', '{}_{}{}_running_mean', '{}_{}{}_running_var']

        n = 'bottleneck{}'.format(bn_n)

        for i in range(3):
            
            f = 'features_linearbottleneck{}'.format(bn_n)
            w.append(args[0].format(f, i))
            gamma.append(args[1].format(f, bn, i))
            beta.append(args[2].format(f, bn, i))
            mean.append(args[3].format(f, bn, i))
            var.append(args[4].format(f, bn, i))

            pw.append('arg:{}_{}_weight'.format(l, pblock[i]))
            pgamma.append('arg:{}_{}_bn_gamma'.format(l, pblock[i]))
            pbeta.append('arg:{}_{}_bn_beta'.format(l, pblock[i]))
            pmean.append('aux:{}_{}_bn_moving_mean'.format(l, pblock[i]))
            pvar.append('aux:{}_{}_bn_moving_var'.format(l, pblock[i]))
        
        bn_n += 1


tw = [w, gamma, beta, mean, var, b]

ptw = [pw, pgamma, pbeta, pmean, pvar, pb]

for _w, _pw in zip(tw, ptw):

    for __w, __pw in zip(_w, _pw):

        assert(nds_zoo[__w].shape == nds_p[__pw].shape)

        #print(nds_target[__pw].shape)

        if scale > 1:

            expand = scale - 1

            if 'beta' in __w or 'gamma' in __w or 'mean' in __w or 'var' in __w or 'output' in __w:

                #nds_target[__pw] = nds_scale[__w]

                continue

            cout, cin, h, w = nds_scale[__pw].shape # here use pw for alreay trained
            #cout, cin, h, w = nds_scale[__w].shape # here use w download from gluon model zoo?

            if cin == 3 or cin == 1:

                exout = int(cout * expand);

                exnd = nds_scale[__pw][:exout]

                ncout = cout * scale;
                
                nds_target[__pw] = nd.concat(*[exnd, nds_scale[__pw]], dim = 0)

            elif cout == 1280 or cout == 1:

                exin = int(cin * expand);

                exnd = nds_scale[__pw][:, :exin]

                nds_target[__pw] = nd.concat(*[exnd, nds_scale[__pw]], dim = 1)

            else:

                exout = int(cout * expand); exin = int(cin * expand)

                exnd_0 = nds_scale[__pw][:exout]

                tmp = nd.concat(*[exnd_0, nds_scale[__pw]], dim = 0)

                exnd_1 = tmp[:,:exin]

                nds_target[__pw] = nd.concat(*[exnd_1, tmp], dim = 1)

        else:

            nds_target[__pw] = nds_scale[__w]

        #print('---------------------------')

        #print('{} : {}'.format(__w, nds_scale[__w].shape))
    
        #print('{} : {}'.format(__pw, nds_target[__pw].shape))

nds_target['arg:fc7_bias'] = nd.zeros([1000, 1])

nd.save('mobilenet_v2_{}-0000.params'.format(scale), nds_target)

