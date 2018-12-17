import mxnet as mx
from mxnet import gluon

x = mx.symbol.Variable('x')

nds = dict()

nds['x'] = mx.nd.array([1, 0, 0])

nds['y'] = mx.nd.array([1])

L = gluon.loss.SoftmaxCrossEntropyLoss()

out = mx.sym.softmax(x, name = 'softmax')  

exe = out.bind(args = nds, ctx = mx.cpu())

exe.forward()

print(exe.outputs[0].asnumpy())

p = exe.outputs[0]

print(L(nds['x'], nds['y']))

print(L(p, nds['y']))
#L = gluon.loss.SoftmaxCrossEntropyLoss()

