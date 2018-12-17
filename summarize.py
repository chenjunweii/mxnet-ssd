import mxnet as mx


net = mx.symbol.load('symbol/se-resnext50-32x4d-symbol.json')

print(net.list_outputs())
