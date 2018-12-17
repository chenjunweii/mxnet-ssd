import mxnet as mx

# Set dummy dimensions

net = mx.symbol.load('symbol/se-resnext50-32x4d-symbol.json')
# Visualize your network
mx.viz.plot_network(net)
