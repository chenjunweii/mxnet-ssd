from mxnet import nd
import mxnet as mx

_, oargs, oaux = mx.model.load_checkpoint('mobilenet_v2', 0)

_, targs, taux = mx.model.load_checkpoint('mobilenet_v2_025', 0)

print(len(oargs))

print(len(targs))

for _o in oargs:

    if _o not in targs.keys():

        print(_o)

for _o in oaux:

    if _o not in taux.keys():

        print(_o)



