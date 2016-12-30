from lasagne.regularization import regularize_network_params, l2
from lasagne.objectives import categorical_crossentropy
from lasagne.layers import (InputLayer, DropoutLayer,
    Conv2DLayer, MaxPool2DLayer, GlobalPoolLayer,
    DenseLayer, ConcatLayer, get_output)
from lasagne.nonlinearities import softmax
from collections import OrderedDict
from theano import tensor as T
from PIL import Image
import numpy as np
import lasagne
import theano
import os


def make_net(input_var, input_var2=None, n_classes=99, base_conv_feat=16):
    net = OrderedDict()
    net['input'] = InputLayer((1, 1, None, None), input_var)
    net['drop_in'] = DropoutLayer(net['input'], 0.2)

    net['conv1_1'] = Conv2DLayer(net['drop_in'], int(base_conv_feat), (3, 3))
    net['pool1'] = MaxPool2DLayer(net['conv1_1'], 2)
    net['conv1_2'] = Conv2DLayer(net['pool1'], int(base_conv_feat / 2), (1, 1))
    net['drop1'] = DropoutLayer(net['conv1_2'], 0.5)

    net['conv2_1'] = Conv2DLayer(net['drop1'], int(base_conv_feat * 2), (3, 3))
    net['pool2'] = MaxPool2DLayer(net['conv2_1'], 2)
    net['conv2_2'] = Conv2DLayer(net['pool2'], int(base_conv_feat), (1, 1))
    net['drop2'] = DropoutLayer(net['conv2_2'], 0.5)

    net['conv3_1'] = Conv2DLayer(net['drop2'], int(base_conv_feat * 4), (3, 3))
    net['pool3'] = MaxPool2DLayer(net['conv3_1'], 2)
    net['conv3_2'] = Conv2DLayer(net['pool3'], int(base_conv_feat * 2), (1, 1))
    net['drop3'] = DropoutLayer(net['conv3_2'], 0.5)

    net['global_pool'] = GlobalPoolLayer(net['drop3'])

    if input_var2 is None:
        net['pre_prob'] = net['global_pool']
    else:
        net['concat'] = ConcatLayer(
            [
                DropoutLayer(InputLayer((None, 192), input_var2), 0.2),
                net['global_pool']
            ]
        )
        net['fc1'] = DenseLayer(net['concat'], base_conv_feat * 4)
        net['fc1_drop'] = DropoutLayer(net['fc1'], 0.5)
        net['pre_prob'] = net['fc1_drop']

    net['prob'] = DenseLayer(net['pre_prob'], n_classes, nonlinearity=softmax)

    return net


def main(args):

    data_dir = os.path.abspath('./data')
    image_dir = os.path.join(data_dir, 'images')

    with open(os.path.join(data_dir, 'train.csv')) as fd:
        data = fd.read().splitlines()[1:]
    labels = [line.split(',')[1] for line in data]
    ids = [line.split(',')[0] for line in data]
    feats = [
        np.array([list(map(float, line.split(',')[2:]))])
        for line in data
    ]
    label_names = sorted(set(labels))
    label_lookup = {v:k for k,v in list(enumerate(label_names))}

    y = [label_lookup[name] for name in labels]

    input_var = T.tensor4()
    input_var2 = T.matrix()
    target_var = T.vector('gt', dtype='int64')
    net = make_net(input_var, input_var2)

    loss = categorical_crossentropy(
        get_output(net['prob'],
            dict(input_var=input_var, input_var2=input_var2)
        ),
        target_var
    ).mean() + 5e-3 * regularize_network_params(
        net['prob'],
        lasagne.regularization.l2
    )
    params = lasagne.layers.get_all_params(net['prob'], trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params,
        learning_rate=0.005,
        momentum=0.95
    )

    train_fn = theano.function([input_var, input_var2, target_var], loss, updates=updates)


    from itertools import cycle
    entries = cycle(zip(ids, y, feats))

    loss_accum = 0.
    epoch_loss = 0.
    report_iter = 99
    epochs = 1
    for idx, entry in enumerate(entries, 1):
        this_id, label, feats = entry
        image_name = os.path.join(image_dir, this_id + '.jpg')
        image = (np.array(Image.open(image_name)) / 255)[np.newaxis, np.newaxis, ::6, ::6]
        loss_val = train_fn(image, feats, np.array([label]))
        loss_accum += loss_val / report_iter
        epoch_loss += loss_val / len(ids)

        #if idx % report_iter == 0:
        #    print(loss_accum)
        #    loss_accum = 0.

        if idx % ((epochs)*len(ids)) == 0:
            print('Epoch {}, loss: {}'.format(epochs, epoch_loss))
            epoch_loss = 0.
            epochs += 1

            if epochs >= 10:
                print('Breaking')
                break

if __name__ == '__main__':
    main(None)
