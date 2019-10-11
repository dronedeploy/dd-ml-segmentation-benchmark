from keras import layers, models
import numpy as np
import tensorflow as tf

def build_unet(size=300, basef=64, maxf=512, encoder='resnet50', pretrained=True):
    input = layers.Input((size, size, 3))

    encoder_model = make_encoder(input, name=encoder, pretrained=pretrained)

    crosses = []

    for layer in encoder_model.layers:
        # don't end on padding layers
        if type(layer) == layers.ZeroPadding2D:
            continue
        idx = get_scale_index(size, layer.output_shape[1])
        if idx is None:
            continue
        if idx >= len(crosses):
            crosses.append(layer)
        else:
            crosses[idx] = layer

    x = crosses[-1].output
    for scale in range(len(crosses)-2, -1, -1):
        nf = min(basef * 2**scale, maxf)
        x = upscale(x, nf)
        x = act(x)
        x = layers.Concatenate()([
            pad_to_scale(x, scale, size=size),
            pad_to_scale(crosses[scale].output, scale, size=size)
        ])
        x = conv(x, nf)
        x = act(x)

    x = conv(x, 6)
    x = layers.Activation('softmax')(x)

    return models.Model(input, x)

def make_encoder(input, name='resnet50', pretrained=True):
    if name == 'resnet18':
        from classification_models.keras import Classifiers
        ResNet18, _ = Classifiers.get('resnet18')
        model = ResNet18(
            weights='imagenet' if pretrained else None,
            input_tensor=input,
            include_top=False
        )
    elif name == 'resnet50':
        from keras.applications.resnet import ResNet50
        model = ResNet50(
            weights='imagenet' if pretrained else None,
            input_tensor=input,
            include_top=False
        )
    elif name == 'resnet101':
        from keras.applications.resnet import ResNet101
        model = ResNet101(
            weights='imagenet' if pretrained else None,
            input_tensor=input,
            include_top=False
        )
    elif name == 'resnet152':
        from keras.applications.resnet import ResNet152
        model = ResNet152(
            weights='imagenet' if pretrained else None,
            input_tensor=input,
            include_top=False
        )
    elif name == 'vgg16':
        from keras.applications.vgg16 import VGG16
        model = VGG16(
            weights='imagenet' if pretrained else None,
            input_tensor=input,
            include_top=False
        )
    elif name == 'vgg19':
        from keras.applications.vgg19 import VGG19
        model = VGG19(
            weights='imagenet' if pretrained else None,
            input_tensor=input,
            include_top=False
        )
    else:
        raise Exception(f'unknown encoder {name}')

    return model

def get_scale_index(in_size, l_size):
    for i in range(8):
        s_size = in_size // (2 ** i)
        if abs(l_size - s_size) <= 4:
            return i
    return None

def pad_to_scale(x, scale, size=300):
    expected = int(np.ceil(size / (2. ** scale)))
    diff = expected - int(x.shape[1])
    if diff > 0:
        left = diff // 2
        right = diff - left
        x = reflectpad(x, (left, right))
    elif diff < 0:
        left = -diff // 2
        right = -diff - left
        x = layers.Cropping2D(((left, right), (left, right)))(x)
    return x

def reflectpad(x, pad):
    return layers.Lambda(lambda x: tf.pad(x, [(0, 0), pad, pad, (0, 0)], 'REFLECT'))(x)

def upscale(x, nf):
    x = layers.UpSampling2D((2, 2))(x)
    x = conv(x, nf, kernel_size=(1, 1))
    return x

def act(x):
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def conv(x, nf, kernel_size=(3, 3), **kwargs):
    padleft = (kernel_size[0] - 1) // 2
    padright = kernel_size[0] - 1 - padleft
    if padleft > 0 or padright > 0:
        x = reflectpad(x, (padleft, padright))
    return layers.Conv2D(nf, kernel_size=kernel_size, padding='valid', **kwargs)(x)
