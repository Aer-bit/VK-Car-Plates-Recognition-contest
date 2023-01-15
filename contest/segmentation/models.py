import segmentation_models_pytorch as smp


def get_model():
    # TODO TIP: There's a lot of tasty things to try here.
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model


def get_model1():
    # TODO TIP: There's a lot of tasty things to try here.
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model


def get_model2():
    # TODO TIP: There's a lot of tasty things to try here.
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model


def get_model3():
    # TODO TIP: There's a lot of tasty things to try here.
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model


def get_model4():
    # TODO TIP: There's a lot of tasty things to try here.
    model = smp.Unet(
        encoder_name="densenet169",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model


def get_model5():
    # TODO TIP: There's a lot of tasty things to try here.
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model


def get_model6():
    # TODO TIP: There's a lot of tasty things to try here.
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model


def get_model7():
    # TODO TIP: There's a lot of tasty things to try here.
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model