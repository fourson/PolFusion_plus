from torchvision import models

from .l12 import l2

VGG19_FEATURES = models.vgg19(weights='DEFAULT').features
CONV3_3_IN_VGG_19 = VGG19_FEATURES[0:15].cuda()


def perceptual(pred, gt):
    model = CONV3_3_IN_VGG_19  # change it to whatever you want
    pred_feature_map = model(pred)
    gt_feature_map = model(gt).detach()
    return l2(pred_feature_map, gt_feature_map)
