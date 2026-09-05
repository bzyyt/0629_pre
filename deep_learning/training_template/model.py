from config import cfg
from torch import nn
from torchvision.models import ResNet, ResNet18_Weights, resnet18


# 模型构建
# 空模型
def build_scratch_model() -> ResNet:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, cfg.NUM_CLASSES)
    return model


# 预训练模型，冻结卷积层，只训练全连接层
def build_feature_extractor_model() -> ResNet:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, cfg.NUM_CLASSES)
    return model


# 解冻最后一层卷积层，训练全连接层和最后一层卷积层
def unfreeze_layer4(model: ResNet) -> ResNet:
    for param in model.layer4.parameters():
        param.requires_grad = True
    return model
