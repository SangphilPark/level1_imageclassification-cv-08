import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import *

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

    
class VGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = vgg19_bn()

        #self.backbone.features.requires_grad_(requires_grad=False) # feature 부분얼리기

        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088,4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes,bias=True)
        )
        

    def forward(self, x):
        x = self.backbone(x)
        return x

class vit32(nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        self.backbone = timm.models.vit_base_patch16_224(pretrained=True)

        self.backbone.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, x):

        x = self.backbone(x)

        return x
    

""" class VGG19LN(nn.Module):
    def __init__(self, num_classes):
        super(VGG19LN, self).__init__()

        # Pre-trained VGG19bn model
        self.vgg19ln = vgg19_bn(pretrained=True)

        # Replace batch normalization layers with layer normalization layers
        for name, module in self.vgg19ln.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                new_module = nn.LayerNorm(module.num_features)
                new_module.weight.data = module.weight.data.clone().detach().view(-1)
                new_module.bias.data = module.bias.data.clone().detach().view(-1)
                self._replace_module(name, new_module, self.vgg19ln)

        # Freeze all layers
        for param in self.vgg19ln.parameters():
            param.requires_grad = False
    
        self.vgg19ln.classifier = nn.Sequential(
            nn.Linear(51277,4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes,bias=True)
        )

    def forward(self, x):
        x = x.unsqueeze(0)  # [64, 64, 224, 224] -> [1, 64, 224, 224]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        x = self.vgg19ln.features(x)
        x = self.vgg19ln.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(x.size(0), x.size(1), 1, 1)  # shape을 (batch_size, num_features, 1, 1)로 변경
        x = self.vgg19ln.classifier(x)
        return x

    def _replace_module(self, name, new_module, parent_module):
        #Replaces a module with a new one.
        parent_name, base_name = name.rsplit('.', 1)
        parent_module = self._get_module(parent_name, parent_module)
        parent_module._modules[base_name] = new_module

    def _get_module(self, name, parent_module):
        #Gets a module from its name.
        if '.' in name:
            parent_name, base_name = name.rsplit('.', 1)
            parent_module = self._get_module(parent_name, parent_module)
            return parent_module._modules[base_name]
        else:
            return parent_module._modules[name]

"""
 
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = vgg16_bn()

        #self.backbone.features.requires_grad_(requires_grad=False) # feature 부분얼리기

        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088,4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes,bias=True)
        )
        

    def forward(self, x):
        x = self.backbone(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088,4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes,bias=True)   
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet101(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048,1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,512,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,num_classes,bias=True) 
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = densenet121(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1024,512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,256,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256,num_classes,bias=True)   
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return x

class DenseNet201(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = densenet201(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1920,1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,256,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256,num_classes,bias=True)   
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return x

class EfficientNet_b1(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = timm.models.efficientnet_b1(pretrained=True)
        self.backbone.classifier = nn.Linear(1280, num_classes, bias=True)
    def forward(self, x):
        x = self.backbone(x)
        return x

#pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet 
class EfficientNet_b1_2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientNet = EfficientNet.from_pretrained('efficientnet-b1')
        num_features = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Linear(num_features, num_classes)
    def forward(self, x):
        x = self.efficientNet(x)
        return x
    
class Inception_ResNet_v2(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = timm.models.inception_resnet_v2(pretrained=True)
        self.backbone.classifier = nn.Linear(1536, num_classes, bias=True)
    def forward(self, x):
        x = self.backbone(x)
        return x

    
class ViT_tiny_p16_224(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = timm.models.vit_tiny_patch16_224(pretrained=True)
        self.backbone.head = nn.Linear(in_features=192, out_features=num_classes, bias=True)
    def forward(self, x):
        x = self.backbone(x)
        return x


class ViT_small_p16_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.models.vit_small_patch16_384(pretrained=True)
        self.backbone.head = nn.Linear(in_features=384, out_features=num_classes, bias=True)
    def forward(self, x):
        x = self.backbone(x)
        return x

    
class SwinTransformer_tiny_p4_224(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = timm.models.swin_tiny_patch4_window7_224(pretrained=True)
        self.backbone.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    def forward(self, x):
        x = self.backbone(x)
        return x