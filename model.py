import torch.nn as nn
import torch.nn.functional as F
import timm


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


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. ?œ„??? ê°™ì´ ?ƒ?„±??˜ parameter ?— num_claases ë¥? ?¬?•¨?•´ì£¼ì„¸?š”.
        2. ?‚˜ë§Œì˜ ëª¨ë¸ ?•„?‚¤?…ì³ë?? ?””??¸ ?•´ë´…ë‹ˆ?‹¤.
        3. ëª¨ë¸?˜ output_dimension ??? num_classes ë¡? ?„¤? •?•´ì£¼ì„¸?š”.
        """

    def forward(self, x):
        """
        1. ?œ„?—?„œ ? •?˜?•œ ëª¨ë¸ ?•„?‚¤?…ì³ë?? forward propagation ?„ ì§„í–‰?•´ì£¼ì„¸?š”
        2. ê²°ê³¼ë¡? ?‚˜?˜¨ output ?„ return ?•´ì£¼ì„¸?š”
        """
        return x


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. ?œ„??? ê°™ì´ ?ƒ?„±??˜ parameter ?— num_claases ë¥? ?¬?•¨?•´ì£¼ì„¸?š”.
        2. ?‚˜ë§Œì˜ ëª¨ë¸ ?•„?‚¤?…ì³ë?? ?””??¸ ?•´ë´…ë‹ˆ?‹¤.
        3. ëª¨ë¸?˜ output_dimension ??? num_classes ë¡? ?„¤? •?•´ì£¼ì„¸?š”.
        """
        self.backbone = vgg19_bn()

        # self.backbone.features.requires_grad_(requires_grad=False) # feature ë¶?ë¶„ì–¼ë¦¬ê¸°

        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes, bias=True)
        )

    def forward(self, x):
        """
        1. ?œ„?—?„œ ? •?˜?•œ ëª¨ë¸ ?•„?‚¤?…ì³ë?? forward propagation ?„ ì§„í–‰?•´ì£¼ì„¸?š”
        2. ê²°ê³¼ë¡? ?‚˜?˜¨ output ?„ return ?•´ì£¼ì„¸?š”
        """
        x = self.backbone(x)
        return x


class vit32(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = timm.models.vit_base_patch16_224(pretrained=True)

        self.backbone.head = nn.Linear(
            in_features=768, out_features=num_classes, bias=True)

    def forward(self, x):

        x = self.backbone(x)

        return x
