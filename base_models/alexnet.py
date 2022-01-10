import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, vocab_size, mode):
        super(AlexNet, self).__init__()
        self.mode = mode
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc_cls = nn.Sequential(
                nn.Linear(256, vocab_size),
                nn.Sigmoid()
            )
        self.pooling = nn.AdaptiveAvgPool2d(1)


    def get_features(self, x):
        x = self.features(x)

        return x    

    def mul_classification(self, x):
        mul_class = self.pooling(x).view(x.size(0), -1)
        mul_class = self.fc_cls(mul_class)   
        
        return mul_class

    def forward(self, x):
        x = self.get_features(x)      
        mul_class = self.mul_classification(x)
        
        return mul_class



def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root), strict=False)
        
    return model
