import torch
import torch.nn as nn
import torchvision.models as models


def generate_mobilenet(args):
    model = MobileNet(args)

    return model



class MobileNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.use_emb = False
        
        arch_name = self.args.model
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        
        if arch_name == 'mobilenet_v2':
            print('MODEL = MOBILENET_V2')
            model = models.mobilenet_v2(pretrained=args.pretrained)
            num_ftrs = model.classifier[-1].in_features
            
            ml = list(model.children())
            self.feature_module = ml[0]
            proxy_feat = num_ftrs
            
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, 2),
            )
        
        elif arch_name == 'mobilenet_v3_small' :
            print('MODEL = MOBILENET_V3_SMALL')
            model = models.mobilenet_v3_small(pretrained=args.pretrained) # model scretch learning
            num_ftrs = model.classifier[-1].in_features
            
            ml = list(model.children())
            self.feature_module = ml[0]
            proxy_feat = 576

            self.classifier = nn.Sequential(
                nn.Linear(576, num_ftrs), #lastconv_output_channels, last_channel
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_ftrs, 2) #last_channel, num_classes
            )
        elif arch_name == 'mobilenet_v3_large' :
            print('MODEL = MOBILENET_V3_LARGE')
            model = models.mobilenet_v3_large(pretrained=args.pretrained)
            num_ftrs = model.classifier[-1].in_features
            
            ml = list(model.children())
            self.feature_module = ml[0]
            proxy_feat = 960

            self.classifier = nn.Sequential(
                nn.Linear(960, num_ftrs), #lastconv_output_channels, last_channel
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_ftrs, 2) #last_channel, num_classes
            )
            
        if 'hem-emb' in self.args.hem_extract_mode:
            self.use_emb = True
            self.proxies = nn.Parameter(torch.randn(proxy_feat, 2))
            
    def forward(self, x):
        features = self.feature_module(x)
        gap = self.gap(features).view(x.size(0), -1)
        output = self.classifier(gap)
        
        if self.use_emb and self.training:
            return gap, output
        else:
            return output