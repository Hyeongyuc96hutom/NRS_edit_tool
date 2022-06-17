import torch
import torch.nn as nn
import torchvision.models as models


def generate_resnet(args):
    model = ResNet(args)
    
    return model


class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.use_emb = False
    
        arch_name = self.args.model
        
        if arch_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            
        self.feature_module = nn.Sequential(
            *list(model.children())[:-1]
        )
        
        if self.args.experiment_type == 'theator':
            for p in self.feature_module.parameters():
                p.requires_grad = False
        
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, 2)
        
        if 'hem-emb' in self.args.train_method:
            self.use_emb = True
            self.proxies = nn.Parameter(torch.randn(512, 2))
            # self.proxies = nn.Parameter(torch.randn(512, 4))
        
    
    def forward(self, x):
        features = self.feature_module(x)
        gap = self.gap(features).view(x.size(0), -1)
        output = self.fc(gap)
        
        if self.use_emb and self.training:
            return gap, output
        else:
            return output
    
        
        