import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def generate_efficientnet(args):
    model = CustomEfficientNet(args)
    
    return model


class CustomEfficientNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.use_emb = False
    
        arch_name = self.args.model
        
        arch_name = arch_name[:-3] + '-' + arch_name[-2:]

        print('MODEL = EFFICIENTNET-{}'.format(arch_name[-2:]))
        if args.pretrained:
            model = EfficientNet.from_pretrained(arch_name, advprop=False, num_classes=2)
        else:
            model = EfficientNet.from_name(arch_name, num_classes=2)
        
        ml = list(model.children())
        self.model = model
        
        if 'hem-emb' in self.args.train_method:
            self.use_emb = True
            self.proxies = nn.Parameter(torch.randn(ml[-2].in_features, 2))
        
        
    def forward(self, x):
        features = self.model.extract_features(x)
        gap = self.model._avg_pooling(features).view(x.size(0), -1)
        if self.model._global_params.include_top:
            output = self.model._dropout(gap)
            output = self.model._fc(output)
        else:
            output = gap
        
        if self.use_emb and self.training:
            return gap, output
        else:
            return output