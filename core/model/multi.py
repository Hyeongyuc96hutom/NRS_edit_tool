import os
import torch
import torch.nn as nn
from glob import glob
import natsort
import timm
from core.accessory.RepVGG.repvgg import RepVGG


def generate_multi_model(args):
    model = MultiModel(args)
    
    return model


class MultiModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.use_emb = False
        self.use_avg_feature = True
        
        target_features = 512
        
        # each model part
        # MobileNet - 1280
        model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        self.mobilenet_feature = nn.Sequential(*list(model.children())[:-1])
        
        # ResNet18 - 512
        model = timm.create_model('resnet18', pretrained=True)
        self.resnet_feature = nn.Sequential(*list(model.children())[:-1])
        
        # RepVGG - 1280
        model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2,
                        width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)
        
        self.repvgg_feature = nn.Sequential(*list(model.children())[:-1])
        
        # mobilenet, resnet, repvgg
        self.feature_to_emb = nn.ModuleList([
            nn.Linear(1280, target_features),
            nn.Linear(512, target_features),
            nn.Linear(1280, target_features),
        ])
        
        if 'hem-emb' in self.args.hem_extract_mode:
            self.use_emb = True
            self.proxies = nn.Parameter(torch.randn(target_features, 2))
            
        self.classifier = nn.Linear(target_features, 2, bias=True)
        
    def change_deploy_mode(self):
        model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2,
                        width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=True)
        
        self.repvgg_feature = nn.Sequential(*list(model.children())[:-1])
        
        ckpoint_path = os.path.join(self.args.restore_path, 'checkpoints', '*.pt')
        ckpts = glob(ckpoint_path)
        ckpts = natsort.natsorted(ckpts)
        state = torch.load(ckpts[-1])
        self.repvgg_feature.load_state_dict(state)
        self.repvgg_feature = self.repvgg_feature.cuda()
    
        
    def forward(self, x):
        feat_mobile = self.mobilenet_feature(x).view(x.size(0), -1)
        feat_resnet = self.resnet_feature(x).view(x.size(0), -1)
        feat_repvgg = self.repvgg_feature(x).view(x.size(0), -1)
        
        features = [feat_mobile, feat_resnet, feat_repvgg]
        
        features = [self.feature_to_emb[idx](features[idx]) for idx in range(len(features))]

        if self.use_avg_feature:
            avg_features = torch.mean(torch.stack(features, -1), -1)
            output = self.classifier(avg_features)
            
            if self.use_emb and self.training:
                return avg_features, output
            else:
                return output
    
    def forward_multi(self, x):
        feat_mobile = self.mobilenet_feature(x).view(x.size(0), -1)
        feat_resnet = self.resnet_feature(x).view(x.size(0), -1)
        feat_repvgg = self.repvgg_feature(x).view(x.size(0), -1)
        
        features = [feat_mobile, feat_resnet, feat_repvgg]
        
        features = [self.feature_to_emb[idx](features[idx]) for idx in range(len(features))]
        
        outputs = [self.classifier(features[idx]) for idx in range(len(features))]
            
        return outputs