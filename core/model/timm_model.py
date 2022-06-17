import timm
import torch
import torch.nn as nn


def generate_timm_model(args):
    model = TIMM(args)
    
    return model


class TIMM(nn.Module):
    """
        SOTA model usage
        1. resnet18
        2. repvgg_b0
        3. mobilenetv3_large_100
        
    """
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.use_emb = False
        self.emb_only = args.use_emb_only
        
        arch_name = self.args.model
        
        model = timm.create_model(arch_name, pretrained=True)
        
        # help documents - https://fastai.github.io/timmdocs/create_model (how to use feature_extractor in timm)
        if self.args.model == 'swin_large_patch4_window7_224':
            self.feature_module = nn.Sequential(
                *list(model.children())[:-2],
            )
            self.gap = nn.AdaptiveAvgPool1d(1)
        else:
            self.feature_module = nn.Sequential(
                *list(model.children())[:-1]
            )
        
        if self.args.experiment_type == 'theator':
            for p in self.feature_module.parameters():
                p.requires_grad = False
                
        self.classifier = nn.Linear(model.num_features, 2)
        
        if 'hem-emb' in self.args.hem_extract_mode or 'hem-focus' in self.args.hem_extract_mode:
            self.use_emb = True
            self.proxies = nn.Parameter(torch.randn(model.num_features, 2))
        
        else : # off-line and genral
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.args.dropout_prob, inplace=True),
                nn.Linear(model.num_features, 2)
            )
        
        if self.args.use_online_mcd:
            self.dropout = nn.Dropout(self.args.dropout_prob)
        
    def forward(self, x):
        features = self.feature_module(x)
        
        if self.args.model == 'swin_large_patch4_window7_224':
            features = self.gap(features.permute(0, 2, 1))
            
        if self.args.use_online_mcd: 
            if self.training: 
                features = self.dropout(features)
            else:
                mcd_outputs = []
                for _ in range(self.args.n_dropout):
                    mcd_outputs.append(self.dropout(features).unsqueeze(0))
                    
                a = torch.vstack(mcd_outputs)
                features = torch.mean(a, 0)

        if self.emb_only:
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            
            return features
        else:    
            output = self.classifier(features.view(x.size(0), -1))
            
            if self.use_emb and self.training:
                return features, output
            else:
                return output