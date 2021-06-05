from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

import requests
from tqdm import tqdm


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_model, num_classes):
        super().__init__()

        self.fc1_bbox = nn.Linear(dim_model, dim_model)
        self.fc2_bbox = nn.Linear(dim_model, dim_model)
        self.fc3_bbox = nn.Linear(dim_model, 2)

        self.fc_logits = nn.Linear(dim_model, num_classes + 1)

    def forward(self, x):
        bbox_out = self.fc1_bbox(x)
        bbox_out = F.relu(bbox_out)

        bbox_out = self.fc2_bbox(bbox_out)
        bbox_out = F.relu(bbox_out)

        bbox_out = self.fc3_bbox(bbox_out)
        bbox_out = torch.sigmoid(bbox_out)

        logits_out = self.fc_logits(x)

        return bbox_out, logits_out


class SimpleFeedForwardNetwork(nn.Module):
    def __init__(self, dim_model, num_classes):
        super().__init__()

        self.linear_class = nn.Linear(dim_model, num_classes + 1)
        self.linear_bbox = nn.Linear(dim_model, 2)

    def forward(self, x):
        logits_out = self.linear_class(x)
        bbox_out = self.linear_bbox(x).sigmoid()
        return bbox_out, logits_out


class DETR(nn.Module):
    def __init__(self,
                 num_classes,
                 dim_model=256,
                 n_heads=8,
                 n_encoder_layers=6,
                 n_decoder_layers=6,
                 n_queries=100,
                 head_type='complex'):
        super().__init__()

        # initialize resnet and remove the last two layers (avgpool and fc)
        # reduction in spatial dimensionality is Wo, Ho -> Wo/32, Ho/32
        # output channels are 2048
        self.backbone = resnet50(pretrained=False)
        self.backbone = nn.Sequential(OrderedDict(
            (name, child) for (name, child) in self.backbone.named_children()
            if name not in ['fc', 'avgpool']
        ))

        # use the running stats of the backbone's batchnorm layers
        self.backbone.eval()

        self.conv = nn.Conv2d(2048, dim_model, kernel_size=1)
        self.transformer = nn.Transformer(dim_model, n_heads, n_encoder_layers, n_decoder_layers)

        self.object_queries = nn.Parameter(torch.rand(n_queries, dim_model))

        # positional embeddings are learnt, the size 50 is due to the assumption that
        # no feature map will have more than 50 rows/cols, considering the backbone
        # has a reduction of 32, no original image should be larger than 1600 px
        self.row_pos_embed = nn.Parameter(torch.rand(50, dim_model // 2))
        self.col_pos_embed = nn.Parameter(torch.rand(50, dim_model // 2))

        if head_type == 'simple':
            self.ffn = SimpleFeedForwardNetwork(dim_model, num_classes)
        else:
            self.ffn = FeedForwardNetwork(dim_model, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # [batch, 2048, h, w]
        features = self.conv(features)  # [batch, dim_model, h, w]
        batch_size, _, height, width = features.shape

        row_pos_embed = self.row_pos_embed[:height]
        row_pos_embed = row_pos_embed.unsqueeze(1).repeat(1, width, 1)  # [h, w, dim_model//2]

        col_pos_embed = self.col_pos_embed[:width]
        col_pos_embed = col_pos_embed.unsqueeze(0).repeat(height, 1, 1)  # [h, w, dim_model//2]

        pos_embed = torch.cat([col_pos_embed, row_pos_embed], dim=-1)  # [h, w, dim_model]
        pos_embed = pos_embed.flatten(0, 1).unsqueeze(1)  # [h*w, 1, dim_model]

        features = features.flatten(2)  # [batch, dim_model, h*w]
        features = 0.1 * features.permute(2, 0, 1) + pos_embed  # [h*w, batch, dim_model]

        object_queries = self.object_queries.unsqueeze(1).repeat(1, batch_size, 1)
        output_embedding = self.transformer(features, object_queries)  # [n_queries, batch, dim_model]

        pred_bbox, pred_logits = self.ffn(output_embedding.permute(1, 0, 2))

        preds = {
            'bboxes': pred_bbox,
            'logits': pred_logits
        }

        return preds

    def load_demo_state_dict(self, path_to_dict):
        url = 'https://drive.google.com/uc?export=download&id=109o6As_ocO8Q1_J6eDqMxQ3KNlleUUVE'

        path_to_dict = Path(path_to_dict)
        if not path_to_dict.exists():

            raise ValueError("No DETR pretrained weights found. Currently the pretrained " 
                             f"weights can't be automatically downloaded, use this link: {url} "
                             f"and save the file as: {str(path_to_dict)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(path_to_dict, map_location=device)['state_dict']
        self.to(device)

        # TODO: try except when loading dict, some weights are different size
        # temporarily deleting commenting out the ffn weights
        name_changes = {
            'query_pos': 'object_queries',
            'row_embed': 'row_pos_embed',
            'col_embed': 'col_pos_embed',
            # 'linear_class': 'ffn.linear_class',
            # 'linear_bbox': 'ffn.linear_bbox',
        }

        # explicit conversion to list is necessary because keys change during iteration
        for name in list(state_dict.keys()):
            if '.' in name:
                name_prefix = name.split('.')[0]
            else:
                name_prefix = name

            if name_prefix in name_changes:
                new_name = name.replace(name_prefix, name_changes[name_prefix])
                state_dict[new_name] = state_dict[name]
                state_dict.pop(name)

        self.load_state_dict(state_dict, strict=True)

