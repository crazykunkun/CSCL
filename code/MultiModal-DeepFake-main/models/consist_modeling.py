import torch
import torch.nn as nn
import pdb
from .interaction import Self_Interaction
from timm.models.layers import trunc_normal_

def calculate_patch_labels(images, boxes, fake_text_pos, num_patches=(16, 16)):
    # 获取图片的尺寸
    _, height, width = images.shape[1:4]
    
    # 计算每个 patch 的大小
    patch_height = height // num_patches[0]
    patch_width = width // num_patches[1]

    # 将 boxes 转换为张量
    # boxes = torch.tensor(boxes)  # shape: [N, 4]

    # 计算框的坐标
    box_x1 = (boxes[:, 0] * width).int()
    box_y1 = (boxes[:, 1] * height).int()
    box_w = (boxes[:, 2] * width).int()
    box_h = (boxes[:, 3] * height).int()
    
    # box_x2 = box_x1 + box_w
    # box_y2 = box_y1 + box_h

    box_x2 = box_x1 + 0.5*box_w
    box_y2 = box_y1 + 0.5*box_h

    box_x1 = box_x1 - 0.5*box_w
    box_y1 = box_y1 - 0.5*box_h
    
    # 计算 patch 的坐标
    patch_x1 = torch.arange(0, width, patch_width).view(1, -1).expand(boxes.size(0), -1).to(boxes.device)
    patch_y1 = torch.arange(0, height, patch_height).view(1, -1).expand(boxes.size(0), -1).to(boxes.device)
    patch_x2 = patch_x1 + patch_width
    patch_y2 = patch_y1 + patch_height

    # 计算每个 patch 的面积
    patch_area = patch_width * patch_height

    # 计算相交区域
    inter_x1 = torch.max(patch_x1, box_x1.view(-1, 1))
    inter_y1 = torch.max(patch_y1, box_y1.view(-1, 1))
    inter_x2 = torch.min(patch_x2, box_x2.view(-1, 1))
    inter_y2 = torch.min(patch_y2, box_y2.view(-1, 1))

    # 计算相交区域的面积

    zero = torch.tensor(0, device=boxes.device)
    inter_area = torch.max(zero, inter_x2 - inter_x1).unsqueeze(1) * torch.max(zero, inter_y2 - inter_y1).unsqueeze(2)

    # 判断条件：相交面积是否大于 patch 面积的一半
    labels = (inter_area > (patch_area / 2)).int()

    labels_extented = labels.view(images.shape[0], -1, 1)

    consistency_matrix = (labels_extented == labels_extented.transpose(2, 1)).int()
    
    labels_extented_it = labels.view(images.shape[0], 1, -1)
    fake_text_pos_extented = fake_text_pos.view(images.shape[0], -1, 1)

    consistency_matrix_it = ((labels_extented_it + fake_text_pos_extented)<1).int()

    return consistency_matrix, consistency_matrix_it, labels.view(images.shape[0], -1)

def _expand_patch_labels(labels, target_edge):
    current_edge = labels.shape[1]
    if current_edge == target_edge:
        return labels
    scale = target_edge // current_edge
    return labels.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)


def _build_consistency_from_labels(labels):
    flat_labels = labels.view(labels.shape[0], -1).float()
    consistency_matrix = (flat_labels.unsqueeze(-1) == flat_labels.unsqueeze(1)).float()
    return consistency_matrix, flat_labels


def get_sscore_label(img, fake_img_box, fake_text_pos, len_edge=16, multi_scales=(16, 8)):
    consistency_matrices = []
    expanded_labels = []
    consistency_matrix_it = None

    for scale in multi_scales:
        scale_consistency_matrix, scale_consistency_matrix_it, scale_labels = calculate_patch_labels(
            img,
            fake_img_box,
            fake_text_pos,
            (scale, scale),
        )
        scale_labels = scale_labels.view(img.shape[0], scale, scale)
        expanded_scale_labels = _expand_patch_labels(scale_labels, len_edge)
        expanded_consistency_matrix, expanded_flat_labels = _build_consistency_from_labels(expanded_scale_labels)
        consistency_matrices.append(expanded_consistency_matrix)
        expanded_labels.append(expanded_flat_labels)
        if scale == len_edge:
            consistency_matrix_it = scale_consistency_matrix_it

    consistency_matrix = torch.stack(consistency_matrices, dim=0).mean(dim=0)
    labels = torch.stack(expanded_labels, dim=0).mean(dim=0)
    patch_score = consistency_matrix.sum(dim=-1) / (len_edge * len_edge)
    img_score = patch_score.sum(dim=-1) / (len_edge * len_edge)

    return consistency_matrix, labels, patch_score, img_score, consistency_matrix_it

def get_sscore_label_text(fake_text_pos):

    fake_text_pos_extend = fake_text_pos.unsqueeze(-1)
    sim_matrix = ((fake_text_pos_extend == fake_text_pos_extend.transpose(2,1))).int()
    matrix_mask = ((fake_text_pos_extend + fake_text_pos_extend.transpose(2,1))>=0)
    for i in range(fake_text_pos.shape[0]):
        sim_matrix[i].fill_diagonal_(1)
    return sim_matrix, matrix_mask

class Intra_Modal_Modeling(nn.Module):
    
    def __init__(self, num_head, hidden_dim, input_dim, output_dim, tok_num):
        super().__init__()

        self.correlation_model = Self_Interaction(num_head, hidden_dim, input_dim, output_dim, layers=3)
        self.consist_encoder = nn.Sequential(nn.Linear(output_dim, 256),
                                                  nn.LayerNorm(256),
                                                  nn.GELU(),
                                                  nn.Linear(256, 128),
                                                  nn.LayerNorm(128),
                                                  nn.GELU(),
                                                  nn.Linear(128, 64))
        self.token_number = tok_num
        self.aggregator = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.aggregator_2 = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp_2 = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.num_head = 4
        self.fixed_token_number = tok_num

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def build_fixed_attn_mask(self, scores, largest, valid_count):
        batch_size, num_tokens, _ = scores.shape
        sorted_indices = torch.argsort(scores, dim=-1, descending=largest)
        rank_index = torch.arange(num_tokens, device=scores.device).view(1, 1, num_tokens)
        fixed_k = torch.clamp(
            torch.minimum(
                torch.full_like(valid_count.long(), self.fixed_token_number),
                valid_count.long(),
            ),
            min=1,
        )

        keep_mask = rank_index < fixed_k.unsqueeze(-1)
        attn_mask = torch.ones(batch_size, num_tokens, num_tokens, dtype=torch.bool, device=scores.device)
        attn_mask.scatter_(-1, sorted_indices, ~keep_mask)
        return attn_mask.repeat(self.num_head, 1, 1)
    
    def forward(self, feats, mask, pos_emb, matrix_mask=None):
        
        B, N, C = feats.shape
        feats = self.correlation_model(feats, mask, pos_emb)
        consist_feats = self.consist_encoder(feats)

        norms = torch.norm(consist_feats, p=2, dim=2, keepdim=True)
        normalized_vectors = consist_feats / norms
        similarity_matrix = torch.bmm(normalized_vectors, normalized_vectors.transpose(1, 2))
        similarity_matrix = torch.clamp((similarity_matrix+1)/2, 0, 1)

        if mask.sum() > 0: # for text inputs
            valid_count = (~mask).sum(dim=-1, keepdim=True) - 1
            valid_count = torch.clamp(valid_count, min=1).expand(-1, N)
            similarity_matrix_unsim = similarity_matrix.clone()
            similarity_matrix_unsim[~matrix_mask] = 2

            similarity_matrix_sim = similarity_matrix.clone()
            similarity_matrix_sim[~matrix_mask] = -1
            diagonal_mask = torch.eye(N, device=feats.device).unsqueeze(0).expand(B, N, N)
            similarity_matrix_sim = similarity_matrix_sim - diagonal_mask

        else: # for image inputs
            valid_count = torch.full((B, N), N - 1, device=feats.device, dtype=torch.long)
            similarity_matrix_unsim = similarity_matrix.clone()
            similarity_matrix_sim = similarity_matrix.clone()
            diagonal_mask = torch.eye(N, device=feats.device).unsqueeze(0).expand(B, N, N)
            similarity_matrix_sim = similarity_matrix_sim - diagonal_mask # ignore them self

        unsim_attn_mask = self.build_fixed_attn_mask(similarity_matrix_unsim, largest=False, valid_count=valid_count)
        sim_attn_mask = self.build_fixed_attn_mask(similarity_matrix_sim, largest=True, valid_count=valid_count)
        
        feats = feats + self.aggregator_mlp(self.aggregator(query=feats, 
                                              key=feats, 
                                              value=feats,
                                              attn_mask=sim_attn_mask)[0])
        
        feats = feats + self.aggregator_mlp_2(self.aggregator_2(query=feats, 
                                              key=feats, 
                                              value=feats,
                                              attn_mask=unsim_attn_mask)[0])

        return feats, similarity_matrix, consist_feats
    

class Extra_Modal_Modeling(nn.Module):
    
    def __init__(self, num_head, output_dim, tok_num):
        super().__init__()

        self.feat_encoder = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.cross_encoder = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.token_number = tok_num

        self.consist_encoder_feat = nn.Sequential(nn.Linear(output_dim, 256),
                                                  nn.LayerNorm(256),
                                                  nn.GELU(),
                                                  nn.Linear(256, 128),
                                                  nn.LayerNorm(128),
                                                  nn.GELU(),
                                                  nn.Linear(128, 64))
        
        self.consist_encoder_cross = nn.Sequential(nn.Linear(output_dim, 256),
                                                  nn.LayerNorm(256),
                                                  nn.GELU(),
                                                  nn.Linear(256, 128),
                                                  nn.LayerNorm(128),
                                                  nn.GELU(),
                                                  nn.Linear(128, 64))
        
        self.cls_token_cross = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.aggregator_cross = nn.MultiheadAttention(output_dim, num_head, dropout=0.0, batch_first=True)
        self.norm_layer_cross =nn.LayerNorm(output_dim)

        self.cls_token_feat = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.aggregator_feat = nn.MultiheadAttention(output_dim, num_head, dropout=0.0, batch_first=True)
        self.norm_layer_feat =nn.LayerNorm(output_dim)

        self.aggregator = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.aggregator_2 = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp_2 = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.fixed_token_number = tok_num

        trunc_normal_(self.cls_token_cross, std=.02)
        trunc_normal_(self.cls_token_feat, std=.02)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def compute_fixed_k(self, valid_count):
        fixed_k = torch.minimum(
            torch.full_like(valid_count.long(), self.fixed_token_number),
            valid_count.long(),
        )
        return torch.clamp(fixed_k, min=1)

    def gather_fixed_tokens(self, feats, scores, fixed_k, largest):
        batch_size, num_tokens, _ = feats.shape
        sorted_index = torch.argsort(scores, dim=-1, descending=largest)
        max_k = int(fixed_k.max().item())
        selected_index = sorted_index[:, :max_k]
        gather_index = selected_index.unsqueeze(-1).expand(-1, -1, feats.shape[-1])
        selected_tokens = feats.gather(1, gather_index)
        selected_mask = torch.arange(max_k, device=feats.device).view(1, max_k) >= fixed_k.unsqueeze(-1)
        return selected_tokens, selected_mask
    
    def forward(self, feats, gloabl_feature, cross_feat, feats_mask, cross_mask):
        
        bs, _, _ = feats.shape

        feats = self.feat_encoder(feats)
        cross_feat = self.cross_encoder(cross_feat)

        cls_token_cross = self.cls_token_cross.expand(bs, -1, -1)
        feat_aggr_cross = self.aggregator_cross(query=self.norm_layer_cross(cls_token_cross), 
                                            key=self.norm_layer_cross(cross_feat), 
                                            value=self.norm_layer_cross(cross_feat),
                                            key_padding_mask=cross_mask)[0]
        
        feats_consist = self.consist_encoder_feat(feats)
        cross_feats_consist = self.consist_encoder_feat(feat_aggr_cross)

        norms_feat = torch.norm(feats_consist, p=2, dim=2, keepdim=True)
        norms_cross = torch.norm(cross_feats_consist, p=2, dim=2, keepdim=True)
        sim_matrix = torch.bmm(feats_consist/norms_feat, (cross_feats_consist/norms_cross).transpose(1, 2))
        sim_matrix = torch.clamp((sim_matrix+1)/2, 0, 1).squeeze(-1)

        cls_token = self.cls_token_feat.expand(bs, -1, -1)
        global_feats_mask = torch.zeros(feats_mask.shape[0], 1).bool().to(feats_mask.device)
        feat_aggr = self.aggregator_feat(query=self.norm_layer_feat(cls_token), 
                                            key=self.norm_layer_feat(torch.cat([gloabl_feature, feats], dim=1)), 
                                            value=self.norm_layer_feat(torch.cat([gloabl_feature, feats], dim=1)),
                                            key_padding_mask=torch.cat([global_feats_mask,feats_mask],dim=1))[0]
        
        if feats_mask.sum() > 0: # for text inputs
            sim_score = sim_matrix.clone()
            sim_score[feats_mask] = -1

            unsim_score = sim_matrix.clone()
            unsim_score[feats_mask] = 2
            valid_count = (~feats_mask).sum(dim=-1)

        else: # for image inputs
            sim_score = sim_matrix.clone()
            unsim_score = sim_matrix.clone()
            valid_count = torch.full((feats.shape[0],), feats.shape[1], device=feats.device, dtype=torch.long)

        fixed_k = self.compute_fixed_k(valid_count)
        unsim_patch, unsim_patch_mask = self.gather_fixed_tokens(feats, unsim_score, fixed_k, largest=False)
        sim_patch, sim_patch_mask = self.gather_fixed_tokens(feats, sim_score, fixed_k, largest=True)

        feat_aggr = feat_aggr + self.aggregator_mlp(self.aggregator(query=feat_aggr, 
                                              key=sim_patch, 
                                              value=sim_patch,
                                              key_padding_mask=sim_patch_mask)[0])
        
        feat_aggr = feat_aggr + self.aggregator_mlp_2(self.aggregator_2(query=feat_aggr, 
                                              key=unsim_patch, 
                                              value=unsim_patch,
                                              key_padding_mask=unsim_patch_mask)[0])
        
        return feat_aggr, sim_matrix, feats_consist
