
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import AutoConfig, AutoModel
from timm.models.vision_transformer import VisionTransformer

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, txt_feat, img_feat):
        # Attend from text to image
        attn_output, _ = self.attn(txt_feat.unsqueeze(1), img_feat.unsqueeze(1), img_feat.unsqueeze(1))
        fused = self.norm(txt_feat + attn_output.squeeze(1))
        return fused

class MMTDPlus(nn.Module):
    def __init__(self,
                 text_model='bert-base-multilingual-cased',
                 image_model='facebook/deit-base-distilled-patch16-224',
                 lang_embed_dim=32):
        super(MMTDPlus, self).__init__()

        # Text encoder (BERT)
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.text_hidden = self.text_encoder.config.hidden_size

        # Language embedding
        self.lang_embedding = nn.Embedding(20, lang_embed_dim)  # support up to 20 language IDs

        # Image encoder (ViT-like model)
        self.image_encoder = AutoModel.from_pretrained(image_model)
        self.image_hidden = self.image_encoder.config.hidden_size

        # Fusion
        fusion_input_dim = self.text_hidden + self.image_hidden + lang_embed_dim
        self.cross_fusion = CrossAttentionFusion(dim=fusion_input_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_input_dim // 2, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values, lang_ids):
        # Text
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.pooler_output  # [B, hidden]

        # Language embedding
        lang_emb = self.lang_embedding(lang_ids)  # [B, lang_embed_dim]

        # Image
        image_out = self.image_encoder(pixel_values=pixel_values)
        image_feat = image_out.pooler_output  # [B, hidden]

        # Combine all features
        combined = torch.cat([text_feat, image_feat, lang_emb], dim=1)

        # Cross-modal attention fusion
        fused_feat = self.cross_fusion(combined, combined)

        # Classification
        logits = self.classifier(fused_feat)
        return logits
