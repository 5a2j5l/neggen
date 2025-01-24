import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers=3, lambda_param=0.5, tau=0.1, alpha=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lambda_param = lambda_param
        self.tau = tau
        self.alpha = alpha
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        
        self.proj_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def self_attention(self, embeddings):
        q = k = v = embeddings.unsqueeze(0)
        attn_output, _ = self.attention(q, k, v)
        return attn_output.squeeze(0)
        
    def forward(self, user_ids, item_ids, modal_embeddings, neg_modal_embeddings):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        modal_attn = self.self_attention(modal_embeddings)
        neg_modal_attn = self.self_attention(neg_modal_embeddings)
        
        causal_effect = modal_attn - neg_modal_attn
        
        causal_emb = self.proj_network(causal_effect)
        neg_causal_emb = self.proj_network(neg_modal_attn)
        
        base_score = torch.sum(user_emb * item_emb, dim=1)
        causal_score = torch.sum(user_emb * causal_emb, dim=1)
        
        final_score = base_score + self.lambda_param * causal_score
        
        return final_score, causal_emb, neg_causal_emb, item_emb
        
    def calculate_loss(self, user_ids, item_ids, modal_embeddings, neg_modal_embeddings):
        scores, causal_emb, neg_causal_emb, item_emb = self.forward(
            user_ids, item_ids, modal_embeddings, neg_modal_embeddings
        )
        
        rec_loss = -torch.mean(
            torch.log(torch.sigmoid(scores - torch.sum(
                self.user_embedding(user_ids) * neg_causal_emb, dim=1
            )))
        )
        
        align_loss = -torch.mean(
            torch.log(
                torch.exp(torch.sum(causal_emb * item_emb, dim=1) / self.tau) /
                torch.exp(torch.sum(neg_causal_emb * item_emb, dim=1) / self.tau)
            )
        )
        
        total_loss = rec_loss + self.alpha * align_loss
        
        return total_loss, rec_loss, align_loss