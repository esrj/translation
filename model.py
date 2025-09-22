import torch
import torch.nn as nn
import math

class SelfAttentionEncoder(nn.Module):
    def __init__(
        self,
        src_embed,
        padding_idx,
        max_seq_len,
        embed_dim=512,
        num_heads=2,
        ff_dim=2048,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.src_embed = src_embed
        self.embed_dim = embed_dim

        self.pos_embedding = nn.Embedding(max_seq_len,embed_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = ff_dim,
            dropout = dropout,
            activation = 'relu',
            batch_first = False         # (seq , batch , embed)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers
        )

    def forward(self,src_tokens):
        batch_size, seq_len = src_tokens.size()

        # 建立 [0,1,2,…,seq_len-1] 的一維張量
        positions = (torch.arange(seq_len, device=src_tokens.device)).unsqueeze(0).expand(batch_size, -1)
        x = self.src_embed(src_tokens) * math.sqrt(self.embed_dim)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)

        x = x.transpose(0, 1)  # (batch, seq_len) -> (seq_len, batch)

        # 長度不夠的地方被補成 PAD，需要被忽略。
        # key_padding_mask 會將資料部分寫成 False，PAD 部分寫成 True
        key_padding_mask = src_tokens.eq(self.padding_idx)

        outputs = self.transformer_encoder(
            x,
            src_key_padding_mask = key_padding_mask
        )

        return outputs, key_padding_mask # encoder 輸入跟輸出長度一樣，因此 padding_mask 可以直接用


class SelfAttentionDecoder(nn.Module):
    def __init__(
        self,
        tgt_embed,
        tgt_vocab_size,
        bos_idx,
        padding_idx,
        max_seq_len,
        embed_dim = 512,
        num_heads = 2,
        ff_dim = 2048,
        num_layers = 4,
        dropout = 0.1,
    ):
        super().__init__()
        self.tgt_embed     = tgt_embed
        self.embed_dim     = embed_dim
        self.padding_idx   = padding_idx
        self.bos_idx       = bos_idx

        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 堆疊多層 TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # 最後的 linear 投影到詞表
        self.output_proj = nn.Linear(embed_dim, tgt_vocab_size, bias=False)
        self.output_proj.weight = self.tgt_embed.weight

    def forward(self, tgt_tokens, encoder_outputs, encoder_padding_mask):

        batch_size, target_len = tgt_tokens.size()

        # 右移：在最前面插 BOS，丟掉最後一個 token (EOS)
        shift_input = torch.cat([
            torch.full((batch_size, 1), self.bos_idx, device=tgt_tokens.device, dtype=tgt_tokens.dtype),
            tgt_tokens[:, :-1]  # 丟掉最後一個 token (EOS)
        ], dim=1)

        # token + pos embedding
        target_len = shift_input.size(1)

        positions = torch.arange(target_len, device=tgt_tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.tgt_embed(shift_input) * math.sqrt(self.embed_dim)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        x = x.transpose(0, 1)

        # masks
        # causal mask: 上三角設 True，讓 Decoder 只能 attend 左邊或自己
        causal_mask = torch.triu(
            torch.ones((target_len, target_len), device=x.device, dtype=torch.bool),
            diagonal=1
        )
        # padding mask: True 表示該位置是 pad，要被忽略
        tgt_key_padding_mask = shift_input.eq(self.padding_idx)

        # 呼叫多層 TransformerDecoder
        dec_out = self.transformer_decoder(
            x,
            memory=encoder_outputs,
            tgt_mask=causal_mask, # mask self attention 用
            tgt_key_padding_mask=tgt_key_padding_mask, # decoder 輸入時的 mask
            memory_key_padding_mask=encoder_padding_mask,  # encoder 輸入時的 mask
        )

        # 投影到 vocab-size → logits
        logits = self.output_proj(dec_out)
        return logits.transpose(0, 1)



