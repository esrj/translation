import torch
from model import SelfAttentionDecoder, SelfAttentionEncoder
import torch.nn as nn
from tqdm.auto import tqdm
from fairseq.data import Dictionary
import os
from data import get_dataloader

# 裝置設定
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# data loader
train_loader = get_dataloader('train', batch_size=16, shuffle=True)

# 單字對應成 embedding layer
# src_dict 和 tgt_dict 為兩個不同的詞表
src_dict = Dictionary.load(os.path.join('data-bin','spm4000', 'dict.en.txt'))
tgt_dict = Dictionary.load(os.path.join('data-bin','spm4000', 'dict.zh.txt'))
src_embed = nn.Embedding(len(src_dict), 512, padding_idx=src_dict.pad())
tgt_embed = nn.Embedding(len(tgt_dict), 512, padding_idx=tgt_dict.pad())

# Transformer
pad_idx = tgt_dict.pad()
bos_idx = tgt_dict.bos()
encoder = SelfAttentionEncoder(src_embed = src_embed,padding_idx=pad_idx,max_seq_len=1024).to(device)
decoder = SelfAttentionDecoder(tgt_embed = tgt_embed, tgt_vocab_size=len(tgt_dict) ,bos_idx = bos_idx ,padding_idx = pad_idx ,max_seq_len=1024).to(device)

# setting
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
model = nn.Module()
model.encoder = encoder
model.decoder = decoder
model.to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 1.0,
    betas = (0.9,0.98),  # β₁ 一階動量（類似 momentum）越接近 1 保留越多歷史
    eps = 1e-9,   # 避免除以 0
    weight_decay=1e-4
)
d_model = 512
warmup = 8000
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda s: (d_model ** -0.5) * min((s+1) ** -0.5, (s+1) * (warmup ** -1.5))
)

# training
for epoch in range(15):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
    for batch in loop:
        src_tokens = batch['net_input']['src_tokens'].to(device)
        tgt_tokens = batch['target'].to(device)

        enc_out, enc_padding_mask = model.encoder(src_tokens)
        logits = model.decoder(tgt_tokens, enc_out, enc_padding_mask)

        B, T, V = logits.size()  # (batch_size, seq_len, vocab_size)
        logits = logits.reshape(-1, V) # (batch_size*seq_len, vocab_size)
        labels = tgt_tokens.view(-1) # (batch_size*seq_len)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"epoch {epoch:2d} | avg train loss: {avg_loss:.4f}")

model = model.to("cpu")
torch.save(model.state_dict(), f'transformer.pt')
model = model.to(device)