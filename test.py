from data import get_dataloader,strQ2B,clean_s,len_s
from fairseq.data import Dictionary
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from model import SelfAttentionEncoder,SelfAttentionDecoder
import sentencepiece as spm

test_input = 'Now I have to take off my shoes or boots to get on an airplane!'

test_input_clean = clean_s(strQ2B(test_input),'en')

sp = spm.SentencePieceProcessor()
sp.load('spm4000.en.model')
pieces = sp.encode(test_input_clean.strip(), out_type=str)
test_input_line = " ".join(pieces)

src_dict = Dictionary.load('data-bin/spm4000/dict.en.txt')
tgt_dict = Dictionary.load('data-bin/spm4000/dict.zh.txt')

test_input_token = src_dict.encode_line(test_input_line, add_if_not_exist=False, append_eos=False).long()

pad_id = src_dict.pad()
eos_id = src_dict.eos()
batch = test_input_token.unsqueeze(0)  # (1, B, S_max)

# =============== load model ===============

device = 'cpu'
src_embed = nn.Embedding(len(src_dict), 512, padding_idx=src_dict.pad())
tgt_embed = nn.Embedding(len(tgt_dict), 512, padding_idx=tgt_dict.pad())

encoder = SelfAttentionEncoder(src_embed=src_embed, padding_idx=src_dict.pad(), max_seq_len=1024).to(device)
decoder = SelfAttentionDecoder(tgt_embed=tgt_embed, tgt_vocab_size=len(tgt_dict),bos_idx=tgt_dict.bos(), padding_idx=tgt_dict.pad(),max_seq_len=1024).to(device)

model = nn.Module()
model.encoder = encoder
model.decoder = decoder
model.load_state_dict(torch.load('transformer.pt', map_location="cpu"), strict=True)


# =============== testing ===============
y = []  # 已生成的 target token ids（不含 <bos>/<eos>）

model.eval()
with torch.no_grad():
    enc_out, enc_padding_mask = model.encoder(batch)
    for _ in range(1024):
        # 關鍵：用「y + [pad]」當輸入；decoder 內會 prepend BOS、drop last
        tgt_tokens = torch.tensor([y + [pad_id]], dtype=torch.long, device=device)  # (1, L+1)

        logits = model.decoder(tgt_tokens, enc_out, enc_padding_mask)  # (1, L+1, V)
        next_id = logits[:, -1, :].argmax(dim=-1).item()

        if next_id == eos_id or next_id == pad_id:
            break
        y.append(next_id)

subword_line = tgt_dict.string(torch.tensor(y), escape_unk=False)
pieces = subword_line.split()   # 變回 list[str]
tgt_text = sp.decode(pieces)

print(test_input)
print(tgt_text)