import re
import os
import random
import sentencepiece as spm
import subprocess
from pathlib import Path
from fairseq.data import indexed_dataset, Dictionary
from fairseq.data.language_pair_dataset import LanguagePairDataset
from torch.utils.data import DataLoader

# 把「全形字元」轉成「半形字元」
# 也就是把寬度佔兩個位元（通常用在中文輸入法裡的全形標點、英數字）轉換成一般的 ASCII 半形字元
def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全形空格直接轉換
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全形字元（除空格）根據關係轉化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace('-', '')  # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s)  # 這些標點符號前後加入空格, \1 就是選中的字
    elif lang == 'zh':
        s = strQ2B(s)  # Q2B
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s)  # keep punctuation
    s = ' '.join(s.strip().split())
    return s

def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    else:
        return len(s.split())

def clean_corpus(src, tgt):
    if os.path.exists(f'data/clean.en') and os.path.exists(f'data/clean.zh'):
        print(f'clean.en & zh exists. skipping clean.')
        return
    total = 0
    with open(f'data/raw.{src}', 'r') as l1_in_f:
        with open(f'data/raw.{tgt}', 'r') as l2_in_f:
            with open(f'data/clean.{src}', 'w') as l1_out_f:
                with open(f'data/clean.{tgt}', 'w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()  # 讀下一行

                        s1 = clean_s(s1, src)
                        s2 = clean_s(s2, tgt)

                        s1_len = len_s(s1, src)
                        s2_len = len_s(s2, tgt)

                        if s1_len < 1 or s2_len < 1:
                            continue
                        if s1_len > 1000 or s2_len > 1000:
                            continue
                        if s1_len / s2_len > 9 or s2_len / s1_len > 9:
                            continue
                        total += 1
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)
    print(f"Data : {total}")

# def train_val_data_writing(src, tgt):
#     if os.path.exists(f'data/clean.en') and os.path.exists(f'data/clean.zh'):
#         line_num = sum(1 for _ in open(f'data/clean.{src}'))
#         labels = list(range(line_num))
#
#         random.shuffle(labels)  # 給所有 pairs 資料隨機的編號
#         for lang in [src, tgt]:
#             with open(f'data/train/clean.{lang}', 'w') as train_f:
#                 with open(f'data/val/clean.{lang}', 'w') as test_f:
#                     count = 0
#                     for line in open(f'data/clean.{lang}', 'r'):
#                         if labels[count] / line_num < 0.9:  # 90% pairs training
#                             train_f.write(line)
#                         else:
#                             test_f.write(line)
#                         count += 1
#         # 刪除切分前的，避免佔空間
#         for lang in ["en", "zh"]:
#             path = f"data/clean.{lang}"
#             if os.path.exists(path):
#                 os.remove(path)

def subword_units(src: str, tgt: str):
    vocab_size = 16000
    model_prefix = f'spm{vocab_size}'
    model_file = f'{model_prefix}.model'

    # 1. train if needed
    if os.path.exists(model_file):
        print(f'{model_file} exists. skipping training.')
    else:
        spm.SentencePieceTrainer.train(
            input=','.join([
                f'data/clean.{src}',
                f'data/clean.{tgt}',
            ]),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type='unigram',
            input_sentence_size=int(1e6),
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc_cf',
        )
    # 2. load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)

    # 3. apply to train & val
    src_in = f'data/clean.{src}'
    tgt_in = f'data/clean.{tgt}'
    src_out = f'data/train/subword.{src}'
    tgt_out = f'data/train/subword.{tgt}'

    # 這裡同時開四個檔案
    with open(src_in, 'r', encoding='utf8') as rf_src, \
            open(tgt_in, 'r', encoding='utf8') as rf_tgt, \
            open(src_out, 'w', encoding='utf8') as wf_src, \
            open(tgt_out, 'w', encoding='utf8') as wf_tgt:

        for s_line, t_line in zip(rf_src, rf_tgt):
            s_tok = sp.encode(s_line.strip(), out_type=str)
            t_tok = sp.encode(t_line.strip(), out_type=str)
            wf_src.write(' '.join(s_tok) + '\n')
            wf_tgt.write(' '.join(t_tok) + '\n')

# 1. dict.en.txt 和 dict.zh.txt 兩個檔案 (一樣，因為共享詞表) 是紀錄每個 subword 出現幾次
# 2. bin, 將 subword 轉成 token id，連續存放
# 3. idx, 是句子的「起始位置與長度」索引表 (offset0 從第幾個字開始是這句 , length0 這句多長)
def preprocess_fairseq_py(src: str, tgt: str):
    # 假設你的檔名是：data/train.subword.{lang}, data/val.subword.{lang}, data/val.subword.{lang}
    datadir = Path("data")
    destdir = Path("data-bin") / "spm16000"
    destdir.mkdir(parents=True, exist_ok=True)

    argv = [
        "fairseq-preprocess",
        "--source-lang", src,
        "--target-lang", tgt,
        "--trainpref", str(datadir / "train/subword"),
        "--destdir", str(destdir),
        "--joined-dictionary",                         # ← 共用 SPM 建議用
        "--workers", "4",
    ]

    # 執行並檢查結果
    subprocess.run(argv, capture_output=True, text=True)
    print(f"完成！二進位資料在：{destdir}")

def get_dataloader(split: str, batch_size: int, shuffle: bool):
    bin_dir = 'data-bin/spm16000'
    src, tgt = 'en', 'zh'
    # prefix 直接對到 train.en-zh.en.{bin,idx}
    src_prefix = os.path.join(bin_dir, f'{split}.{src}-{tgt}.{src}')
    tgt_prefix = os.path.join(bin_dir, f'{split}.{src}-{tgt}.{tgt}')

    # MMapIndexedDataset 會自動去找 .bin 和 .idx，然後用 memory‐map 的方式載入整個語料庫
    src_data = indexed_dataset.MMapIndexedDataset(src_prefix)
    tgt_data = indexed_dataset.MMapIndexedDataset(tgt_prefix)

    # 依照檔案「行號」分配給每個 token 一個整數 ID，因此為 token -> id 對應檔案
    src_dict = Dictionary.load(os.path.join(bin_dir, 'dict.en.txt'))
    tgt_dict = Dictionary.load(os.path.join(bin_dir, 'dict.zh.txt'))

    dataset = LanguagePairDataset(
        src_data, src_data.sizes, src_dict,
        tgt_data, tgt_data.sizes, tgt_dict,
        left_pad_source=False,
        left_pad_target=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collater,
        num_workers=0,
    )

# data pre-process
if __name__ == '__main__':
    clean_corpus('en', 'zh')
    subword_units('en', 'zh')
    preprocess_fairseq_py('en', 'zh')

# test data loader
# if __name__ == '__main__':
#     train_loader = get_dataloader('train', batch_size=32, shuffle=True)
#     valid_loader = get_dataloader('valid', batch_size=32, shuffle=False)
#
#     # 印出一筆資料檢查使否正確
#     train_iter = iter(train_loader)
#     first_batch = next(train_iter)
#
#     src = first_batch['net_input']['src_tokens']
#     tgt = first_batch['target']
#
#     # 印出第 0 筆資料（batch 裡的第一個句對）
#     print("第一筆 src ids:", src[0].tolist())
#     print("第一筆 tgt ids:", tgt[0].tolist())
#
#
#     # 假設你已經有 bin/dict.en.txt, dict.zh.txt
#     src_dict = Dictionary.load('data-bin/spm8000/dict.en.txt')
#     tgt_dict = Dictionary.load('data-bin/spm8000/dict.zh.txt')
#
#     # decode 第一筆
#     src_tokens = src[0].tolist()
#     tgt_tokens = tgt[0].tolist()
#     print("第一筆 src text:", src_dict.string(src_tokens, bpe_symbol='@@'))
#     print("第一筆 tgt text:", tgt_dict.string(tgt_tokens, bpe_symbol='@@'))