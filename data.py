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

def train_spm_for_lang(files, model_prefix: str, vocab_size=4000, character_coverage=1.0):
    if not os.path.exists(f"{model_prefix}.model"):
        spm.SentencePieceTrainer.train(
            input=",".join(files),
            model_prefix=model_prefix,  # 輸出檔名
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type="unigram",
            input_sentence_size=int(1e6),
            shuffle_input_sentence=True,
            normalization_rule_name="nmt_nfkc_cf",
        )
    else:
        print(f"{model_prefix}.model exists. skipping training.")

# 建立 spm8000 model
# 利用 spm8000 model 生成 subword
def subword_units_separate(src: str, tgt: str, vs_src=4000, vs_tgt=4000):
    # 1) 為每個語言各自訓練一個 SPM
    en_files = [f"data/clean.{src}"]
    zh_files = [f"data/clean.{tgt}"]

    # 英文 coverage 1.0，中文建議 0.9995（涵蓋更多字形）
    train_spm_for_lang( en_files, model_prefix=f"spm{vs_src}.{src}", vocab_size=vs_src, character_coverage=0.9999)
    train_spm_for_lang( zh_files, model_prefix=f"spm{vs_tgt}.{tgt}", vocab_size=vs_tgt, character_coverage=0.9995)

    # 2) 載入兩個處理器
    sp_src = spm.SentencePieceProcessor()
    sp_src.load(f"spm{vs_src}.{src}.model")

    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(f"spm{vs_tgt}.{tgt}.model")

    src_in = f"data/clean.{src}"
    tgt_in = f"data/clean.{tgt}"
    src_out = f"data/train/subword.{src}"
    tgt_out = f"data/train/subword.{tgt}"

    with open(src_in, "r", encoding="utf8") as rf_src, \
            open(tgt_in, "r", encoding="utf8") as rf_tgt, \
            open(src_out, "w", encoding="utf8") as wf_src, \
            open(tgt_out, "w", encoding="utf8") as wf_tgt:
        for s_line, t_line in zip(rf_src, rf_tgt):
            s_tok = sp_src.encode(s_line.strip(), out_type=str)
            t_tok = sp_tgt.encode(t_line.strip(), out_type=str)
            wf_src.write(" ".join(s_tok) + "\n")
            wf_tgt.write(" ".join(t_tok) + "\n")

# 1. dict.en.txt 和 dict.zh.txt 紀錄每個 subword 出現幾次
# 2. bin, 將 subword 轉成 token id，連續存放
# 3. idx, 是句子的「起始位置與長度」索引表 (offset0 從第幾個字開始是這句 , length0 這句多長)
def preprocess_fairseq_py(src: str, tgt: str):
    datadir = Path("data")
    destdir = Path("data-bin") / "spm4000"
    destdir.mkdir(parents=True, exist_ok=True)

    argv = [
        "fairseq-preprocess",
        "--source-lang", src,
        "--target-lang", tgt,
        "--trainpref", str(datadir / "train/subword"),
        "--destdir", str(destdir),
        "--workers", "4",
    ]

    # 執行並檢查結果
    subprocess.run(argv, capture_output=True, text=True)
    print(f"完成！二進位資料在：{destdir}")

def get_dataloader(split = 'train', batch_size = 16, shuffle = True):
    bin_dir = 'data-bin/spm4000'
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

# pre-process
if __name__ == '__main__':
    clean_corpus('en', 'zh')
    subword_units_separate('en', 'zh')
    preprocess_fairseq_py('en', 'zh')

