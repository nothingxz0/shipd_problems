#!/usr/bin/env python3
"""
ONNX Autopsy: Seq2Seq Model with Custom Byte-Level BPE Tokenizer
Compatible with eval.py interface.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Optional: tokenizers library
try:
    from tokenizers import ByteLevelBPETokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("Warning: 'tokenizers' library not installed. Falling back to character-level tokenizer.")

# -----------------------------------------------------------------------------
# Reproducibility and Environment
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_COUNT = max(1, (os.cpu_count() or 1))
os.environ["OMP_NUM_THREADS"] = str(CPU_COUNT)
torch.set_num_threads(CPU_COUNT)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class Config:
    # Tokenizer
    BPE_VOCAB_SIZE = 4096
    MAX_INPUT_TOKENS = 8192
    # Model
    EMBED_DIM = 256
    ENC_LAYERS = 4
    DEC_LAYERS = 4
    ENC_ATTN_HEADS = 8
    DEC_ATTN_HEADS = 8
    FF_DIM = 1024
    LINFORMER_K = 128
    MAX_TARGET_LEN = 64
    DROPOUT = 0.15
    # Training
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
    EPOCHS = 40
    WARMUP_STEPS = 500
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    # Inference
    BEAM_SIZE = 3
    # Data
    VAL_SPLIT = 0.1
    # Augmentation
    AUG_BYTE_DROPOUT = 0.02
    AUG_BLOCK_SHUFFLE = 0.3

cfg = Config()

# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------
LAYER_TYPES = [
    "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm", "InstanceNorm2d",
    "ReLU", "GELU", "SiLU", "Mish", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "PReLU",
    "Hardswish", "Hardtanh", "ReLU6", "Softmax",
    "MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d", "AdaptiveAvgPool2d",
    "Dropout", "Dropout2d", "AlphaDropout",
    "LSTM", "GRU", "RNN",
    "MultiheadAttention",
    "Linear", "Embedding", "Bilinear",
    "Flatten", "Upsample", "PixelShuffle",
]
assert len(LAYER_TYPES) == 42

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Add UNK_TOKEN to target vocabulary as well
TARGET_ID2TOKEN = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + LAYER_TYPES
TARGET_TOKEN2ID = {tok: i for i, tok in enumerate(TARGET_ID2TOKEN)}
TARGET_VOCAB_SIZE = len(TARGET_ID2TOKEN)

# -----------------------------------------------------------------------------
# Helper functions from original solution (for compatibility)
# -----------------------------------------------------------------------------
def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def parse_target_sequence(raw: str) -> List[str]:
    try:
        value = json.loads(raw)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str)]

def has_required_dataset_files(directory: Path) -> bool:
    return directory.is_dir() and (directory / "train.csv").exists() and (directory / "test.csv").exists()

def resolve_data_dir() -> Path:
    candidates = [
        Path("./dataset/public"),
        Path("./dataset"),
        Path("../dataset/public"),
        Path("../dataset"),
        Path("/kaggle/input"),
    ]
    for path in candidates:
        if has_required_dataset_files(path):
            return path.resolve()
    raise FileNotFoundError("Could not locate dataset directory containing train.csv and test.csv.")

# -----------------------------------------------------------------------------
# Tokenizer (Byte-level BPE or fallback character-level)
# -----------------------------------------------------------------------------
class CharLevelTokenizer:
    """Fallback character-level tokenizer for hex strings."""
    def __init__(self):
        self.vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + list("0123456789abcdef")
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}

    def encode(self, text: str):
        ids = [self.token2id.get(ch, self.token2id['<unk>']) for ch in text]
        return type('Encoding', (), {'ids': ids})()

    def get_vocab_size(self):
        return len(self.vocab)

    def token_to_id(self, token):
        return self.token2id.get(token, self.token2id['<unk>'])

def train_byte_bpe_tokenizer(hex_strings: List[str], vocab_size: int):
    """Train a ByteLevelBPETokenizer on the raw hex strings."""
    if not TOKENIZERS_AVAILABLE:
        print("Using character-level tokenizer fallback.")
        return CharLevelTokenizer()

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        hex_strings,
        vocab_size=vocab_size,
        special_tokens=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
    )

    # Wrap to provide a consistent interface
    class WrappedTokenizer:
        def __init__(self, tok):
            self.tok = tok
        def encode(self, text: str):
            enc = self.tok.encode(text)
            return type('Encoding', (), {'ids': enc.ids})()
        def get_vocab_size(self):
            return self.tok.get_vocab_size()
        def token_to_id(self, token):
            return self.tok.token_to_id(token)
    return WrappedTokenizer(tokenizer)

# -----------------------------------------------------------------------------
# Data Augmentation (on hex string)
# -----------------------------------------------------------------------------
def augment_hex_string(hex_str: str) -> str:
    """Apply augmentations directly to the hex string."""
    # Convert to list of characters for manipulation
    chars = list(hex_str)
    if cfg.AUG_BYTE_DROPOUT > 0:
        for i in range(len(chars)):
            if random.random() < cfg.AUG_BYTE_DROPOUT:
                chars[i] = '0'
    # Block shuffling: split at high-entropy points (e.g., every 32 chars)
    if cfg.AUG_BLOCK_SHUFFLE > 0 and random.random() < cfg.AUG_BLOCK_SHUFFLE:
        block_size = 32
        chunks = [chars[i:i+block_size] for i in range(0, len(chars), block_size)]
        random.shuffle(chunks)
        chars = sum(chunks, [])
    return ''.join(chars)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class ONNXDataset(Dataset):
    def __init__(self, hex_strings, target_seqs, tokenizer, target_token2id, is_train=False):
        self.hex_strings = hex_strings
        self.targets = target_seqs
        self.tokenizer = tokenizer
        self.target_token2id = target_token2id
        self.is_train = is_train
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.sos_id = tokenizer.token_to_id(SOS_TOKEN)
        self.eos_id = tokenizer.token_to_id(EOS_TOKEN)
        self.unk_id = tokenizer.token_to_id(UNK_TOKEN)

    def __len__(self):
        return len(self.hex_strings)

    def __getitem__(self, idx):
        hex_str = self.hex_strings[idx]
        if self.is_train:
            hex_str = augment_hex_string(hex_str)

        encoding = self.tokenizer.encode(hex_str)
        input_ids = encoding.ids
        if len(input_ids) > cfg.MAX_INPUT_TOKENS:
            input_ids = input_ids[:cfg.MAX_INPUT_TOKENS]

        target_tokens = self.targets[idx]
        target_ids = [self.target_token2id[SOS_TOKEN]]
        for tok in target_tokens:
            target_ids.append(self.target_token2id.get(tok, self.target_token2id[UNK_TOKEN]))
        target_ids.append(self.target_token2id[EOS_TOKEN])
        if len(target_ids) > cfg.MAX_TARGET_LEN:
            target_ids = target_ids[:cfg.MAX_TARGET_LEN - 1] + [self.target_token2id[EOS_TOKEN]]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_input_len = max(len(x) for x in inputs)
    max_target_len = max(len(y) for y in targets)
    padded_inputs = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    for i, x in enumerate(inputs):
        padded_inputs[i, :len(x)] = x
    padded_targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    for i, y in enumerate(targets):
        padded_targets[i, :len(y)] = y
    return padded_inputs, padded_targets

# -----------------------------------------------------------------------------
# Model Components (unchanged)
# -----------------------------------------------------------------------------
class LinformerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, k, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k = k
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _pool_seq_len(self, x: torch.Tensor, out_len: int) -> torch.Tensor:
        """Project sequence length L -> out_len while preserving head_dim."""
        bsz, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(bsz * heads, head_dim, seq_len)
        x = F.adaptive_avg_pool1d(x, out_len)
        x = x.view(bsz, heads, head_dim, out_len).permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        h = self.num_heads
        Q = self.q_proj(x).view(batch_size, seq_len, h, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, h, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, h, self.head_dim).transpose(1, 2)

        proj_len = min(self.k, seq_len)
        K_proj = self._pool_seq_len(K, proj_len)
        V_proj = self._pool_seq_len(V, proj_len)

        attn = torch.matmul(Q, K_proj.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V_proj)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(context)

class LinformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, k, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = LinformerAttention(embed_dim, num_heads, k, dropout)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class ConvProjection(nn.Module):
    def __init__(self, embed_dim, kernel_size=5, stride=2):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size, stride=stride, padding=kernel_size // 2)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.transpose(1, 2)
        return x

class LinformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, k, ff_dim, dropout, max_len=8192):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.conv_proj = ConvProjection(embed_dim)
        self.layers = nn.ModuleList([
            LinformerEncoderLayer(embed_dim, num_heads, k, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        x = self.conv_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, dropout, max_len=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_ids, memory, tgt_mask=None):
        seq_len = tgt_ids.size(1)
        positions = torch.arange(seq_len, device=tgt_ids.device).unsqueeze(0).expand(tgt_ids.size(0), -1)
        x = self.token_embedding(tgt_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask)
        x = self.norm(x)
        return self.output_proj(x)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_ids, tgt_ids, tgt_mask=None):
        memory = self.encoder(src_ids)
        return self.decoder(tgt_ids, memory, tgt_mask)

    def encode(self, src_ids):
        return self.encoder(src_ids)

    def decode(self, tgt_ids, memory, tgt_mask=None):
        return self.decoder(tgt_ids, memory, tgt_mask)

# -----------------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------------
def create_mask(size):
    return torch.triu(torch.ones(size, size, device=DEVICE) * float('-inf'), diagonal=1)

def train_epoch(model, dataloader, optimizer, criterion, scheduler, epoch, scaler):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, (src, tgt) in enumerate(pbar):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = create_mask(tgt_input.size(1))

        with torch.amp.autocast('cuda', enabled=DEVICE.type=='cuda'):
            logits = model(src, tgt_input, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        loss = loss / cfg.GRAD_ACCUM
        scaler.scale(loss).backward()

        if (step + 1) % cfg.GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * cfg.GRAD_ACCUM
        pbar.set_postfix({"loss": f"{loss.item()*cfg.GRAD_ACCUM:.4f}"})
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Validation"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = create_mask(tgt_input.size(1))
        logits = model(src, tgt_input, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)

def beam_search_decode(model, src_ids, beam_size, max_len, sos_id, eos_id, pad_id):
    model.eval()
    src_ids = src_ids.unsqueeze(0).to(DEVICE)
    memory = model.encode(src_ids)

    beams = [([sos_id], 0.0, False)]
    completed = []

    for _ in range(max_len - 1):
        new_beams = []
        for seq, log_prob, finished in beams:
            if finished:
                new_beams.append((seq, log_prob, finished))
                continue
            tgt_ids = torch.tensor([seq], device=DEVICE)
            tgt_mask = create_mask(len(seq))
            logits = model.decode(tgt_ids, memory, tgt_mask)
            next_logits = logits[0, -1, :]
            probs = F.log_softmax(next_logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, beam_size)
            for prob, tok in zip(topk_probs, topk_ids):
                new_seq = seq + [tok.item()]
                new_log_prob = log_prob + prob.item()
                if tok.item() == eos_id or len(new_seq) >= max_len:
                    completed.append((new_seq, new_log_prob))
                else:
                    new_beams.append((new_seq, new_log_prob, False))
        if not new_beams:
            break
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

    if not completed:
        completed = beams
    best_seq = max(completed, key=lambda x: x[1])[0]
    if best_seq[0] == sos_id:
        best_seq = best_seq[1:]
    if best_seq and best_seq[-1] == eos_id:
        best_seq = best_seq[:-1]
    return best_seq

def ids_to_tokens(ids, id2token):
    return [id2token[i] for i in ids if i not in [0, 1, 2]]

# -----------------------------------------------------------------------------
# Main Training and Prediction Function (Interface for eval.py)
# -----------------------------------------------------------------------------
def build_predictions(train_rows: List[Dict], test_rows: List[Dict]) -> List[Tuple[str, ...]]:
    """
    Train a Seq2Seq model on train_rows and predict sequences for test_rows.
    Returns list of tuples of layer type strings.
    """
    start_time = time.time()
    print(f"Running on {DEVICE}")
    print(f"Train samples: {len(train_rows)}, Test samples: {len(test_rows)}")

    # Extract data
    train_hex = [r["onnx_hex"] for r in train_rows]
    train_targets = [parse_target_sequence(r.get("target_sequence", "[]")) for r in train_rows]
    test_hex = [r["onnx_hex"] for r in test_rows]

    # Train tokenizer
    print("Training tokenizer...")
    tokenizer = train_byte_bpe_tokenizer(train_hex, cfg.BPE_VOCAB_SIZE)

    # Split train into train/val for monitoring
    val_size = int(len(train_rows) * cfg.VAL_SPLIT)
    if val_size > 0:
        train_hex_sub = train_hex[:-val_size]
        train_targets_sub = train_targets[:-val_size]
        val_hex = train_hex[-val_size:]
        val_targets = train_targets[-val_size:]
    else:
        train_hex_sub, train_targets_sub = train_hex, train_targets
        val_hex, val_targets = [], []

    # Create datasets
    train_dataset = ONNXDataset(train_hex_sub, train_targets_sub, tokenizer, TARGET_TOKEN2ID, is_train=True)
    val_dataset = ONNXDataset(val_hex, val_targets, tokenizer, TARGET_TOKEN2ID, is_train=False) if val_hex else None

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=min(4, CPU_COUNT), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=min(2, CPU_COUNT), pin_memory=True
    ) if val_dataset else None

    # Build model
    encoder = LinformerEncoder(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=cfg.EMBED_DIM,
        num_layers=cfg.ENC_LAYERS,
        num_heads=cfg.ENC_ATTN_HEADS,
        k=cfg.LINFORMER_K,
        ff_dim=cfg.FF_DIM,
        dropout=cfg.DROPOUT,
    )
    decoder = TransformerDecoder(
        vocab_size=TARGET_VOCAB_SIZE,
        embed_dim=cfg.EMBED_DIM,
        num_layers=cfg.DEC_LAYERS,
        num_heads=cfg.DEC_ATTN_HEADS,
        ff_dim=cfg.FF_DIM,
        dropout=cfg.DROPOUT,
    )
    model = Seq2Seq(encoder, decoder).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=TARGET_TOKEN2ID[PAD_TOKEN], label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    total_steps = (len(train_loader) // cfg.GRAD_ACCUM) * cfg.EPOCHS
    warmup_steps = cfg.WARMUP_STEPS

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type=='cuda') if DEVICE.type=='cuda' else torch.amp.GradScaler(enabled=False)

    best_val_loss = float("inf")
    for epoch in range(cfg.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, epoch, scaler)
        if val_loader:
            val_loss = validate(model, val_loader, criterion)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), Path("./working/best_model.pt"))
                print("  -> Saved best model")
        else:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
            torch.save(model.state_dict(), Path("./working/best_model.pt"))

    # Load best model
    model.load_state_dict(torch.load(Path("./working/best_model.pt")))
    model.eval()

    # Predict on test set
    print("Generating predictions...")
    predictions = []
    sos_id = TARGET_TOKEN2ID[SOS_TOKEN]
    eos_id = TARGET_TOKEN2ID[EOS_TOKEN]
    pad_id = TARGET_TOKEN2ID[PAD_TOKEN]

    test_dataset = ONNXDataset(test_hex, [[] for _ in test_hex], tokenizer, TARGET_TOKEN2ID, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for src, _ in tqdm(test_loader, desc="Predicting"):
        src = src.to(DEVICE)
        pred_ids = beam_search_decode(
            model, src[0], beam_size=cfg.BEAM_SIZE,
            max_len=cfg.MAX_TARGET_LEN, sos_id=sos_id, eos_id=eos_id, pad_id=pad_id
        )
        pred_tokens = ids_to_tokens(pred_ids, TARGET_ID2TOKEN)
        predictions.append(tuple(pred_tokens))

    print(f"Training + prediction completed in {(time.time() - start_time)/60:.2f} minutes")
    return predictions

# -----------------------------------------------------------------------------
# Main (if run directly, for full test set submission)
# -----------------------------------------------------------------------------
def main():
    data_dir = resolve_data_dir()
    train_rows = read_csv_rows(data_dir / "train.csv")
    test_rows = read_csv_rows(data_dir / "test.csv")
    predictions = build_predictions(train_rows, test_rows)

    out_path = Path("./working/submission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target_sequence"])
        for row, seq in zip(test_rows, predictions):
            writer.writerow([row["id"], json.dumps(list(seq))])
    print(f"Submission saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()