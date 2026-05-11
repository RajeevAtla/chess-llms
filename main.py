# ============================================================
# Qwen2.5-0.5B Chess Self-Play Fine-Tuning with python-chess
# ============================================================



import os
import json
import math
import random
from dataclasses import dataclass
from contextlib import nullcontext
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import chess

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    MODEL_ID: str = "Qwen/Qwen2.5-0.5B"
    OUT_DIR: str = "qwen25_05b_chess_selfplay_lora"

    SEED: int = 42

    MAX_SEQ_LEN: int = 1024
    MAX_HISTORY_PLIES: int = 80

    NUM_SELFPLAY_ITERS: int = 10
    GAMES_PER_ITER: int = 20
    MAX_PLIES_PER_GAME: int = 120

    TEMPERATURE: float = 0.85
    TOP_K_MOVES: Optional[int] = 8
    RANDOM_MOVE_PROB: float = 0.03

    MODEL_WEIGHT: float = 1.0
    HEURISTIC_WEIGHT: float = 0.35

    KEEP_ONLY_WINNER_MOVES: bool = True
    INCLUDE_DRAW_MOVES: bool = False

    TRAIN_EPOCHS_PER_ITER: int = 1
    TRAIN_BATCH_SIZE: int = 2
    GRAD_ACCUM_STEPS: int = 8
    LR: float = 2e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.03
    MAX_GRAD_NORM: float = 1.0

    SCORE_BATCH_SIZE: int = 32

    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05


cfg = Config()

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed_all(cfg.SEED)

os.makedirs(cfg.OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    compute_dtype = torch.float32

print("Device:", device)
print("Compute dtype:", compute_dtype)

# ============================================================
# Load model and tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    cfg.MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    cfg.MODEL_ID,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
)

base_model.to(device)
base_model.config.use_cache = False

target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

lora_config = LoraConfig(
    r=cfg.LORA_R,
    lora_alpha=cfg.LORA_ALPHA,
    lora_dropout=cfg.LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules,
)

model = get_peft_model(base_model, lora_config)

try:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
except TypeError:
    model.gradient_checkpointing_enable()

model.print_trainable_parameters()

# ============================================================
# Chess formatting
# ============================================================

def board_side_name(board: chess.Board) -> str:
    return "White" if board.turn == chess.WHITE else "Black"


def legal_uci_moves(board: chess.Board) -> List[str]:
    return sorted([move.uci() for move in board.legal_moves])


def build_prompt(board: chess.Board, history_uci: List[str]) -> str:
    legal_moves = legal_uci_moves(board)
    recent_history = history_uci[-cfg.MAX_HISTORY_PLIES:]
    history_text = " ".join(recent_history) if recent_history else "<start>"

    prompt = (
        "You are a chess move policy.\n"
        "Choose exactly one legal move in UCI notation.\n\n"
        f"Side to move: {board_side_name(board)}\n"
        f"FEN: {board.fen()}\n"
        f"Legal moves: {' '.join(legal_moves)}\n"
        f"Recent move history UCI: {history_text}\n\n"
        "Best legal move:\n"
        "<move>"
    )

    return prompt


def build_answer(move_uci: str) -> str:
    return f"{move_uci}</move>"

# ============================================================
# Tokenization and dataset
# ============================================================

def tokenize_supervised_sample(prompt: str, answer: str) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]

    max_prompt_len = cfg.MAX_SEQ_LEN - len(answer_ids)

    if max_prompt_len <= 0:
        answer_ids = answer_ids[: cfg.MAX_SEQ_LEN]
        prompt_ids = []
    elif len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class ChessSFTDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.examples = [
            tokenize_supervised_sample(r["prompt"], r["answer"])
            for r in records
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def chess_collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])

        input_ids.append(x["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ============================================================
# Simple chess heuristic
# ============================================================

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate_board_white_pov(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)

    if outcome is not None:
        if outcome.winner == chess.WHITE:
            return 100000.0
        if outcome.winner == chess.BLACK:
            return -100000.0
        return 0.0

    score = 0.0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value

    return score


def heuristic_score_move(board: chess.Board, move: chess.Move) -> float:
    mover = board.turn

    board.push(move)

    score = evaluate_board_white_pov(board)

    if board.is_checkmate():
        score = 100000.0 if mover == chess.WHITE else -100000.0
    elif board.is_check():
        score += 25.0 if mover == chess.WHITE else -25.0

    board.pop()

    return score if mover == chess.WHITE else -score

# ============================================================
# Model scoring for legal moves
# ============================================================

def autocast_context():
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=compute_dtype)
    return nullcontext()


def pad_scoring_batch(examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    max_len = max(len(x["input_ids"]) for x in examples)

    input_ids = []
    attention_mask = []
    labels = []

    for x in examples:
        pad_len = max_len - len(x["input_ids"])

        input_ids.append(x["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


@torch.no_grad()
def score_uci_moves_with_model(
    model,
    board: chess.Board,
    history_uci: List[str],
    moves: List[chess.Move],
) -> np.ndarray:
    was_training = model.training
    model.eval()

    prompt = build_prompt(board, history_uci)
    all_scores = []

    for start in range(0, len(moves), cfg.SCORE_BATCH_SIZE):
        batch_moves = moves[start : start + cfg.SCORE_BATCH_SIZE]

        examples = [
            tokenize_supervised_sample(
                prompt,
                build_answer(move.uci()),
            )
            for move in batch_moves
        ]

        batch = pad_scoring_batch(examples)

        with autocast_context():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_input_ids = batch["input_ids"][:, 1:]
        shift_labels = batch["labels"][:, 1:]

        answer_mask = shift_labels.ne(-100)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_input_ids.unsqueeze(-1),
        ).squeeze(-1)

        scores = (
            (token_log_probs * answer_mask).sum(dim=-1)
            / answer_mask.sum(dim=-1).clamp_min(1)
        )

        all_scores.extend(scores.detach().float().cpu().numpy().tolist())

    if was_training:
        model.train()

    return np.array(all_scores, dtype=np.float32)

# ============================================================
# Legal move policy
# ============================================================

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32)

    if len(x) <= 1:
        return np.zeros_like(x)

    std = x.std()

    if std < 1e-6:
        return np.zeros_like(x)

    return (x - x.mean()) / std


def softmax_np(x: np.ndarray, temperature: float) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    temperature = max(temperature, 1e-6)

    x = x / temperature
    x = x - np.max(x)

    probs = np.exp(x)
    probs = probs / probs.sum()

    return probs


def choose_legal_move(
    model,
    board: chess.Board,
    history_uci: List[str],
) -> Tuple[chess.Move, Dict]:
    legal_moves = list(board.legal_moves)

    if len(legal_moves) == 0:
        raise ValueError("No legal moves available.")

    if random.random() < cfg.RANDOM_MOVE_PROB:
        move = random.choice(legal_moves)
        return move, {
            "policy": "random",
            "chosen_move": move.uci(),
        }

    model_scores = score_uci_moves_with_model(
        model=model,
        board=board,
        history_uci=history_uci,
        moves=legal_moves,
    )

    heuristic_scores = np.array(
        [heuristic_score_move(board, move) for move in legal_moves],
        dtype=np.float32,
    )

    combined_scores = (
        cfg.MODEL_WEIGHT * zscore(model_scores)
        + cfg.HEURISTIC_WEIGHT * zscore(heuristic_scores)
    )

    if cfg.TOP_K_MOVES is not None and cfg.TOP_K_MOVES < len(legal_moves):
        top_indices = np.argsort(combined_scores)[-cfg.TOP_K_MOVES:]
        filtered_scores = combined_scores[top_indices]
        probs = softmax_np(filtered_scores, cfg.TEMPERATURE)

        chosen_local_idx = np.random.choice(len(top_indices), p=probs)
        chosen_idx = int(top_indices[chosen_local_idx])
    else:
        probs = softmax_np(combined_scores, cfg.TEMPERATURE)
        chosen_idx = int(np.random.choice(len(legal_moves), p=probs))

    chosen_move = legal_moves[chosen_idx]

    debug = {
        "policy": "model_plus_heuristic",
        "chosen_move": chosen_move.uci(),
        "model_score": float(model_scores[chosen_idx]),
        "heuristic_score": float(heuristic_scores[chosen_idx]),
        "combined_score": float(combined_scores[chosen_idx]),
    }

    return chosen_move, debug

# ============================================================
# Self-play
# ============================================================

def play_selfplay_game(
    model,
    game_id: int,
    iteration: int,
) -> Tuple[List[Dict], Dict]:
    board = chess.Board()
    history_uci = []
    records = []

    for ply in range(cfg.MAX_PLIES_PER_GAME):
        if board.is_game_over(claim_draw=True):
            break

        prompt = build_prompt(board, history_uci)
        side_to_move_bool = board.turn
        side_to_move_name = board_side_name(board)

        move, debug = choose_legal_move(
            model=model,
            board=board,
            history_uci=history_uci,
        )

        records.append(
            {
                "iteration": iteration,
                "game_id": game_id,
                "ply": ply,
                "fen": board.fen(),
                "side_to_move": side_to_move_name,
                "side_to_move_bool": side_to_move_bool,
                "legal_moves": legal_uci_moves(board),
                "move": move.uci(),
                "prompt": prompt,
                "answer": build_answer(move.uci()),
                "debug": debug,
            }
        )

        board.push(move)
        history_uci.append(move.uci())

    outcome = board.outcome(claim_draw=True)

    if outcome is None:
        result = "1/2-1/2"
        winner = None
        termination = "max_plies"
    else:
        result = board.result(claim_draw=True)
        winner = outcome.winner
        termination = str(outcome.termination)

    for r in records:
        r["result"] = result
        r["winner"] = (
            "white" if winner == chess.WHITE
            else "black" if winner == chess.BLACK
            else "draw"
        )

    if cfg.KEEP_ONLY_WINNER_MOVES:
        if winner is None:
            filtered_records = records if cfg.INCLUDE_DRAW_MOVES else []
        else:
            filtered_records = [
                r for r in records
                if r["side_to_move_bool"] == winner
            ]
    else:
        if winner is None and not cfg.INCLUDE_DRAW_MOVES:
            filtered_records = []
        else:
            filtered_records = records

    for r in filtered_records:
        r.pop("side_to_move_bool", None)

    summary = {
        "iteration": iteration,
        "game_id": game_id,
        "result": result,
        "winner": (
            "white" if winner == chess.WHITE
            else "black" if winner == chess.BLACK
            else "draw"
        ),
        "plies": len(history_uci),
        "termination": termination,
        "moves_uci": history_uci,
        "kept_training_positions": len(filtered_records),
    }

    return filtered_records, summary


def save_jsonl(records: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def generate_selfplay_dataset(
    model,
    iteration: int,
) -> Tuple[List[Dict], List[Dict]]:
    all_records = []
    summaries = []

    for game_id in tqdm(
        range(cfg.GAMES_PER_ITER),
        desc=f"Self-play iteration {iteration}",
    ):
        records, summary = play_selfplay_game(
            model=model,
            game_id=game_id,
            iteration=iteration,
        )

        all_records.extend(records)
        summaries.append(summary)

    data_path = os.path.join(
        cfg.OUT_DIR,
        f"selfplay_iter_{iteration}.jsonl",
    )

    summary_path = os.path.join(
        cfg.OUT_DIR,
        f"selfplay_iter_{iteration}_summaries.json",
    )

    save_jsonl(all_records, data_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    results = {}

    for s in summaries:
        results[s["result"]] = results.get(s["result"], 0) + 1

    print(f"Saved {len(all_records)} training positions to {data_path}")
    print("Game results:", results)

    return all_records, summaries

# ============================================================
# Training
# ============================================================

def train_on_selfplay_records(
    model,
    records: List[Dict],
    iteration: int,
):
    if len(records) == 0:
        print("No training records for this iteration. Skipping training.")
        return

    dataset = ChessSFTDataset(records)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=chess_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(len(dataloader) / cfg.GRAD_ACCUM_STEPS)
    total_updates = max(1, updates_per_epoch * cfg.TRAIN_EPOCHS_PER_ITER)
    warmup_steps = int(cfg.WARMUP_RATIO * total_updates)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_update_step = 0
    losses = []

    for epoch in range(cfg.TRAIN_EPOCHS_PER_ITER):
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Training iter {iteration}, epoch {epoch + 1}",
        )

        for micro_step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast_context():
                outputs = model(**batch)
                loss = outputs.loss / cfg.GRAD_ACCUM_STEPS

            loss.backward()

            raw_loss = float(loss.detach().cpu()) * cfg.GRAD_ACCUM_STEPS
            losses.append(raw_loss)

            should_update = (
                ((micro_step + 1) % cfg.GRAD_ACCUM_STEPS == 0)
                or ((micro_step + 1) == len(dataloader))
            )

            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.MAX_GRAD_NORM,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_update_step += 1

            pbar.set_postfix(
                loss=float(np.mean(losses[-20:])),
                updates=global_update_step,
            )

    iter_out_dir = os.path.join(
        cfg.OUT_DIR,
        f"adapter_iter_{iteration}",
    )

    model.save_pretrained(iter_out_dir)
    tokenizer.save_pretrained(iter_out_dir)

    print(f"Saved LoRA adapter to: {iter_out_dir}")
    print(f"Mean train loss: {np.mean(losses):.4f}")

# ============================================================
# Inference helpers
# ============================================================

@torch.no_grad()
def predict_legal_move(
    model,
    fen: str,
    history_uci: Optional[List[str]] = None,
    verbose: bool = True,
) -> str:
    if history_uci is None:
        history_uci = []

    board = chess.Board(fen)

    move, debug = choose_legal_move(
        model=model,
        board=board,
        history_uci=history_uci,
    )

    if verbose:
        print("FEN:", fen)
        print("Side to move:", board_side_name(board))
        print("Chosen move:", move.uci())
        print("Debug:", debug)

    return move.uci()


def play_demo_game(model, max_plies: int = 80):
    board = chess.Board()
    history = []

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        move, debug = choose_legal_move(
            model=model,
            board=board,
            history_uci=history,
        )

        print(
            f"{ply + 1:03d}. "
            f"{board_side_name(board):5s} "
            f"{move.uci():6s} "
            f"policy={debug.get('policy')}"
        )

        board.push(move)
        history.append(move.uci())

    print()
    print(board)
    print()
    print("Result:", board.result(claim_draw=True))
    print("Moves:", " ".join(history))

    return history, board.result(claim_draw=True)

# ============================================================
# Main iterative self-play fine-tuning loop
# ============================================================

all_training_records = []

for iteration in range(cfg.NUM_SELFPLAY_ITERS):
    print("=" * 80)
    print(f"SELF-PLAY ITERATION {iteration}")
    print("=" * 80)

    records, summaries = generate_selfplay_dataset(
        model=model,
        iteration=iteration,
    )

    all_training_records.extend(records)

    train_on_selfplay_records(
        model=model,
        records=records,
        iteration=iteration,
    )

final_adapter_dir = os.path.join(cfg.OUT_DIR, "final_adapter")
model.save_pretrained(final_adapter_dir)
tokenizer.save_pretrained(final_adapter_dir)

print("=" * 80)
print("Done.")
print("Final adapter saved to:", final_adapter_dir)
print("Total training records:", len(all_training_records))
print("=" * 80)

# ============================================================
# Quick tests
# ============================================================

start_board = chess.Board()
print("Starting position prediction:")
predict_legal_move(model, start_board.fen())

print("\nDemo self-play game:")
demo_moves, demo_result = play_demo_game(model, max_plies=80)
