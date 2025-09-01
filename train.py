# train.py
import os
import os, time, math, pickle
from contextlib import nullcontext
import datetime
import wandb

import torch.distributed as dist


import numpy as np
import torch
from model import GPTConfig, GPT

# NEW: hydra/omegaconf
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import process_group_manager as pgm
from process_group_manager import setup_process_group_manager

os.environ["DEVICE"] = "cuda"
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
backend = "nccl"

dist.init_process_group(rank=global_rank, world_size=world_size, backend=backend, init_method=f"env://", timeout=datetime.timedelta(minutes=2))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ----- config normalization -----
    dtype = torch.bfloat16
    config = OmegaConf.to_container(cfg, resolve=True)  # for logging/checkpoint parity
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    device_type = 'cuda'


    print(f"local rank {local_rank}")
    if local_rank == 0: print(f"cfg {cfg}")

    # dist stuff
    setup_process_group_manager(dp_size=cfg.dp_size, pp_size=cfg.pp_size, tp_size=cfg.tp_size)

    is_wandb_rank = pgm.process_group_manager.tp_rank == 0 and pgm.process_group_manager.dp_rank == 0 and pgm.process_group_manager.pp_is_last_stage
    # set seed
    

    tokens_per_iter = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)


    # data dir: keep relative to project root
    data_dir = os.path.join(to_absolute_path('data'), cfg.dataset)

    def get_batch(split):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+cfg.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+cfg.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # init globals that were previously overridden
    iter_num = 0
    best_val_loss = 1e9

    # infer vocab_size if present
    meta_vocab_size = None
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size,
        bias=cfg.bias, vocab_size=None, dropout=cfg.dropout
    )
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
        model_args['block_size'] = cfg.block_size
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    dist.barrier()
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
    dist.barrier()

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        if cfg.decay_lr is False:
            return cfg.learning_rate
        if it < cfg.warmup_iters:
            return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    if cfg.wandb_log:
        if is_wandb_rank:
            wandb.init(
                project="nanogpt_dist",
                name=f"{cfg.wandb_run_name}_{pgm.process_group_manager}",
                config={
                    "tensor_parallel_size": pgm.process_group_manager.tp_world_size,
                    "pipeline_parallel_size": pgm.process_group_manager.pp_world_size,
                    "data_parallel_size": pgm.process_group_manager.dp_world_size,
                    "model": cfg.model_name,
                    "learning_rate": cfg.learning_rate,
                    "seed": cfg.seed,
                },
            )

    X, Y = get_batch('train')
    t0 = time.time()
    local_iter_num = 0
    raw_model = model

    # ensure out_dir exists (relative to project root)
    out_dir_abs = to_absolute_path(cfg.out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    while True:
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if cfg.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })
        if iter_num == 0 and cfg.eval_only:
            break

        for micro_step in range(cfg.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0:
            lossf = loss.item() * cfg.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1
        local_iter_num += 1

        if iter_num > cfg.max_iters:
            break

if __name__ == "__main__":
    main()
