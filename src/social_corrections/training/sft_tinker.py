"""Supervised fine-tuning on correction pairs, using Tinker.

Usage:
    python -m social_corrections.training.sft_tinker \\
        --train-path data/processed/train_sft.jsonl \\
        --val-path data/processed/val_sft.jsonl \\
        --model-name meta-llama/Llama-3.1-8B-Instruct \\
        --output-name my-sft-run

This is patterned on ``tinker-cookbook/tinker_cookbook/recipes/sl_loop.py``
but adapted for our small correction dataset and made self-contained (no
cookbook dependency other than ``tinker`` itself).

Input JSONL rows: ``{"messages": [{"role": "user"|"assistant"|"system", "content": "..."}, ...]}``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    train_path: str
    val_path: str | None
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_name: str = "social-corrections-sft"
    lora_rank: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 8
    max_length: int = 2048
    eval_every: int = 50
    save_every: int = 100
    seed: int = 0
    log_path: str = "/tmp/tinker-social-corrections/sft"
    base_url: str | None = None


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _example_to_tokens(
    example: dict[str, Any],
    renderer,
    max_length: int,
):
    """Render a chat example into a Tinker Datum using the cookbook renderer.

    We import the cookbook renderer inside the function so that the module
    can be imported without the cookbook installed (e.g. for tests).
    """
    from tinker_cookbook.renderers import TrainOnWhat  # type: ignore
    from tinker_cookbook.supervised.data import conversation_to_datum  # type: ignore

    return conversation_to_datum(
        example["messages"],
        renderer,
        max_length,
        TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )


def _batched(lst: list[Any], batch_size: int) -> list[list[Any]]:
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def _mean_loss(training_client, batch, adam_params) -> float:
    """Helper: compute forward loss without stepping optimizer. Used for val."""
    from tinker_cookbook.supervised.common import compute_mean_nll  # type: ignore

    fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
    # For eval we still need the grads to be discarded, which Tinker does by
    # skipping optim_step. We just don't call optim_step here. Call .result()
    # so the future resolves.
    fwd_bwd_result = fwd_bwd_future.result()
    logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
    weights = [d.loss_fn_inputs["weights"] for d in batch]
    return float(compute_mean_nll(logprobs, weights))


def train(cfg: SFTConfig) -> None:
    import tinker  # type: ignore
    from tinker_cookbook import model_info, renderers  # type: ignore
    from tinker_cookbook.supervised.common import compute_mean_nll  # type: ignore
    from tinker_cookbook.tokenizer_utils import get_tokenizer  # type: ignore

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    Path(cfg.log_path).mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using model={cfg.model_name} renderer={renderer_name} lora_rank={cfg.lora_rank}")

    train_rows = _load_jsonl(cfg.train_path)
    val_rows = _load_jsonl(cfg.val_path) if cfg.val_path else []
    logger.info(f"Loaded {len(train_rows)} train rows, {len(val_rows)} val rows")

    train_data = [_example_to_tokens(r, renderer, cfg.max_length) for r in train_rows]
    val_data = [_example_to_tokens(r, renderer, cfg.max_length) for r in val_rows]

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=cfg.model_name, rank=cfg.lora_rank
    )

    rng = random.Random(cfg.seed)
    total_steps = 0
    best_val = float("inf")
    metrics_log: list[dict[str, Any]] = []

    for epoch in range(cfg.num_epochs):
        rng.shuffle(train_data)
        batches = _batched(train_data, cfg.batch_size)
        logger.info(f"Epoch {epoch + 1}/{cfg.num_epochs}: {len(batches)} batches")

        for step_in_epoch, batch in enumerate(batches):
            t0 = time.time()

            # Linear LR decay across total epochs
            progress = total_steps / max(1, cfg.num_epochs * len(batches))
            lr_mult = max(0.0, 1.0 - progress)
            current_lr = cfg.learning_rate * lr_mult
            adam = tinker.AdamParams(
                learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
            )

            fwd = training_client.forward_backward(batch, loss_fn="cross_entropy")
            opt = training_client.optim_step(adam)
            fwd_result = fwd.result()
            _ = opt.result()

            logprobs = [x["logprobs"] for x in fwd_result.loss_fn_outputs]
            weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = float(compute_mean_nll(logprobs, weights))

            row: dict[str, Any] = {
                "step": total_steps,
                "epoch": epoch,
                "train_nll": train_nll,
                "lr": current_lr,
                "time_sec": time.time() - t0,
            }

            if val_data and cfg.eval_every > 0 and total_steps % cfg.eval_every == 0:
                val_batches = _batched(val_data, cfg.batch_size)
                val_losses = [_mean_loss(training_client, vb, adam) for vb in val_batches]
                val_nll = sum(val_losses) / max(1, len(val_losses))
                row["val_nll"] = val_nll
                if val_nll < best_val:
                    best_val = val_nll
                    row["best_val"] = True

            metrics_log.append(row)
            if total_steps % 10 == 0:
                logger.info(json.dumps(row))

            if cfg.save_every > 0 and total_steps > 0 and total_steps % cfg.save_every == 0:
                save_name = f"{cfg.output_name}-step{total_steps:05d}"
                training_client.save_state(save_name)
                logger.info(f"Saved intermediate state: {save_name}")

            total_steps += 1

    # Final save + sampling client
    final_name = f"{cfg.output_name}-final"
    save_result = training_client.save_weights_for_sampler(name=final_name).result()
    model_path = save_result.path
    logger.info(f"Saved final weights as {final_name} at {model_path}")

    # Log summary
    summary = {
        "config": cfg.__dict__,
        "total_steps": total_steps,
        "best_val_nll": best_val if best_val != float("inf") else None,
        "final_model_path": model_path,
    }
    with open(Path(cfg.log_path) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(Path(cfg.log_path) / "metrics.jsonl", "w", encoding="utf-8") as f:
        for row in metrics_log:
            f.write(json.dumps(row) + "\n")

    logger.info(f"Training complete. Summary written to {cfg.log_path}/summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tinker SFT on correction pairs.")
    ap.add_argument("--train-path", required=True)
    ap.add_argument("--val-path", default=None)
    ap.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output-name", default="social-corrections-sft")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--num-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-path", default="/tmp/tinker-social-corrections/sft")
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()
    train(SFTConfig(**vars(args)))


if __name__ == "__main__":
    main()
