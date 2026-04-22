"""Tinker-based training scripts.

The two entry points:
    - sft_tinker.py: supervised fine-tuning on correction pairs.
    - dpo_tinker.py: direct preference optimization on (bad, better) pairs.

Both follow the pattern of ``tinker_cookbook/recipes/sl_loop.py`` and do their
own flat training loops against the Tinker primitives (``forward_backward``,
``optim_step``, ``save_state``). We avoid the cookbook's higher-level
abstractions so the scripts are easier to read as a finished research artifact.
"""
