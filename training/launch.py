# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
import warnings

def main():
    parser = argparse.ArgumentParser(description="Train VGGT model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()

    # Optional: ignore deprecation warnings to reduce log clutter
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Load Hydra configuration
    with initialize(version_base=None, config_path="config"):
        cfg: DictConfig = compose(config_name=args.config)
        # Optionally print config for debugging
        print(OmegaConf.to_yaml(cfg))

    # --- Pass the config sections as keyword arguments ---
    trainer = Trainer(
        data=cfg.data,
        model=cfg.model,
        logging=cfg.logging,
        checkpoint=cfg.checkpoint,
        max_epochs=cfg.max_epochs,
        mode="train",
        device="cuda",
        seed_value=cfg.seed_value,
        val_epoch_freq=cfg.val_epoch_freq,
        distributed=cfg.distributed,
        cuda=cfg.cuda,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        optim=cfg.optim_conf,
        loss=cfg.loss_conf if hasattr(cfg, "loss_conf") else None,
        accum_steps=cfg.accum_steps,
    )

    # Run training
    trainer.run()


if __name__ == "__main__":
    main()