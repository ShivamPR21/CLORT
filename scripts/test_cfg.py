from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb


def flatten_cfg(cfg: DictConfig) -> Dict[str, Any]:
    dct = {}

    for key1 in cfg.keys():
        for key2 in cfg[key1].keys():
            dct.update({f'{key1}.{key2}': cfg[key1][key2]})

    return dct

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    wandb.login()
    _run = wandb.init(
        project = "Test",

        config= flatten_cfg(cfg)
    )

    print(cfg.dataset.train_splits)
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
