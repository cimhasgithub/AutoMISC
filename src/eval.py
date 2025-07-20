import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import os
import openai
import logging
from hydra.utils import log
from validation.IRR import IRR
from validation.relate_outcomes import relate_outcomes

@hydra.main(config_path="../conf", config_name="eval_config.yaml", version_base=None)
def eval(cfg: DictConfig) -> None:
    openai.api_key = os.environ['OPENAI_API_KEY']
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("Starting AutoMISC evals with configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    if cfg.method == 'IRR':
        IRR(cfg)

    elif cfg.method == 'relate_outcomes':
        relate_outcomes(cfg)
    else:
        return

if __name__ == "__main__":
    eval()