"""Entrypoint for AutoMISC automated behavioural code classifier."""
import json
from pathlib import Path

import hydra
from hydra.utils import log
from omegaconf import OmegaConf
from datatypes.corpus import Corpus
from components.parser import Parser
from components.annotator import Annotator
from datatypes.dataset import DatasetSpec
import os
import openai
import logging

DATASET_SPEC: list[DatasetSpec] = [
    DatasetSpec(name="random_install_multiBC_20convos",     filename="random_install_multiBC_20convos.csv",     id_col="Prolific ID",   volley_text="Volley",           speaker_col="Speaker"),
]

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg) -> None:
    openai.api_key = os.environ['OPENAI_API_KEY']
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("Starting AutoMISC run with configuration:\n%s", OmegaConf.to_yaml(cfg))
    dataset_spec = next((s for s in DATASET_SPEC if cfg.input_dataset.name.startswith(s.name)), None)

    # 1) Load volley-level dataset
    corpus = Corpus(cfg, dataset_spec, cfg.n_conversations)
    log.info(corpus)
    log.info("Dataset loaded. Now moving onto parsing")
    # corpus.save_to_csv()

    # 2) Parse each conversation into utterances
    parser = Parser(cfg, dataset_spec)
    parsed = parser.parse_corpus(corpus)
    log.info("Parsing complete. Now moving onto annotation")
    # parsed.save_to_csv()

    # 3) Annotate utterances
    annotator = Annotator(cfg, dataset_spec)
    annotator.annotate_corpus(parsed)

    log.info("Annotating done. Returning from main")
    return


if __name__ == "__main__":
    main()