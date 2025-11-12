"""Module for annotating Utterance objects with additional metadata or labels."""

from pathlib import Path
from typing import Literal
from datatypes.corpus import Corpus, Utterance
import pandas as pd
import hydra
import multiprocessing
from hydra.utils import log
from omegaconf import OmegaConf
from lmstudio import BaseModel
import yaml
from components.utils import call_chat_model, get_provider, chat_model_llama_cpp
from tqdm import tqdm
from components.prompts.loader import render_prompt, render_user_prompt

from .prompts.response_formats import (
    CounsellorUtterance_t1,
    CounsellorUtterance_t2,
    CounsellorUtterance_flat,
    ClientUtterance_flat,
    ClientUtterance_t1,
    ClientUtterance_t2
)


global_call_chat_model = None
global_cfg = None
global_provider = None

def pool_initializer(cfg):
    global global_call_chat_model, global_cfg, global_provider
    global_cfg = cfg
    global_provider = get_provider(cfg.annotator.model)
    if global_provider == "llama_cpp":
        llama_model = chat_model_llama_cpp(model_name=cfg.annotator.model)
        global_call_chat_model = llama_model.call_model
    else:
        global_call_chat_model = call_chat_model

def mp_annotate_utterance_wrapper(args):
    row, df = args
    return annotate_utterance(row, df, global_cfg, global_provider, global_call_chat_model)
def annotate_utterance(row, df, cfg, provider, chat_paradigm):
    speaker = row["speaker"]
    utt_idx = row["corp_utt_idx"]
    utterance = row["utt_text"]
    context = Annotator(cfg).get_context_excerpt(df, utt_idx, cfg.annotator.context_mode, cfg.annotator.num_context_turns)
    user_prompt = render_user_prompt(
        transcript=context,
        speaker=speaker,
        utterance=utterance
    )
    match (cfg.annotator.class_structure):
        case ("tiered"):
            t1_system_prompt = render_prompt(speaker=speaker,structure="t1")
            t1_messages = [
                {'role': 'system', 'content': t1_system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            t1 = chat_paradigm(
                messages=t1_messages,
                model=cfg.annotator.model, 
                provider=provider,
                response_format=CounsellorUtterance_t1 if speaker=="counsellor" else ClientUtterance_t1,
                temperature=cfg.annotator.temperature
            )
            t2_system_prompt = render_prompt(speaker=speaker,structure="t2",label=t1['label'])
            t2_messages = [
                {'role': 'system', 'content': t2_system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            t2 = chat_paradigm(
                messages=t2_messages,
                model=cfg.annotator.model, 
                provider=provider,
                response_format=CounsellorUtterance_t2 if speaker=="counsellor" else ClientUtterance_t2,
                temperature=cfg.annotator.temperature
            )
            return {
                **row.to_dict(),
                "t1_label_auto": t1['label'],
                "t1_expl_auto": t1['explanation'],
                "t2_label_auto": t2['label'],
                "t2_expl_auto": t2['explanation'],
            }
        case ("flat"):
            flat_system_prompt = render_prompt(speaker=speaker,structure="flat")
            messages = [
                {'role': 'system', 'content': flat_system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            res = chat_paradigm(
                messages=messages,
                model=cfg.annotator.model, 
                provider=provider,
                response_format=CounsellorUtterance_flat if speaker=="counsellor" else ClientUtterance_flat,
                temperature=cfg.annotator.temperature
            )
            return {
                **row.to_dict(),
                "t2_label_auto": res['label'],
                "t2_expl_auto": res['explanation'],
            }



class Annotator:
    """Annotator for adding annotations to Utterance objects."""

    def __init__(self, cfg):
        """
        Initialize annotator with configuration.
        """
        self.cfg = cfg
        self.provider = get_provider(cfg.annotator.model)

        if self.provider == "llama_cpp":
            llama_model = chat_model_llama_cpp(model_name=cfg.annotator.model) # dont need to reload everytime = waste
            self.chat_paradigm = llama_model.call_model
        else:
            self.chat_paradigm = call_chat_model

    def get_context_excerpt(self, df, utt_idx: int, context_mode: Literal["all", "cumulative", "interval"], num_context_turns: int = 0
                            ) -> str:
        '''
        Get context excerpt for the utterance based on the specified context mode.
        '''
        row = df.iloc[utt_idx]
        conv_id = row["conv_id"]
        conv_utt_idx = row["conv_utt_idx"]
        conv_vol_idx = row["conv_vol_idx"]
        conv_df = df[df["conv_id"] == conv_id].reset_index(drop=True)

        if context_mode == "all":
            context_df = conv_df
        elif context_mode == "cumulative":
            context_df = conv_df[conv_df["conv_utt_idx"] <= conv_utt_idx]
        elif context_mode == "interval":
            vol_start = max(0, conv_vol_idx - num_context_turns)
            prev_vol_df = conv_df[
                (conv_df["conv_vol_idx"] >= vol_start) &
                (conv_df["conv_vol_idx"] < conv_vol_idx)
            ]
            curr_vol_df = conv_df[
                (conv_df["conv_vol_idx"] == conv_vol_idx) &
                (conv_df["conv_utt_idx"] <= conv_utt_idx)
            ]
            context_df = pd.concat([prev_vol_df, curr_vol_df])
        else:
            raise ValueError(f"Invalid context mode: {context_mode}")

        formatted_segments = []
        prev_speaker = None
        segment = ""

        for _, row in context_df.iterrows():
            speaker = row['speaker']
            text = row['utt_text']
            if speaker != prev_speaker:
                if segment:
                    formatted_segments.append(segment.strip())
                segment = f"{speaker}: {text}"
            else:
                segment += f" {text}"
            prev_speaker = speaker

        if segment:
            formatted_segments.append(segment.strip())

        return "\n".join(formatted_segments)
    
    def annotate_corpus(self, corp: Corpus):
        """
        """
        df = corp.to_df()
        output_rows = []
        exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        output_dir = Path('data/annotated')
        fn = (
            f"{self.cfg.input_dataset.name}_"
            f"{self.cfg.input_dataset.subset}_"
            f"{self.cfg.annotator.class_structure}_"
            f"{self.cfg.annotator.model.rsplit('/', 1)[-1]}_"
            f"{self.cfg.annotator.context_mode}_"
            f"{self.cfg.annotator.num_context_turns if self.cfg.annotator.context_mode == 'interval' else ''}" 
            f"_annotated.csv"
        )
        save_path = output_dir / fn
        exp_save_path = exp_output_dir / fn

        utt_checkpoint = 0
        existing_df = None
        if save_path.exists():
            try:
                existing_df = pd.read_csv(save_path)
                if not existing_df.empty:
                    utt_checkpoint = existing_df["corp_utt_idx"].max()
                    log.info(f"Found existing annotations up to utterance {utt_checkpoint} at {save_path}.")
            except Exception as e:
                log.warning(f"Error reading existing annotations: {e}")
        else:
            log.info(f"No existing annotations found. Starting from beginning.")

    

        try:
            grouped_df = df.groupby("conv_id", sort=False)
            if self.cfg.annotator.n_processes == 1:
                for _, (conv_id, volleys) in enumerate(tqdm(grouped_df, desc="Corpus", unit="conv", leave=False)):
                    conv = volleys.reset_index(drop=True)
                    for _, row in tqdm(list(conv.iterrows()), desc=f"Conv {conv_id}", unit="utt", leave=False):
                        if row["corp_utt_idx"] <= utt_checkpoint and utt_checkpoint != 0:
                            continue
                        annotated_row = annotate_utterance(row, df, self.cfg, self.provider, self.chat_paradigm)
                        output_rows.append(annotated_row)
            else:
                rows_to_annotate = []
                for _, (conv_id, volleys) in enumerate(tqdm(grouped_df, desc="Corpus", unit="conv", leave=False)):
                    conv = volleys.reset_index(drop=True)
                    for _, row in conv.iterrows():
                        if row["corp_utt_idx"] <= utt_checkpoint and utt_checkpoint != 0:
                            continue
                        rows_to_annotate.append((row, df))
                try:
                    with multiprocessing.Pool(processes=self.cfg.annotator.n_processes, initializer=pool_initializer, initargs=(self.cfg,)) as pool:
                        for result in tqdm(pool.imap(mp_annotate_utterance_wrapper, rows_to_annotate), total=len(rows_to_annotate)):
                            output_rows.append(result)
                except KeyboardInterrupt:
                    pool.terminate()
                    pool.join()
        except Exception as e:
            log.warning(f"Error annotating corpus: {e}", exc_info=True)
        finally:
            if output_rows:
                output_df = pd.DataFrame(output_rows)
                if existing_df is not None and save_path.exists():
                    output_df = pd.concat([existing_df, output_df], ignore_index=True)
                output_df.to_csv(save_path, index=False)
                output_df.to_csv(exp_save_path, index=False)
                log.info(f"Annotated utterances saved to {save_path}")
            else:
                log.warning("No utterances annotated from corpus.")
        return
    

    