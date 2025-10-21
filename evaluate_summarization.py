import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

try:
    from bert_score import score as _bert_score
except Exception:
    _bert_score = None
import csv
import importlib


def _ensure_nltk_data():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    for pkg in ['punkt_tab', 'punkt']:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def load_raw_snippets(path: str) -> List[Dict[str, Any]]:
    """
    Load and flatten raw snippet data from JSON file.

    Args:
        path: Path to JSON file containing snippets organized by query

    Returns:
        List of flattened snippet dictionaries with keys:
        'query', 'snippet_number', 'title', 'url', 'text'
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    flat = []
    for block in data:
        query = block.get('query')
        for sn in block.get('snippets', []):
            flat.append({
                'query': query,
                'snippet_number': sn.get('number'),
                'title': sn.get('title'),
                'url': sn.get('full_url') or sn.get('url'),
                'text': sn.get('formatted_snippet') or sn.get('snippet') or ''
            })
    return flat


def load_reference_batches(path: str) -> List[Dict[str, Any]]:
    """
    Load reference summaries from JSON file.

    Args:
        path: Path to JSON file containing reference summaries

    Returns:
        List of dictionaries with 'combined_summary' key
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_snippets(flat_snippets: List[Dict[str, Any]], batch_size: int = 10) -> List[List[str]]:
    """
    Group flat snippets into batches of specified size.

    Args:
        flat_snippets: List of snippet dictionaries
        batch_size: Number of snippets per batch (default: 10)

    Returns:
        List of batches, where each batch is a list of snippet texts
    """
    return [[s.get('text', '') for s in flat_snippets[i:i + batch_size]] for i in
            range(0, len(flat_snippets), batch_size)]


class Summarizer:
    def __init__(self, model_local_dir: str = None, model_name: str = 'facebook/bart-large-cnn',
                 target_min_words: int = 100, target_max_words: int = 140):
        use_cuda = torch.cuda.is_available()
        prefer_fp16 = use_cuda and ('pegasus' not in (model_name or '').lower())

        self.model_name = model_name or ''

        # Target length boundaries in words
        self.target_min_words = max(1, int(target_min_words or 100))
        self.target_max_words = max(self.target_min_words, int(target_max_words or 140))
        # Tolerance for exceeding upper boundary (softness)
        self.overshoot_ratio = 0.15  # 15% over upper boundary is acceptable without trimming
        # Rough estimate of tokens/words ratio for different tokenizers
        name_l = (self.model_name or '').lower()
        if ('t5' in name_l) or ('pegasus' in name_l):
            self.tokens_per_word_est = 1.5
        else:
            self.tokens_per_word_est = 1.35

        if use_cuda:
            self.device = torch.device('cuda')
            # Prefer bfloat16 if GPU supports BF16; otherwise use FP16 for acceleration
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16 if prefer_fp16 else torch.float32
        else:
            self.device = torch.device('cpu')
            dtype = torch.float32
        load_path = model_local_dir if model_local_dir and os.path.isdir(model_local_dir) else model_name

        self.tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

        config = AutoConfig.from_pretrained(load_path, trust_remote_code=True)

        self.is_encoder_decoder = bool(getattr(config, 'is_encoder_decoder', False))

        self.has_chat_template = hasattr(self.tokenizer, 'apply_chat_template')
        self.system_prompt = """You are an expert summarization assistant. Your task is to create accurate, concise, and informative summaries that capture the key points and essential information from the given text."""

        self._quantized_8bit = False
        self._quantized_4bit = False
        self._using_device_map = False

        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                load_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        else:
            try:
                self.tokenizer.padding_side = 'left'
            except Exception:
                pass

            loaded = False

            if use_cuda and (BitsAndBytesConfig is not None):
                try:
                    print("[Model Loading] Attempting 8-bit loading...", flush=True)
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    load_kwargs = {
                        'trust_remote_code': True,
                        'quantization_config': bnb_config,
                        'device_map': 'auto',
                        'low_cpu_mem_usage': True,
                    }
                    self.model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs)
                    self._quantized_8bit = True
                    self._using_device_map = True
                    loaded = True
                    print("[Model Loading] 8-bit loading successful!", flush=True)
                except Exception as e:
                    print(f"[Model Loading] 8-bit loading failed: {e}", flush=True)

            if not loaded:
                if use_cuda:
                    print("[Model Loading] Switching to FP16/BF16 loading for GPT-like model", flush=True)
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        load_path,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        device_map='auto' if use_cuda else None,
                        low_cpu_mem_usage=True,
                    )
                    if use_cuda:
                        self._using_device_map = True
                except Exception as e:
                    print(f"[Model Loading] Failed to load model: {e}", flush=True)
                    raise

        if not self._using_device_map:
            self.model = self.model.to(self.device)
        self.model.eval()

        self.model_type = getattr(self.model.config, 'model_type', '') or ''
        default_limit = 2048
        max_len = getattr(self.model.config, 'max_position_embeddings', None)
        if isinstance(max_len, int) and max_len > 0:
            self.max_source_length = min(max_len, default_limit)
        else:
            tml = getattr(self.tokenizer, 'model_max_length', None)

            if isinstance(tml, int) and 8 <= tml < 32768:
                self.max_source_length = min(tml, default_limit)
            else:
                self.max_source_length = default_limit

        if self.tokenizer.pad_token_id is None and getattr(self.tokenizer, 'eos_token', None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, 'pad_token_id', None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = getattr(self.tokenizer, 'eos_token_id', None) or getattr(self.model.config, 'eos_token_id',
                                                                                     None)

        try:
            print(
                f"[Model] name='{model_name}', type='{self.model_type}', encoder_decoder={self.is_encoder_decoder}, "
                f"device='{self.device.type}', quant8={self._quantized_8bit}, quant4={self._quantized_4bit}, "
                f"device_map_auto={self._using_device_map}, max_source_length={self.max_source_length}",
                flush=True
            )
        except Exception:
            pass

    def _est_lengths_tokens(self, min_words: int, max_words: int) -> Dict[str, int]:
        # Slightly expand the corridor: min. by -10%, max. by +15% in words, then to tokens
        eff_min_words = max(1, int(round(min_words * 0.9)))
        eff_max_words = max(eff_min_words + 1, int(round(max_words * 1.15)))
        min_len = max(1, int(round(eff_min_words * self.tokens_per_word_est)))
        max_len = max(min_len + 8, int(round(eff_max_words * self.tokens_per_word_est)))
        return {'min_tokens': min_len, 'max_tokens': max_len}

    def _seq2seq_params(self, mode: str) -> Dict[str, Any]:
        name = (self.model_name or '').lower()
        mtype = (self.model_type or '').lower()

        params = {
            'num_beams': 4,
            'length_penalty': 1.2,
            'no_repeat_ngram_size': 4,
            'early_stopping': True,
            'do_sample': False,
        }
        if mode == 'snippet':
            params.update({'max_length': 128, 'min_length': 32})
        else:
            # Target 100–140 words (translating to tokens)
            lens = self._est_lengths_tokens(self.target_min_words, self.target_max_words)
            params.update({'max_length': lens['max_tokens'], 'min_length': lens['min_tokens']})

        if 'bart' in name or 'bart' in mtype:
            if mode == 'snippet':
                params.update({'max_length': 160, 'min_length': 40, 'length_penalty': 1.3, 'num_beams': 6})
            else:
                lens = self._est_lengths_tokens(self.target_min_words, self.target_max_words)
                params.update({'num_beams': 6, 'length_penalty': 1.2, 'max_length': lens['max_tokens'],
                               'min_length': lens['min_tokens'], 'no_repeat_ngram_size': 4})

        if 'pegasus' in name or 'pegasus' in mtype:
            if mode == 'snippet':
                params.update({'max_length': 128, 'min_length': 40, 'length_penalty': 0.9, 'num_beams': 8})
            else:
                lens = self._est_lengths_tokens(self.target_min_words, self.target_max_words)
                params.update({'num_beams': 8, 'length_penalty': 0.9, 'max_length': lens['max_tokens'],
                               'min_length': lens['min_tokens'], 'no_repeat_ngram_size': 4})

        if 'xsum' in name:
            if mode == 'snippet':
                params.update({'max_length': 96, 'min_length': 24, 'length_penalty': 1.0, 'num_beams': 6})
            else:
                lens = self._est_lengths_tokens(self.target_min_words, self.target_max_words)
                params.update({'max_length': lens['max_tokens'], 'min_length': lens['min_tokens'],
                               'length_penalty': 1.0, 'num_beams': 6})

        if 't5' in name or 't5' in mtype:
            if mode == 'snippet':
                params.update({'max_length': 144, 'min_length': 36, 'length_penalty': 1.0, 'num_beams': 4})
            else:
                lens = self._est_lengths_tokens(self.target_min_words, self.target_max_words)
                params.update({'max_length': lens['max_tokens'], 'min_length': lens['min_tokens'],
                               'length_penalty': 1.0, 'num_beams': 4})

        if 'arxiv' in name:
            if mode == 'snippet':
                params.update({'num_beams': 6, 'length_penalty': 1.1})
            else:
                params.update({'num_beams': 6, 'length_penalty': 1.1})

        params['pad_token_id'] = self.tokenizer.pad_token_id
        return params

    def _gpt_params(self, mode: str) -> Dict[str, Any]:
        name = (self.model_name or '').lower()

        params = {
            'do_sample': True,
            'top_p': 0.9,
            'temperature': 0.7,
            'repetition_penalty': 1.15,
            'no_repeat_ngram_size': 4,
            'num_beams': 1,
            'early_stopping': True,
        }
        if mode == 'snippet':
            params.update({'max_new_tokens': 80, 'min_new_tokens': 24})
        else:
            # 100–140 words → approximate through tokens
            lens = self._est_lengths_tokens(self.target_min_words, self.target_max_words)
            params.update({'max_new_tokens': lens['max_tokens'], 'min_new_tokens': lens['min_tokens']})

        if 'gemma' in name:
            params.update({'temperature': 0.6, 'top_p': 0.9})
        if 'qwen' in name:
            params.update({'temperature': 0.7, 'top_p': 0.9})
        if 'falcon' in name:
            params.update({'temperature': 0.75, 'top_p': 0.9, 'top_k': 10})

        params['pad_token_id'] = self.tokenizer.pad_token_id
        params['eos_token_id'] = self.eos_token_id
        return params

    def _prepare_inputs_for_generate(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        clean: Dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                clean[k] = v
                continue

            try:
                if isinstance(v, (list, tuple, np.ndarray)):
                    tv = torch.as_tensor(v)
                    clean[k] = tv
                    continue
            except Exception:

                pass
        return clean

    def _safe_generate(self, inputs: Dict[str, torch.Tensor], gen_kwargs: Dict[str, Any]):
        try:

            inputs = self._prepare_inputs_for_generate(inputs)

            if self._using_device_map and torch.cuda.is_available():
                moved: Dict[str, torch.Tensor] = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        moved[k] = v.to('cuda') if v.device.type != 'cuda' else v
                inputs = moved
            else:
                inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            return self.model.generate(**inputs, **gen_kwargs)
        except RuntimeError as e:
            if self._quantized_8bit:
                raise
            if ('CUDA error' in str(e) or 'device-side assert' in str(e)) and self.device.type == 'cuda':
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device).to(torch.float32)
                inputs_cpu = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                return self.model.generate(**inputs_cpu, **gen_kwargs)
            raise

    def _build_prompt(self, text: str) -> str:
        return f"""Your task is to create a comprehensive yet concise summary of the following text. 

Text to summarize:
{text}

Summary:"""

    def _build_chat_prompt(self, text: str) -> str:
        if self.has_chat_template and not self.is_encoder_decoder:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Please create a comprehensive yet concise summary of the following text.

Text to summarize:
{text}"""}
            ]
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:

                return self._build_prompt(text)
        return self._build_prompt(text)

    def _maybe_t5_prefix(self, text: str) -> str:
        name = (self.model_name or '').lower()
        mtype = (self.model_type or '').lower()
        if ('t5' in name) or ('t5' in mtype):
            return f"summarize: {text}"
        return text

    def _soft_trim_summary(self, summary: str) -> str:
        txt = (summary or '').strip()
        words = [w for w in txt.split() if w.strip()]
        if not words:
            return txt
        limit = int(round(self.target_max_words * (1.0 + float(getattr(self, 'overshoot_ratio', 0.15)))))
        if len(words) <= limit:
            return txt
        # Try to trim by sentences
        try:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(txt)
            acc, wc = [], 0
            for s in sents:
                w = len([t for t in s.split() if t.strip()])
                if wc + w <= limit:
                    acc.append(s.strip())
                    wc += w
                else:
                    break
            if acc:
                soft = ' '.join(acc).strip()
                if soft:
                    return soft
        except Exception:
            pass
        return ' '.join(words[:limit])

    @torch.inference_mode()
    def generate_summary_snippets_batch(self, snippets: List[str], batch_size: int = 8) -> List[str]:
        summaries = []
        effective_bs = min(batch_size, 10) if self.device.type == 'cpu' else batch_size
        for i in range(0, len(snippets), effective_bs):
            batch = snippets[i:i + effective_bs]
            if self.is_encoder_decoder:
                # Prefix for T5
                batch_enc = [self._maybe_t5_prefix(t) for t in batch]
                encoded = self.tokenizer(batch_enc, return_tensors='pt', truncation=True, padding=True,
                                         max_length=self.max_source_length)
                gen_kwargs = self._seq2seq_params('snippet')
                outputs = self._safe_generate(encoded, gen_kwargs)
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
                summaries.extend([d.strip() for d in decoded])
            else:
                prompts = [self._build_chat_prompt(t) for t in batch]
                encoded = self.tokenizer(prompts, return_tensors='pt', truncation=True, padding=True,
                                         max_length=self.max_source_length)
                gen_kwargs = self._gpt_params('snippet')
                outputs = self._safe_generate(encoded, gen_kwargs)
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)

                eff_ids = encoded.get('input_ids')
                try:
                    if isinstance(eff_ids, torch.Tensor):
                        effective_prompts = self.tokenizer.batch_decode(
                            eff_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                    elif isinstance(eff_ids, (list, tuple)) and len(eff_ids) > 0:
                        effective_prompts = self.tokenizer.batch_decode(
                            eff_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                    else:

                        effective_prompts = prompts
                except Exception:
                    effective_prompts = prompts
                for eff_prompt, text_out in zip(effective_prompts, decoded):
                    txt = text_out.strip()
                    if txt.lower().startswith(eff_prompt.strip().lower()):
                        txt = txt[len(eff_prompt):].strip()
                    summaries.append(txt)
        return summaries

    @torch.inference_mode()
    def generate_summary_of_all(self, text: str) -> str:
        if self.is_encoder_decoder:
            # Prefix for T5
            text_in = self._maybe_t5_prefix(text)
            data = self.tokenizer(text_in, return_tensors='pt', truncation=True, max_length=self.max_source_length,
                                  padding=False)
            gen_kwargs = self._seq2seq_params('final')
            try:
                output_ids = self._safe_generate(data, gen_kwargs)[0]
                summary = self.tokenizer.decode(output_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True).strip()
            except Exception:
                summary = ''
            if not summary:

                try:
                    fallback_kwargs = {**gen_kwargs, 'do_sample': True, 'top_p': 0.92, 'temperature': 0.9,
                                       'length_penalty': max(1.0, float(gen_kwargs.get('length_penalty', 1.0)))}
                    output_ids = self._safe_generate(data, fallback_kwargs)[0]
                    summary = self.tokenizer.decode(output_ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True).strip()
                except Exception:
                    summary = ''
        else:
            prompt = self._build_chat_prompt(text)
            data = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_source_length,
                                  padding=False)
            gen_kwargs = self._gpt_params('final')
            try:
                output_ids = self._safe_generate(data, gen_kwargs)[0]
                decoded_full = self.tokenizer.decode(output_ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True).strip()

                ids0 = None
                _ids = data.get('input_ids')
                try:
                    if isinstance(_ids, torch.Tensor):
                        ids0 = _ids[0]
                    elif isinstance(_ids, (list, tuple)) and len(_ids) > 0:
                        ids0 = _ids[0]
                except Exception:
                    ids0 = None
                if ids0 is not None:
                    eff_prompt = self.tokenizer.decode(ids0, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True).strip()
                else:
                    eff_prompt = prompt.strip()
                summary = decoded_full
                if summary.lower().startswith(eff_prompt.lower()):
                    summary = summary[len(eff_prompt):].strip()
            except Exception:
                summary = ''
            if not summary:
                try:
                    fallback_kwargs = {**gen_kwargs,
                                       'temperature': min(1.0, gen_kwargs.get('temperature', 0.75) + 0.15),
                                       'top_p': max(0.85, gen_kwargs.get('top_p', 0.9) - 0.05)}
                    output_ids = self._safe_generate(data, fallback_kwargs)[0]
                    decoded_full = self.tokenizer.decode(output_ids, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True).strip()
                    _ids = data.get('input_ids')
                    ids0 = None
                    try:
                        if isinstance(_ids, torch.Tensor):
                            ids0 = _ids[0]
                        elif isinstance(_ids, (list, tuple)) and len(_ids) > 0:
                            ids0 = _ids[0]
                    except Exception:
                        ids0 = None
                    if ids0 is not None:
                        eff_prompt = self.tokenizer.decode(ids0, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=True).strip()
                    else:
                        eff_prompt = prompt.strip()
                    summary = decoded_full
                    if summary.lower().startswith(eff_prompt.lower()):
                        summary = summary[len(eff_prompt):].strip()
                except Exception:
                    summary = ''
        if not summary:
            summary = text[:400].strip()
        # Gently adjust to the target word range
        summary = self._soft_trim_summary(summary)
        return summary


def compute_bleu_per_batch(generated: List[str], references: List[str]) -> float:
    """
    Compute average BLEU-4 score across all generated summaries.

    Args:
        generated: List of generated summary texts
        references: List of reference summary texts

    Returns:
        Average BLEU-4 score with smoothing
    """
    smoothing = SmoothingFunction().method1
    scores = []
    for gen, ref in zip(generated, references):
        gen_tokens = word_tokenize(gen.lower())
        ref_tokens = [word_tokenize(ref.lower())]
        scores.append(
            sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing, weights=(0.25, 0.25, 0.25, 0.25)))
    return float(np.mean(scores)) if scores else 0.0


def compute_rouge_per_batch(generated: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    try:
        _pkg, _sub = 'rouge_score', 'rouge_scorer'
        module_name = _pkg + '.' + _sub
        rouge_mod = importlib.import_module(module_name)
        RougeScorer = getattr(rouge_mod, 'RougeScorer', None)
    except Exception:
        RougeScorer = None
    if RougeScorer is None:
        return {
            'rouge-1': {'p': 0.0, 'r': 0.0, 'f': 0.0},
            'rouge-2': {'p': 0.0, 'r': 0.0, 'f': 0.0},
            'rouge-l': {'p': 0.0, 'r': 0.0, 'f': 0.0},
        }
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for gen, ref in zip(generated, references):
        scores = scorer.score(ref, gen)
        all_scores['rouge1'].append(scores['rouge1'])
        all_scores['rouge2'].append(scores['rouge2'])
        all_scores['rougeL'].append(scores['rougeL'])
    avg_scores: Dict[str, Dict[str, float]] = {}
    for k, v in all_scores.items():
        key = 'rouge-1' if k == 'rouge1' else ('rouge-2' if k == 'rouge2' else 'rouge-l')
        if v:
            avg_scores[key] = {
                'p': sum(s.precision for s in v) / len(v),
                'r': sum(s.recall for s in v) / len(v),
                'f': sum(s.fmeasure for s in v) / len(v)
            }
        else:
            avg_scores[key] = {'p': 0.0, 'r': 0.0, 'f': 0.0}
    return avg_scores


def compute_bertscore(generated: List[str], references: List[str], lang: str = 'en', model_type: str = None) -> Dict[
    str, float]:
    if _bert_score is None:
        raise ImportError("bert_score is not installed")
    if model_type:
        P, R, F1 = _bert_score(generated, references, lang=lang, model_type=model_type, verbose=True, device='cpu')
    else:
        P, R, F1 = _bert_score(generated, references, lang=lang, verbose=True, device='cpu')
    avgs = {
        'avg_precision': float(np.mean(P.numpy())) if len(P) else 0.0,
        'avg_recall': float(np.mean(R.numpy())) if len(R) else 0.0,
        'avg_f1': float(np.mean(F1.numpy())) if len(F1) else 0.0
    }
    return avgs


def _sanitize_filename(name: str) -> str:
    return ''.join(ch if (ch.isalnum() or ch in ['.', '-', '_']) else '_' for ch in (name or 'model'))


def _count_words(text: str) -> int:
    try:
        return len([w for w in (text or '').split() if w.strip()])
    except Exception:
        return 0


def evaluate(raw_path: str, ref_path: str, output_csv: str, model_name: str = 'facebook/bart-large-cnn',
             model_local_dir: str = None, batch_size_group: int = 10, gen_batch_size: int = 16, max_batches: int = None,
             bertscore_model_type: str = None, json_out_dir: str = None, skip_bertscore: bool = False,
             gpt_concat_first: bool = True, min_words: int = 100, max_words: int = 140) -> None:
    _ensure_nltk_data()
    try:
        flat_snippets = load_raw_snippets(raw_path)
        ref_batches = load_reference_batches(ref_path)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    text_batches = group_snippets(flat_snippets, batch_size_group)
    n_batches = min(len(text_batches), len(ref_batches))
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)
    if n_batches == 0:
        print("No batches to process!", flush=True)
        return
    print(f"[Data] Batches to process: {n_batches} (group={batch_size_group})", flush=True)
    text_batches = text_batches[:n_batches]
    ref_summaries = [b.get('combined_summary', '') for b in ref_batches[:n_batches]]
    try:
        summarizer = Summarizer(model_local_dir=model_local_dir, model_name=model_name,
                                target_min_words=min_words, target_max_words=max_words)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    per_batch_records: List[Dict[str, Any]] = []

    generated_summaries, per_batch_time = [], []
    t0_total = time.time()
    for i, (snippets, ref_sum) in enumerate(zip(text_batches, ref_summaries)):
        non_empty_snippets = [s for s in snippets if s and s.strip()]
        print(f"[Progress] Starting batch {i + 1}/{n_batches} — snippets: {len(non_empty_snippets)}", flush=True)
        if not non_empty_snippets:
            msg = "Empty summarization - no content"
            generated_summaries.append(msg)
            per_batch_time.append(0.0)
            per_batch_records.append({
                'batch_index': i,
                'snippets_count': 0,
                'mode': 'two_pass' if summarizer.is_encoder_decoder else 'single_pass',
                'generated_snippet_summaries': [],
                'generated_final_summary': msg,
                'reference_summary': ref_sum,
                'time_sec': 0.0,
                'words_for_final_summary': 0,
                'error': None
            })
            continue
        t0 = time.time()
        try:
            if summarizer.is_encoder_decoder:
                snippet_summaries = summarizer.generate_summary_snippets_batch(non_empty_snippets,
                                                                               batch_size=gen_batch_size)
                combined = ' '.join(snippet_summaries)
                words_for_final = _count_words(combined)
                print(f"[Progress] Batch {i + 1}: words for final summarization: {words_for_final}", flush=True)
                final_summary = summarizer.generate_summary_of_all(combined)
                mode = 'two_pass'
            else:

                snippet_summaries = []
                if gpt_concat_first:
                    combined = ' '.join(non_empty_snippets)
                    mode = 'single_pass'
                else:

                    combined_raw = ' '.join(non_empty_snippets)
                    prompt_candidate = summarizer._build_chat_prompt(combined_raw)
                    enc_no_trunc = summarizer.tokenizer(prompt_candidate, return_tensors='pt', truncation=False)

                    needs_two_pass = False
                    try:
                        input_ids = enc_no_trunc.get('input_ids')
                        if isinstance(input_ids, torch.Tensor):
                            needs_two_pass = input_ids.shape[-1] >= (summarizer.max_source_length - 8)
                        elif input_ids is not None:

                            first = input_ids[0] if len(input_ids) > 0 else []
                            needs_two_pass = len(first) >= (summarizer.max_source_length - 8)
                        else:

                            needs_two_pass = len(prompt_candidate) > (summarizer.max_source_length * 4)
                    except Exception:
                        needs_two_pass = True
                    if needs_two_pass:
                        snippet_summaries = summarizer.generate_summary_snippets_batch(non_empty_snippets,
                                                                                       batch_size=gen_batch_size)
                        combined = ' '.join(snippet_summaries)
                        mode = 'two_pass'
                    else:
                        combined = combined_raw
                        mode = 'single_pass'
                words_for_final = _count_words(combined)
                print(f"[Progress] Batch {i + 1}: words for final summarization: {words_for_final}", flush=True)
                final_summary = summarizer.generate_summary_of_all(combined)
            dt = time.time() - t0
            print(f"[Progress] Batch {i + 1}/{n_batches} done in {dt:.2f} sec", flush=True)
            generated_summaries.append(final_summary)
            per_batch_time.append(dt)
            per_batch_records.append({
                'batch_index': i,
                'snippets_count': len(non_empty_snippets),
                'mode': mode,
                'generated_snippet_summaries': snippet_summaries,
                'generated_final_summary': final_summary,
                'reference_summary': ref_sum,
                'time_sec': round(dt, 4),
                'words_for_final_summary': int(words_for_final),
                'error': None
            })
        except Exception as e:
            print(f"[Error] Batch {i + 1}/{n_batches}: {e}", flush=True)
            err = f"Generation error: {str(e)}"

            fallback_combined = ' '.join(non_empty_snippets)
            words_for_final = _count_words(fallback_combined)

            print(f"[Progress] Batch {i + 1}: words for final summarization (estimate): {words_for_final}", flush=True)
            generated_summaries.append(err)
            per_batch_time.append(0.0)
            per_batch_records.append({
                'batch_index': i,
                'snippets_count': len(non_empty_snippets),
                'mode': 'two_pass' if summarizer.is_encoder_decoder else 'single_pass',
                'generated_snippet_summaries': [],
                'generated_final_summary': err,
                'reference_summary': ref_sum,
                'time_sec': 0.0,
                'words_for_final_summary': int(words_for_final),
                'error': str(e)
            })

    total_time = time.time() - t0_total
    print(f"[Summary] Total time: {total_time:.2f} sec (~{(total_time / max(1, n_batches)):.2f} sec/batch)", flush=True)
    rouge_avg = compute_rouge_per_batch(generated_summaries, ref_summaries)
    bleu_avg = compute_bleu_per_batch(generated_summaries, ref_summaries)
    bert_failed = False
    if skip_bertscore:
        bert_failed, bert_avg = True, {'avg_precision': 0.0, 'avg_recall': 0.0, 'avg_f1': 0.0}
    else:
        try:
            bs_model = bertscore_model_type or 'allenai/scibert_scivocab_uncased'
            bert_avg = compute_bertscore(generated_summaries, ref_summaries, lang='en', model_type=bs_model)
        except Exception:
            bert_failed, bert_avg = True, {'avg_precision': 0.0, 'avg_recall': 0.0, 'avg_f1': 0.0}
    avg_time_per_batch = float(np.mean(per_batch_time)) if per_batch_time else 0.0
    total_snippets = sum(len(batch) for batch in text_batches)
    snippets_per_second = total_snippets / total_time if total_time > 0 else 0.0

    fieldnames = ['model_name', 'total_batches', 'total_snippets', 'total_time_sec', 'avg_time_per_batch',
                  'snippets_per_second', 'rouge1_f1', 'rouge2_f1', 'rougel_f1', 'bleu4_score', 'bertscore_f1']
    file_exists = os.path.exists(output_csv)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'a' if file_exists else 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'model_name': model_name,
            'total_batches': n_batches,
            'total_snippets': total_snippets,
            'total_time_sec': round(total_time, 2),
            'avg_time_per_batch': round(avg_time_per_batch, 2),
            'snippets_per_second': round(snippets_per_second, 2),
            'rouge1_f1': round(rouge_avg['rouge-1']['f'], 4),
            'rouge2_f1': round(rouge_avg['rouge-2']['f'], 4),
            'rougel_f1': round(rouge_avg['rouge-l']['f'], 4),
            'bleu4_score': round(bleu_avg, 4),
            'bertscore_f1': round(0.0 if bert_failed else bert_avg['avg_f1'], 4)
        })

    base_dir = json_out_dir
    if not base_dir:
        base_dir = os.path.join(os.path.dirname(output_csv) or '.', 'summaries_json')
    os.makedirs(base_dir, exist_ok=True)
    ts = int(time.time())
    json_filename = f"{_sanitize_filename(model_name)}_{ts}.json"
    json_path = os.path.join(base_dir, json_filename)

    result_json: Dict[str, Any] = {
        'meta': {
            'model_name': model_name,
            'model_local_dir': model_local_dir,
            'device': getattr(summarizer, 'device', torch.device('cpu')).type,
            'is_encoder_decoder': getattr(summarizer, 'is_encoder_decoder', False),
            'params': {
                'batch_size_group': batch_size_group,
                'gen_batch_size': gen_batch_size,
                'max_batches': max_batches,
                'bertscore_model_type': bertscore_model_type,
                'skip_bertscore': skip_bertscore,
                'gpt_concat_first': gpt_concat_first,
                'target_min_words': min_words,
                'target_max_words': max_words,
            },
            'paths': {
                'raw_path': raw_path,
                'ref_path': ref_path,
                'csv_path': output_csv,
                'json_dir': base_dir,
                'json_file': json_path,
            },
        },
        'totals': {
            'total_batches': n_batches,
            'total_snippets': total_snippets,
            'total_time_sec': round(total_time, 4),
            'avg_time_per_batch': round(avg_time_per_batch, 4),
            'snippets_per_second': round(snippets_per_second, 4),
        },
        'metrics': {
            'rouge_avg': rouge_avg,
            'bleu4': bleu_avg,
            'bertscore_avg': (None if bert_failed else bert_avg),
            'bertscore_failed': bert_failed,
        },
        'per_batch': per_batch_records,
    }

    try:
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(result_json, jf, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save JSON results: {e}")

    print(f"\n=== Summary for model {model_name} ===")
    print(f"Processed batches: {n_batches}")
    print(f"Total snippets: {total_snippets}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Average time per batch: {avg_time_per_batch:.2f} sec")
    print(f"Snippets/sec: {snippets_per_second:.2f}")
    print(f"ROUGE-1 F1: {rouge_avg['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {rouge_avg['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {rouge_avg['rouge-l']['f']:.4f}")
    print(f"BLEU-4: {bleu_avg:.4f}")
    if not skip_bertscore and not bert_failed:
        print(f"BERTScore F1: {bert_avg['avg_f1']:.4f}")
    elif skip_bertscore:
        print("BERTScore skipped by flag.")
    else:
        print("BERTScore calculation failed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script for summarization models")
    parser.add_argument('--raw', required=True, help="Path to the file with raw snippets (JSON)")
    parser.add_argument('--ref', required=True, help="Path to the file with reference summaries (JSON)")
    parser.add_argument('--out', required=True, help="Path to save CSV with results")
    parser.add_argument('--model', default='facebook/bart-large-cnn', help="Model name or path to local folder")
    parser.add_argument('--local_dir', default=None, help="Path to local model (if any)")
    parser.add_argument('--batch_size_group', type=int, default=10, help="Group size of snippets")
    parser.add_argument('--gen_batch_size', type=int, default=16, help="Batch size for generation")
    parser.add_argument('--max_batches', type=int, default=None, help="Maximum number of batches to process")
    parser.add_argument('--bertscore_model', default=None, help="Model for BERTScore (e.g., 'roberta-large')")
    parser.add_argument('--json_out_dir', default=None, help="Folder to save JSON results for each model")
    parser.add_argument('--skip_bertscore', action='store_true', help="Skip BERTScore for faster execution")

    parser.add_argument('--no_gpt_concat_first', dest='gpt_concat_first', action='store_false',
                        help='Disable initial concatenation of snippets for GPT-like models')
    parser.set_defaults(gpt_concat_first=True)
    # New parameters for controlling summary length
    parser.add_argument('--min_words', type=int, default=100, help='Minimum summary length (in words)')
    parser.add_argument('--max_words', type=int, default=140, help='Maximum summary length (in words)')
    args = parser.parse_args()

    evaluate(
        raw_path=args.raw,
        ref_path=args.ref,
        output_csv=args.out,
        model_name=args.model,
        model_local_dir=args.local_dir,
        batch_size_group=args.batch_size_group,
        gen_batch_size=args.gen_batch_size,
        max_batches=args.max_batches,
        bertscore_model_type=args.bertscore_model,
        json_out_dir=args.json_out_dir,
        skip_bertscore=args.skip_bertscore,
        gpt_concat_first=args.gpt_concat_first,
        min_words=args.min_words,
        max_words=args.max_words
    )
