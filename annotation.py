import argparse
import logging
import os
import re
import time
from typing import Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration,BartTokenizer, BartForConditionalGeneration

from isanlp import PipelineCommon
from isanlp.simple_text_preprocessor import SimpleTextPreprocessor
from isanlp.processor_razdel import ProcessorRazdel
from isanlp.ru.processor_mystem import ProcessorMystem
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd

model_config = {
        'hf_name': "philschmid/bart-large-cnn-samsum",
        'prompt_snippet': "summarize: {text}",
        'prompt_annotation': "summarize: {text}",
        'prompt_final': "summarize: {text}",
        'min_tokens': {'snippet': 48, 'annotation': 130, 'final': 130},
        'max_tokens': {'snippet': 64, 'annotation': 300, 'final': 300},
        'generate_kwargs': {
            'do_sample': True,
            'temperature': 0.2,
            'top_p': 0.3,
            'top_k': 3,
            'repetition_penalty': 1.5,
            'no_repeat_ngram_size': 2
        },
        'tokenizer_class': BartTokenizer,
        'model_class': BartForConditionalGeneration,
        'use_device_map': False,
        'torch_dtype': None
}

BASE_MODEL_DIR = "./models"
device = "cpu" 
logger = logging.getLogger(__name__)

class Annotator:
    def __init__(self,model_config = model_config):
        self.cfg = model_config
        self.model, self.tokenizer = self.load_models()

    def load_models(self):
        hf_name = self.cfg['hf_name']
        tok_cls = self.cfg['tokenizer_class']
        mdl_cls = self.cfg['model_class']
        use_device_map = self.cfg.get('use_device_map', False)
        torch_dtype = self.cfg.get('torch_dtype', None)
        local_path = BASE_MODEL_DIR + hf_name
    
        if not os.path.isdir(local_path) or not os.listdir(local_path):
            os.makedirs(local_path, exist_ok=True)
    
            tokenizer = tok_cls.from_pretrained(hf_name)
    
            model_kwargs = {}
            if torch_dtype is not None:
                model_kwargs['torch_dtype'] = torch_dtype
    
            model = mdl_cls.from_pretrained(hf_name, **model_kwargs)
    
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
        else:
            tokenizer = tok_cls.from_pretrained(local_path)
            model_kwargs = {}
            if torch_dtype is not None:
                model_kwargs['torch_dtype'] = torch_dtype
    
            model = mdl_cls.from_pretrained(local_path, **model_kwargs)
    
    
        model.eval()
        return tokenizer, model
    
    
    def generate_summary(self,text, type):
        prompt = self.cfg['prompt_annotation'].format(text=text)
    
        data = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        data = {k: v.to(device) for k, v in data.items()}
    
        min_tokens_val = self.cfg['min_tokens'][type]
        max_tokens_val = self.cfg['max_tokens'][type]
    
        gen_kwargs = self.cfg['generate_kwargs'].copy()
        gen_kwargs['min_new_tokens'] = min_tokens_val
        gen_kwargs['max_new_tokens'] = max_tokens_val
    
    
        output_ids = self.model.generate(**data, **gen_kwargs)[0]
        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return summary.strip()
    
    def count_non_alphabetical(input_string):
        count = 0
        for char in input_string:
            if (not char.isalpha()) and char != ' ':
                count += 1
        return count
    
    def get_annotation(self,topic,top_id,model,titles,articles):
        ppl = PipelineCommon([
          (SimpleTextPreprocessor(), ['text'],
           {'text': 'text'}),
          (ProcessorRazdel(), ['text'],
           {'tokens': 'tokens',
            'sentences': 'sentences'}),
        ])
        texts = [] 
    
        for i,a in enumerate(articles):
            _,_,tscore,tn =  model.query_topics(a,1,reduced=True)
            if sum(tn) == top_id: 
                if len(texts) == 0 or a.find(texts[len(texts) - 1][0]) == -1:
                    texts.append((a,titles[i],tscore))
    
    
        texts = [(t,ti) for t,ti,s in sorted(texts,key=lambda item: item[2],reverse=True)]
        #print(len(texts))
        sentences = []
        for text,title in texts[:5]:
                sentences_ = []
                sentences_.append(title)
                parsed = ppl(text)
                added = 0 
                for i,sent in enumerate(parsed['sentences']):
                    if i > 1 and i < 7:
                        sent_toks = [t.text.lower() for t in parsed['tokens'][sent.begin:sent.end]]
                        if count_non_alphabetical(" ".join(sent_toks)) < 5:
                            sentences_.append(" ".join(sent_toks)) 
                    else:    
                        for t in topic:
                            words = t.split()
                            matches = 0
                            sent_toks = [t.text.lower() for t in parsed['tokens'][sent.begin:sent.end]]
                            for w in words:
                                if w in set(sent_toks):
                                    matches += 1
                            if matches >= len(words) and (count_non_alphabetical(" ".join(sent_toks)) < 5):
                                sentences_.append(" ".join(sent_toks))  
                                added += 1
                                break
                        if added > 10:
                            break
                sentences.append(sentences_)            
                            
        try:
            chunks = []
            for s in sentences:
                r = self.generate_summary(". ".join(s), "snippet")
                chunks.append(r)
            coherent_text = self.generate_summary(". ".join(chunks), "final") 
        except Exception as e:
            logger.error(str(e))
            return str(e)
        return coherent_text
