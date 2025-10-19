from collections import Counter
from tqdm import tqdm
from isanlp import PipelineCommon
from isanlp.simple_text_preprocessor import SimpleTextPreprocessor
from isanlp.processor_razdel import ProcessorRazdel
from isanlp.ru.processor_mystem import ProcessorMystem
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd
from isanlp.processor_udpipe import ProcessorUDPipe
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS

MODEL = "/home/keen/tests/english-ewt-ud-2.5-191206.udpipe"

class SyntPhrase:
    def __init__(self, **kwargs):
        self.ppl = PipelineCommon([
          (SimpleTextPreprocessor(), ['text'],
           {'text': 'text'}),
          (ProcessorRazdel(), ['text'],
           {'tokens': 'tokens',
            'sentences': 'sentences'}),
          (ProcessorUDPipe(model_path = MODEL), ['tokens', 'sentences'],
           {'lemma': 'lemma',
            'postag': 'postag',
            'syntax_dep_tree': 'syntax_dep_tree'}),
        ])

    def find_phrases(self,tokenized_corpus):

        def gen_all_pairs(tree):
            pairs = []
            for i,t in enumerate(tree):
                if t.parent != -1:
                    pairs.append((t.parent,i))
            return pairs        
        
        phrases = []
        MAIN = {'NOUN','VERB'}
        AUX = {'NOUN','ADJ'}
        for d in tqdm(tokenized_corpus):
            d = " ".join(d)
            p = self.ppl(d)
            for i,sent in enumerate(p['syntax_dep_tree']):
                pairs = gen_all_pairs(sent)    
                start = p['sentences'][i].begin
                for pp in pairs:
                    if p['postag'][i][pp[0]] in MAIN and p['postag'][i][pp[1]] in AUX:
                        if pp[0] < pp[1]: 
                            phrases.append(p['lemma'][i][pp[0]].lower() + " " + p['lemma'][i][pp[1]].lower())
                        else:    
                            phrases.append(p['lemma'][i][pp[1]].lower() + " " + p['lemma'][i][pp[0]].lower())
                            
        phrases_count = Counter(phrases)
        phrases = {k for k,v in phrases_count.items() if v > 4}
        return phrases


class MyPhraseATT:
    def __init__(self, **kwargs):
        self.ppl = PipelineCommon([
          (SimpleTextPreprocessor(), ['text'],
           {'text': 'text'}),
          (ProcessorRazdel(), ['text'],
           {'tokens': 'tokens',
            'sentences': 'sentences'}),
          (ProcessorUDPipe(model_path = MODEL), ['tokens', 'sentences'],
           {'lemma': 'lemma',
            'postag': 'postag',
            'syntax_dep_tree': 'syntax_dep_tree'}),
        ])

        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.trans_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', output_attentions=True)

    def find_phrases(self,tokenized_corpus):
        phrases = []
        for d in tqdm(tokenized_corpus):    
            d = " ".join(d)
            phrases += self.gen_phrases(d)
        phrases_count = Counter(phrases)
        phrases = {k for k,v in phrases_count.items() if v > 4}
        return phrases            

    def gen_phrases(self,text):
        res = []
        ling = self.ppl(text)    
        tok_sents = []
        tok_lemmas = []    
        for i,s in enumerate(ling["sentences"]):
            tok_sents.append([t.text for t in ling['tokens'][s.begin:s.end]])
            tok_lemmas.append([t for t in ling['lemma'][i]])    
    
        
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        def tokenize_and_align(tokenizer, words):
            words_to_tok = []
            words_to_tok.append(['[CLS]'])
            for word in words:
                words_to_tok.append(tokenizer.tokenize(word))
            words_to_tok.append(['[SEP]'])
            return words_to_tok
        
        def sent_to_word_attention(text):
            tok_example = self.tokenizer.tokenize(text, is_split_into_words=True, add_special_tokens=True)
            sent_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tok_example)]).long()
            input_mask = torch.tensor([[1] * len(sent_ids)])
            with torch.no_grad():        
                ws_outputs, sents, attns = trans_model(sent_ids, input_mask, return_dict=False,  output_attentions=True)
                        
            bpe_toks = tokenize_and_align(tokenizer, text)
            i = 0
            word_to_tokens = []
            for word in bpe_toks:
                tokens = []
                for _ in word:
                    tokens.append(i)
                    i += 1
                word_to_tokens.append(tokens)
        
            not_word_starts = []
            for word in word_to_tokens:
                not_word_starts += word[1:]
            
            wwatt_matrices = []
            
            for i in range(len(attns)):
                word_word_attention = attns[i][0].numpy()
                for word in word_to_tokens:
                    word_word_attention[:, :, word[0]] = word_word_attention[:, :, word].sum(axis=-1)
                word_word_attention = np.delete(word_word_attention, not_word_starts, -1)
        
                for word in word_to_tokens:
                    word_word_attention[:, word[0], :] = word_word_attention[:, word, :].mean(axis=1)
                word_word_attention = np.delete(word_word_attention, not_word_starts, 1)
                
                word_word_attention = word_word_attention[:,1:-1, 1:-1]   #удаляем внимание на SEP и CLS и обратно
            
                wwatt_matrices.append(word_word_attention)
                
                
            return np.abs(np.vstack(wwatt_matrices)).sum(axis=0)
        
        for k in range(len(tok_sents)):
            word_attentions = sent_to_word_attention(tok_sents[k])
            threshold = word_attentions.mean() + 0.01
            
            pairs = np.nonzero(word_attentions > threshold)
            for i in range(pairs[0].shape[0]):
                if pairs[0][i] != pairs[1][i]:
                    s1 = tok_lemmas[k][pairs[0][i]].lower()
                    s2 = tok_lemmas[k][pairs[1][i]].lower()
                    if s1 not in ENGLISH_CONNECTOR_WORDS and s2 not in ENGLISH_CONNECTOR_WORDS and len(s1) > 1 and len(s2) > 1:
                        if pairs[1][i] > pairs[0][i]:
                            res.append(s1 +" "+ s2)
                        else:
                            res.append(s2 +" "+ s1)
        return res            
