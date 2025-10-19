#!/usr/bin/env python
# coding: utf-8

import warnings
    
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_20newsgroups

from phrases import SyntPhrase, MyPhraseATT
from gensim.models.phrases import Phrases
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
from transformers import DebertaModel, DebertaTokenizerFast
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from top2vec.top2vec import default_tokenizer
    
from top2vec import Top2Vec

from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.coherence_metrics import WECoherencePairwise

from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
from octis.evaluation_metrics.metrics import AbstractMetric
import numpy as np

def tokenize(dataset,word_indexes):
    all_ = []
    for d in dataset:
        res = []
        tokenized = default_tokenizer(d)
        i = 0
        while i < len(tokenized):
            skip = False
            if i < len(tokenized) - 1:
                if tokenized[i] + " " + tokenized[i + 1]  in word_indexes:
                    res.append(tokenized[i] + " " + tokenized[i + 1])
                    i += 2
                    skip = True
            if not skip:
                if tokenized[i] in word_indexes:
                    res.append(tokenized[i])
                    i += 1
                else:
                    i += 1
        all_.append(res)
    return all_

class BERTScore(AbstractMetric):
    def __init__(self, topic_model,topk=10, topd = 50):
        super().__init__()

        self.topk = topk
        self.topd = topd
        self.topic_model = topic_model
        self.tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-xlarge-mnli", add_prefix_space=True)
        self.model = AutoModel.from_pretrained("microsoft/deberta-xlarge-mnli")

    def encode_texts(self,texts, batch_size=25):

        #print(len(texts),inputs_['input_ids'].shape)
        offsets = []
        w_embeddings = []    
        max_len = min(max([len(t) for t in texts]),512)
        print("max len:",max_len)
        print("len texts:",len(texts))
        for start in range(0,len(texts),batch_size):
            inputs_ = self.tokenizer(texts[start:start+batch_size], return_tensors="pt", is_split_into_words=True,padding="max_length", truncation=True,max_length=max_len)            
            with torch.no_grad():
                outputs = self.model(**inputs_)
                for i in range(0,min(batch_size,len(texts) - start)):
                    offsets.append([inputs_.word_to_tokens(i,w_idx) for w_idx in inputs_.word_ids(i) if w_idx is not None])                
                attention_mask = inputs_["attention_mask"]
                #print(attention_mask)
                non_padded_indices = torch.where(attention_mask != 0)
                w_embedding = outputs.last_hidden_state      
                w_embeddings.append(w_embedding)
        #print(w_embeddings.shape)    
        #print(offsets)
        return zip(torch.vstack(w_embeddings).cpu().numpy(), offsets) #skip [CLS] 
    
    def encode_text(self,text):
        inputs = self.tokenizer(text, return_tensors="pt", is_split_into_words=True)

        offsets = [inputs.word_to_tokens(w_idx) for w_idx in inputs.word_ids() if w_idx is not None]

        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs["attention_mask"]
            #print(attention_mask)
            non_padded_indices = torch.where(attention_mask != 0)
            w_embeddings = outputs.last_hidden_state[non_padded_indices]      
        #print(w_embeddings.shape)    
        #print(offsets)
        return w_embeddings.cpu().numpy(), offsets #skip [CLS]    

    def score(self, model_output, documents):
        #get idfs
        vectorizer = TfidfVectorizer(tokenizer = default_tokenizer, analyzer='word')
        vectorizer.fit(documents)
        idf_values = vectorizer.idf_  
        feature_names = vectorizer.get_feature_names_out()
        word_idf_dict = dict(zip(feature_names, idf_values))        

        topics = model_output["topics"]
        top2doc = {}
        for i,d in enumerate(documents):
            _,_,ts,tn = self.topic_model.query_topics(d, 5 if self.topic_model.get_num_topics() > 5 else self.topic_model.get_num_topics(),reduced=True)
            for tss,tnn in zip(ts,tn):
                if tnn not in top2doc:
                    top2doc[tnn] = []
                top2doc[tnn].append((i,tss))    
        #get top-50 doc for each topic
        for t in top2doc:
            top2doc[t] = sorted(top2doc[t],key=lambda item: item[1],reverse=True)[:self.topd]    
            
        result = 0.0
        den_sum = 0.
        for i,topic in tqdm(enumerate(topics)):
            top_e, offs_top_e = self.encode_text(topic[0:self.topk])
            texts = []
            doc_embs = []
            for doc_id,score in top2doc[i]:
                text = default_tokenizer(documents[doc_id])
                texts.append(text)
                
            doc_embs  = self.encode_texts(texts)            
                
            for d_e, offs_d_e in doc_embs:
                sims = cosine_similarity(top_e,d_e)
                for j,word in enumerate(topic[0:self.topk]):
                    max_sim = sims[offs_top_e[j][0]:offs_top_e[j][1]].max() 
                    idf = word_idf_dict[word]
                    result += max_sim * idf
                    den_sum += idf
            for d,o in doc_embs:
                del d
            del top_e              
        return result / den_sum

class SBERTCoherencePairwise(AbstractMetric):
    def __init__(self, topk=10):
        super().__init__()

        self.topk = topk
        self.model  = SentenceTransformer('all-MiniLM-L6-v2')

    def info(self):
        return {
            "citation": citations.em_coherence_we,
            "name": "Coherence word embeddings pairwise cosine"
        }

    def score(self, model_output):
        topics = model_output["topics"]

        result = 0.0
        for topic in topics:
            E = []

            # Create matrix E (normalize word embeddings of
            # words represented as vectors in wv)
            for word in topic[0:self.topk]:
                word_embedding = self.model.encode(word)
                normalized_we = word_embedding / word_embedding.sum()
                E.append(normalized_we)
            if len(E) > 0:
                E = np.array(E)

                # Perform cosine similarity between E rows
                distances = np.sum(1 - pairwise_distances(E, metric='cosine') - np.diag(np.ones(len(E))))
                topic_coherence = distances/(self.topk*(self.topk-1))
            else:
                topic_coherence = -1

            # Update result with the computed coherence of the topic
            result += topic_coherence
        result = result/len(topics)
        return result




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase", type=str, help="Phrase generator type (GENSIM, SYNT, ATTN)")
    args = parser.parse_args()
    type_ = args.phrase

    if type_.find("SYNT") > -1:
        pcsl = SyntPhrase
    else:
        if type_.find("ATTN") > -1:
            pcsl = MyPhraseATT
        else:    
            pcsl = Phrases
            
    print(type_)
    #print("20 news group")
    #newsgroups_train = fetch_20newsgroups(subset='all')            
    #dataset = newsgroups_train['data']
    
    #phrase_gen = SyntPhrase()
    
    #topic_model = Top2Vec(documents=dataset,
    #                       ngram_vocab=True,
    #                       contextual_top2vec=True,speed="deep-learn",embedding_model="all-MiniLM-L6-v2",index_topics=False, phrase_model_cls=pcsl)
    #topic_model.hierarchical_topic_reduction(100)
    #tokenized_dataset = tokenize(dataset, topic_model.word_indexes)  
    #simple_tokenized_dataset = [default_tokenizer(d) for d in dataset]

    #C = Coherence(measure='c_npmi',texts=tokenized_dataset)
    
        
    #tm = {"topics":topic_model.get_topics(topic_model.get_num_topics(reduced = True),reduced = True)[0].tolist()}

    #print(C.score(tm))
    
    #sep_topics = []
    #for t in tm["topics"]:
    #   words = []
    #   for w in t:
    #       words += w.split()
    #   sep_topics.append(words)    
    #tm_sep = {"topics":sep_topics}
    
    #C = Coherence(measure='c_npmi',texts=simple_tokenized_dataset)
    #print(C.score(tm_sep))
    
    
    #C = Coherence(measure='c_v',texts=tokenized_dataset)
    #print(C.score(tm))
    
    #C = Coherence(measure='c_v',texts=simple_tokenized_dataset)
    #print(C.score(tm_sep))
    

    #C = WECoherencePairwise()
    #print(C.score(tm_sep))    
    
    #C = SBERTCoherencePairwise()#texts=tokenized_dataset)
    #print(C.score(tm))
    
    #C = BERTScore(topic_model=topic_model)
    #print(C.score(tm_sep,dataset))
    
    
    # Yahoo answers
    #print("Yahoo answers")
    
    # Load a single Parquet file
    #df = pd.read_parquet('train-00000-of-00001.parquet')

    #df['all'] = df['question'] + " " + df['answer'] 
    
    #dataset = df['all'].tolist()[:100000]
    
    #topic_model = Top2Vec(documents=dataset,
    #                        ngram_vocab=True,
    #                        contextual_top2vec=True,speed="deep-learn",embedding_model="all-MiniLM-L6-v2",index_topics=False,workers=1,keep_documents=True, phrase_model_cls=pcsl)
    #topic_model.hierarchical_topic_reduction(100)
    #tokenized_dataset = tokenize(dataset, topic_model.word_indexes)  
    #simple_tokenized_dataset = [default_tokenizer(d) for d in dataset]
    
    #C = Coherence(measure='c_npmi',texts=tokenized_dataset)
    
    #tm = {"topics":topic_model.get_topics(topic_model.get_num_topics(reduced = True),reduced = True)[0].tolist()}

    #print(C.score(tm))
    
    #sep_topics = []
    #for t in tm["topics"]:
    #    words = []
    #    for w in t:
    #        words += w.split()
    #    sep_topics.append(words)    
    #tm_sep = {"topics":sep_topics}
    
    #C = Coherence(measure='c_npmi',texts=simple_tokenized_dataset)
    #print(C.score(tm_sep))
    
    #C = Coherence(measure='c_v',texts=tokenized_dataset)
    #print(C.score(tm))
    
    #C = Coherence(measure='c_v',texts=simple_tokenized_dataset)
    #print(C.score(tm_sep))
    
   
    #C = WECoherencePairwise()
    
    #print(C.score(tm_sep))
    
    
    #C = SBERTCoherencePairwise()#texts=tokenized_dataset)
    #print(C.score(tm))
    
    #C = BERTScore(topic_model=topic_model)
    #print(C.score(tm_sep,dataset))
        

    # ICLR 2017
    print("ICLR 2017")
    
    df = pd.read_csv("filtered_rews.csv",sep=";")

    
    dataset = df['title'].tolist()
    
    topic_model = Top2Vec(documents=dataset,
                            ngram_vocab=True,
                            contextual_top2vec=True,speed="deep-learn",embedding_model="all-MiniLM-L6-v2",index_topics=False,workers=1,keep_documents=True, phrase_model_cls=pcsl)
    topic_model.hierarchical_topic_reduction(30)
    tokenized_dataset = tokenize(dataset, topic_model.word_indexes)  
    simple_tokenized_dataset = [default_tokenizer(d) for d in dataset]
    
    C = Coherence(measure='c_npmi',texts=tokenized_dataset)
    
    tm = {"topics":topic_model.get_topics(topic_model.get_num_topics(reduced = True),reduced = True)[0].tolist()}

    print(C.score(tm))
    
    sep_topics = []
    for t in tm["topics"]:
        words = []
        for w in t:
            words += w.split()
        sep_topics.append(words)    
    tm_sep = {"topics":sep_topics}
    
    C = Coherence(measure='c_npmi',texts=simple_tokenized_dataset)
    print(C.score(tm_sep))
    
    C = Coherence(measure='c_v',texts=tokenized_dataset)
    print(C.score(tm))

    C = Coherence(measure='c_v',texts=simple_tokenized_dataset)
    print(C.score(tm_sep))
    
   
    C = WECoherencePairwise()
    
    print(C.score(tm_sep))
    
    
    C = SBERTCoherencePairwise()#texts=tokenized_dataset)
    print(C.score(tm))
    
    C = BERTScore(topic_model=topic_model)
    print(C.score(tm_sep,dataset))

if __name__ == "__main__":
    main()

    

