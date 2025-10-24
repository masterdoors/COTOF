#!/usr/bin/env python
# coding: utf-8

# # Get new topics
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from top2vec import Top2Vec
import pandas as pd
from preprocess import *
from annotation import Annotator
from score import Scorer
from scipy.stats import spearmanr
import logging

REV_PATH = "datasets/reviews/data/ICLR data"

def SpearmanSim(topic_words1,topic_words2):
    all_words = set(topic_words1.tolist() + topic_words2.tolist())
    r1 = []
    r2 = []
    for w in all_words:
        r1_ = np.where(topic_words1 == w)[0]
        if len(r1_) == 0:
            r1_ = 100
        else:
            r1_ = r1_.sum()
        r1.append(r1_)
        r2_ = np.where(topic_words2 == w)[0]
        if len(r2_) == 0:
            r2_ = 100
        else:
            r2_ = r2_.sum()
        r2.append(r2_)  
    return spearmanr(r1,r2)

if __name__ == "__main__":
    trh = 0.5    
    prev_words = None
    prev_scores = None
    new_topics = {}
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get a logger instance
    logger = logging.getLogger(__name__)
    preproc = Preprocessor()
    logger.info("Load a sentence transformer")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Load a review scoring classifier")
    scorer = Scorer()
    logger.info("Load a summarization model")
    annotator = Annotator()
    for year in range(2017,2021):
        logger.info("year: " + str(year))
        rews = preproc.preprocess_reviews(os.path.join(REV_PATH,"tp_" + str(year) + "conference.xlsx"))
        topic_model = Top2Vec(documents=rews["text"].to_list(),
                            ngram_vocab=True,
                            contextual_top2vec=True,speed="deep-learn",embedding_model="all-MiniLM-L6-v2",document_ids=rews["id"].to_list(),index_topics=False)
    
        topic_words, word_scores, topic_nums = topic_model.get_topics(topic_model.get_num_topics()) 
        topic_scores = scorer.getTopicScores(rews, topic_model)
        
        if prev_words is not None:
            for w,s,n in zip(topic_words, word_scores,topic_nums):
                found = False
                for old_w,old_s in zip(prev_words, prev_scores):
                    dist, p_value = SpearmanSim(w,old_w)
                    if dist > 0 and p_values < 0.05:
                        found = True   
                if not found:
                    if n in topic_scores:
                        if np.asarray(topic_scores[n]).mean() > 1.0:
                            logger.info("New topic:")
                            logger.info(annotator.get_annotation(w,80,topic_model,rews['title'].to_list(),rews['text'].to_list()))
                        
        prev_words = topic_words
        prev_scores = word_scores



