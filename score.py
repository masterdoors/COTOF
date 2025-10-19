import torch
import evaluate
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import DataCollatorWithPadding
import numpy as np
import torch
import evaluate
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import DataCollatorWithPadding
import numpy as np
import logging
    
from top2vec import Top2Vec

logger = logging.getLogger(__name__)

class Scorer:
    def __init__(self):
        model_name = "roberta-base"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cpu"
        
        cls_model = AutoModelForSequenceClassification.from_pretrained("Keen89/roberta-base_review_clf",num_labels=3)
        
        data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding='longest',
                max_length=512,
                pad_to_multiple_of=8,
                return_tensors='pt',
            )
        
        training_args = TrainingArguments(
            output_dir="review_clf_model_without_NEW" + model_name,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            weight_decay=0.01, 
            warmup_ratio=0.15,
            max_grad_norm=1.0,    
            push_to_hub=False,
            no_cuda=True
        )
        
        self.trainer = Trainer(
           model=cls_model,
           args=training_args,
           tokenizer=self.tokenizer,
           data_collator=data_collator,
        )
        

    def getTopicScores(self,rews, topic_model):
        def preprocess_function(examples):
            return self.tokenizer([str(e) for e in examples["text"]], truncation=True, max_length=512)
        
        df =rews
        dat = []
        dists = []
        
        top2score = {}
        for i, row in df.iterrows():
            _,_,ts,tn = topic_model.query_topics(row["text"], 1,reduced = True)
            txt = row['review']
            qtags = row["review"].count("?")
         
            if qtags > 1:
                txt = "MANY_QUESTIONS" + " " + txt   
            
            tokens = self.tokenizer(txt, truncation=True)
            predictions = self.trainer.predict([tokens])
            lbl = np.argmax(predictions.predictions, axis=-1)    
            score = 0
            if lbl == 0:
                score = 2
            if lbl == 2:
                score = 1
        
            dists.append(ts[0])
            if tn[0] not in top2score:
                top2score[tn[0]] = []
            top2score[tn[0]].append(score)
        return top2score
