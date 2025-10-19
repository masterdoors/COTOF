import torch
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import tarfile
import pandas as pd
import os
import ngram
import json
import logging
device = "cpu"
logger = logging.getLogger(__name__)
accuracy = evaluate.load("accuracy")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
#    return accuracy.compute(predictions=predictions, references=labels)
    return f1.compute(predictions=predictions, references=labels, average='macro')


def grab_arxiv_txt(titles):
    titles = {t[:t.find("|")].lower().strip() for t in titles}
    G = ngram.NGram(titles)
    #directory of the arcieve with arXiv.org files. *.json files should contain methadata, *.txt files
    #should contain full texts. *.json files should provide the preprint's title in the field
    #with name "202"
    arx_dir = "/data2/arxiv"
    # archieve name
    arh = "2030_arxiv_json_dump.tar.gz"

    to_extract = {}
    datas = {}
    cntr202 = 0
    try:
        with tarfile.open(os.path.join(arx_dir, arh), 'r:gz') as tar:
            for member in tqdm(tar):
                if member.isfile():
                    if member.name.endswith(".json"):
                        file_obj = tar.extractfile(member)
                        meta_str = file_obj.read().decode("utf-8")
                        meta = json.loads(meta_str)["meta"]
                        name = member.name[member.name.rfind("/")+1:member.name.rfind(".json")]
                        if "202" in meta:
                            cntr202 += 1
                            title_res = G.search(meta["202"].lower().strip(), threshold=0.7)
                            if len(title_res) > 0:
                                to_extract[name] = title_res[0][0]    
        logger.info("To extract from arXiv: ", str(len(to_extract)), str(cntr202))                        
        with tarfile.open(os.path.join(arx_dir, arh), 'r:gz') as tar:
            for member in tqdm(tar):
                if member.isfile():
                    if member.name.endswith(".txt"):     
                        name = member.name[member.name.rfind("/")+1:member.name.rfind(".txt")]
                        if name in to_extract:
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                content = file_obj.read().decode("utf-8")    
                                datas[to_extract[name]] = content
                                        
    except Exception as e:
        logger.error(str(e))
    return datas
        
class Preprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

        model = AutoModelForSequenceClassification.from_pretrained(
            "Ryzhik22/rev-classif-twolangs", num_labels=4)        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors = 'pt')
        
        training_args = TrainingArguments(
            output_dir="segm_model/my_model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=8,
            weight_decay=0.01,
            no_cuda=True
        )
        
        self.trainer = Trainer(
            model=model,
            args=training_args,
            #train_dataset=tokenized_ds["train"],
            #eval_dataset=tokenized_ds["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    def remove_trash(self, txt_):
        txts = []
        texts = sent_tokenize(txt_)
        for txt in texts:
            tokens = self.tokenizer(txt, truncation=True)
            predictions = self.trainer.predict([tokens])
            lbl = np.argmax(predictions.predictions, axis=-1)
            if lbl != 1:
                txts.append(txt)
        return " ".join(txts)        
                
    def preprocess_reviews(self,fname): 
        df = pd.read_excel(fname)
        dat = []
        j = 0
        titles = df["title"].to_list()
        content = grab_arxiv_txt(titles)
        for i, row in df.iterrows():
            text = content.get(row["title"][:row["title"].find("|")].lower().strip(),"")
            text = text.replace("Published as a conference paper at ICLR "," ")    
            text = text.replace("Under review as a conference paper at ICLR "," ").replace("\n"," ")    
            if len(text) < 1:
                text = row["title"][:row["title"].find("|")]
            dat.append([i,row['rate'],row["title"][:row["title"].find("|")],row["title"][:row["title"].find("|")] + " " + self.remove_trash(row["review"]),text])
            # if j > 100:
            #     break
            j += 1
        return pd.DataFrame(dat,columns=("id","score","title","review","text"))
    
    def getTexts(self, cell):
        res = []
        for i in range(len(cell)):
            c = cell[i]['content']
            res.append(c[c.find("Summary") + 7:])
        return res    
 
    