# Description

This repository contains source files, notebook, and some datasets for "Dmitry Devyatkin,Ilya Sochenkov, Dmitry Popov, Denis Zubarev, Anastasia Ryzhova, Fyodor Abanin, and Oleg Grigoriev "Identifying new promising research directions with open peer reviews" paper.

# Repository structure
* datasets - includes:
  -  ICLR review dataset (original https://github.com/Seafoodair/Openreview/tree/master),
  -  summarization dataset to train topic summarization model,
  -  arXiv.org dataset (placeholder only, please, upload it youself from arXiv.org),
  -  preprocessed - preprocessed ICLR data with added Arxiv preprint fulltexts, so you do not need to run preprocessing part from notebooks.
* notebooks - includes training review text-to-score experiments, ICLR2017-2019 topic extraction and evaluation experiments, review segmentation experiments (filter out all non-important review fragments).
* top2vec - slightly modified top2vec that supports batch vector quering and has less RAM consumption.
* pipeline.py - demo pipeline that finds, evaluate and summarizes novel topics in ICLR2017-2019
* test_on_standard_ds.py - experiments with different phrase builders for Top2Vec
