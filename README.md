# Code for Paper: "W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain Question Answering"

## Code under construction... Please feel free to look around but I will refactor the code soon... 

We define passage relevance as "how likely the passage elicits correct answer from a LLM" when it comes to RAG. This project re-labels passages from MSMARCO QnA v2.1, NQ, SQuAD, and WebQ according to this relevance definition. These weak labels are then used to train a dense retriever which will finally retrieve relevant passages to help LLM generate better responses. 

<!-- arXiv link: https://arxiv.org/abs/2408.08444 -->

## Setting up: 

We use `python 3.9.18`, and `transformers 4.43.3`, `torch 2.2.0`, `ragatouille 0.0.8`. Note that we made some modifications to `sentence-transformers` and `beir`, so please `cd` into `sentence-transformers_2.2.2` and `beir` in this project and do `pip install .` do install them. <br>
`ragetouille` is used to train ColBERT, we write our own evaluation code, explained below. `sentence-transformers_2.2.2` is used to train and evaluate DPR. We use this earlier version because it is easier to make customizations, although the code is sometimes not that efficient compared to the latest versions. `beir` is used to manage data, and it is a wrapper for `sentence-transformers` for training and evaluating DPR. 

## Original datasets

<!-- https://zenodo.org/records/13246426?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQyYzM4N2V[…]cPYUhw2W8BjmLYnCihCYE0tISVxeiSOhQp34A5GoqbXowwjJUNl5ew4xdRnqN3Q -->

For orginal datasets, please visit https://microsoft.github.io/msmarco/ for their QA datset, and https://github.com/facebookresearch/DPR?tab=readme-ov-file#resources--data-formats for NQ, SQuAD, and WebQ. 

## Conduct the full experiment

### Step 1: Generate Weak Labels
Go to `weak_label_gen/generate_weak_labels.py`, remember to add your huggingface token in the code. `generate_corpus_queries_qrels()` will create a folder under `data` with corpus, queries, qrels (train, val). Use `data_workers/split_val_groundtruth.py` to split val into val and test. `generate_weak_labels()` will create a folder under `data/xxxxxxx` with `full_rankxx.jsonl`,` train_weak.tsv`, and `train_weak_full.tsv`. <br><br> Run using `python weak_label_gen/generate_weak_labels` 
<br>
Or you can just download from Zenodo to skip this step. You can verify everything is good by running Step 2. 

### Step 2: Evaluate Weak Label Quality
Go to `evaluations/weak_label_quality.py` -> specify k values, and the `full_rankxxx.jsonl` object, to see recall, mrr, etc. 

### Step 3: Train Retriever

Before start training, make sure to make `train_groundtruth_allones.tsv` or `train_groundtruth_top1.tsv` and `val.tsv`. Scripts are in `data_workers`. Or if you downloaded the dataset from our Zenodo, then you should already have all of them.

DPR weak label: 
```
python train_retriever/train_dpr_ibneg.py --product="cosine" --gt_or_weak="weak" --weak_label_path="/llama3_0shot_prompt_top100" --num_epochs=20 --encoder_name="Yibin-Lei/ReContriever" --tsv="train_weak" --data_path="xx"
```

DPR ground-truth:
```
python train_retriever/train_dpr_ibneg.py --num_epochs=20 --encoder_name="Yibin-Lei/ReContriever" --product="cosine"  --gt_or_weak="gt" --tsv="train_groundtruth_top1" --data_path="xx"
```

ColBERT weak label:
```
python train_retriever/train_colbert_ibneg.py --gt_or_weak="weak" --weak_label_path="/llama3_0shot_prompt_top100" --encoder_name="bert-base-uncased" --tsv="train_weak" --neg=10 --lr=1e-5 --data_path="xx"
```

ColBERT ground-truth:
```
python train_retriever/train_colbert_ibneg.py --gt_or_weak="gt" --encoder_name="bert-base-uncased" --tsv="train_groundtruth_top1" --neg=10 --lr=1e-5 --data_path="xx"
```
`--data_path` is the path to the folder inside the data folder, for example, for msmarco the data_path should be `xxxx/data/msmarco_qa_v2_train_corpus500000_weakTrainQ2000_ValQ3000`



### Step 4: Retrieval Evaluations 

BM25: Go to `evaluations/bm25_evaluation`

DPR:
``` 
python evaluations/dpr_ability.py --data_path="xx" --model_path="xx"
``` 
Contriever: 
```
python evaluations/dpr_ability.py --eval_contriever --data_path="xx"
```
ReContriever: 
```
python evaluations/dpr_ability.py --eval_recontriever --data_path="xx"
```
ColBERT: Retrieval and QA evaluation are in the same python file, see next Step.

Note: `--model_path` can be found in `train_retriever/output`. For ColBERT, look inside `.ragatouille`. 



### Step 5: OpenQA Evaluations
<!-- Our code supports `llama3, llama3.1, gemma2, phi3, mistral, llama2` -->

Naive: 
```
python evaluations/QA_performance.py --top_k=0 --num_shot=0 --llm_name="llama3" --hf_token="xx" --new_token=20 --use_gt_passage="n" --data="xx"
```
Groundtruth Passage:
```
python evaluations/QA_performance.py --llm_name="llama3" --hf_token="xx" --new_token=20 --use_gt_passage="y" --data="xx"
```
BM25: 
```
python evaluations/QA_performance.py --model_path="bm25" --top_k=5 --num_shot=0 --llm_name="llama3" --hf_token="xx" --new_token=20 --use_gt_passage="n" --data="xx"
```
DPR/Contriever/ReContriever (indexing takes a long time, we save the index by just saving all the embeddings to a `.pth` file):
```
python evaluations/QA_performance.py --top_k=5 --num_shot=0 --llm_name="llama3" --hf_token="xx" --new_token=20 --use_gt_passage="n" --data="xx" --model_path="xx"
``` 
ColBERT: (index creation might take a long time. If you already have index, no need to add model_path)
```
python evaluations/colbert_ability.py --llm_name="llama3" --hf_token="xx" --new_token=20 --data_path="xx" --model_path="xx" --index_name="give it a name"
```

### Ablations:

Table 4 in the paper (Experiments for Direct QA Using LLM Reranked Passages): Go to `evaluations/bm25top100_llmRerank_directlyQA.py` and change the `which_experiment` at the top of the file. <br>

Figure 3 in the paper (Fix Prompt Study Different LLM, MSMARCO on 500 Val): 
Go to `weak_label_gen/generate_weak_labels.py`, change parameters in `study_different_llm_on_msmarco`, comment out other code and run `python weak_label_gen/generate_weak_labels.py`


<!-- ## Citation

```
@misc{nian2024wragweaklysuperviseddense,
      title={W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain Question Answering}, 
      author={Jinming Nian and Zhiyuan Peng and Qifan Wang and Yi Fang},
      year={2024},
      eprint={2408.08444},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.08444}, 
}
``` -->
