import os, sys, csv, random, json, re, pickle
# os.environ['HF_HOME'] = '/local/scratch' # Comment this out if you are not running on WAVE
# os.environ['TRANSFORMERS_CACHE'] = '/local/scratch'
import numpy as np
import pandas as pd 
from tqdm import tqdm
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # pipeline,
)

def set_random_seed(seed_value=42):
    """Set seed for reproducibility"""
    random.seed(seed_value)     
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value) 

def save_to_jsonl(data, file_path):
    if os.path.exists(file_path):
        print(f"File {file_path} already exists.")
        return
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data.values():
            f.write(json.dumps(item) + '\n')
    print(f"Successfully written {file_path}")

def load_jsonl(data_dict, file_path): 
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            data_dict[int(data['_id'])] = data['text'] # cast "1183942" into 1183942 as the keys 
    return data_dict 
            
def save_to_tsv(data, file_path):
    if os.path.exists(file_path):
        print(f"File {file_path} already exists.")
        return
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, corpus_id, score in data:
            f.write(f"{qid}\t{corpus_id}\t{score}\n")
    print(f"Successfully written {file_path}")

def load_or_create_bm25(corpus, filename):
    # Check if the serialized BM25 object exists
    if os.path.exists(filename):
        print(f"Loading BM25 object from {filename}")
        with open(filename, 'rb') as f:
            bm25 = pickle.load(f)
        print(f"BM25 loaded successfully")
    else:
        print("BM25 object not found, creating now...")
        tokenized_corpus = {pid: word_tokenize(text.lower()) for pid, text in corpus.items()}
        sorted_tokenized_passages = [tokenized_corpus[pid] for pid in sorted(tokenized_corpus)]
        bm25 = BM25Okapi(sorted_tokenized_passages)
        with open(filename, 'wb') as f:
            pickle.dump(bm25, f)
        print(f"BM25 object created and saved to {filename}")
    return bm25

def load_llm(llm_name, device, quantize=False, hf_token=""):
    model_name = None
    model = None 
    if llm_name == "llama3":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif llm_name == "llama3.1":
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif llm_name == "gemma2":
        model_name = "google/gemma-2-9b-it"
    elif llm_name == "phi3":
        model_name = "microsoft/Phi-3-small-8k-instruct"
    elif llm_name == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif llm_name == "llama2":
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    else: 
        print(f"'{llm_name}' is not valid")
        return model 
    print(f"Loading {model_name}")
    if quantize: 
        # 4-bit quantization
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",              # the type of quantization
            bnb_4bit_compute_dtype=compute_dtype,   # data type for computation
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.bfloat16, # if using 16 bit
            quantization_config=quant_config, # if using 4 bit
            # device_map={"": 0}, # more advanced way to map different layers to different device
            token=hf_token,
        )
    else: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, # if using 16 bit
            # quantization_config=quant_config, # if using 4 bit
            # device_map={"": 0}, # more advanced way to map different layers to different device
            token=hf_token,
        )
        model.to(device) # comment out if using 4 bit
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token # help handling the end of generated texts
    tokenizer.padding_side = "right" # to fix the issue with fp16 during forward pass
    return model, tokenizer 

def llm_calculate_sequence_log_likelihood(shots, question, passage, answer, model, tokenizer, device):
    # input text contains question, candidate passage, ground truth answer
    # instead of asking LLM to generate, we look at the output logits for generating each answer tokens
    # we calculate the sum of each answer token's generation probability and normalize it by answer length
    #Old prompt: input_text = f"PASSAGE: {passage}Only use text from the above given passage to answer the question in one sentence.QUESTION: {question}ANSWER: {answer}"
    input_text = ""
    if shots == 0:
        input_text = f'''DOCUMENT: {passage}
QUESTION: {question}
Answer the user's QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
Keep your answer within one short sentence.
ANSWER: {answer}'''
    elif shots == 1: 
        input_text = f'''DOCUMENT: If you have arthritis, you may have considered a cortisone shot as part of your treatment plan. These shots are not pain relievers. Cortisone is a type of steroid, a drug that lowers inflammation, which is something that can lead to less pain. Cortisone injections can be used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon. They can also treat inflammation that is widespread throughout the body, such as with allergic reactions, asthma, and rheumatoid arthritis, which affects many joints.
QUESTION: what is a cortisone injection used for
ANSWER: Used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon.

DOCUMENT: {passage}
QUESTION: {question}
Answer the user's QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
Keep your answer within one short sentence.
ANSWER: {answer}'''
    elif shots == 2: 
        input_text = f'''DOCUMENT: If you have arthritis, you may have considered a cortisone shot as part of your treatment plan. These shots are not pain relievers. Cortisone is a type of steroid, a drug that lowers inflammation, which is something that can lead to less pain. Cortisone injections can be used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon. They can also treat inflammation that is widespread throughout the body, such as with allergic reactions, asthma, and rheumatoid arthritis, which affects many joints.
QUESTION: what is a cortisone injection used for
ANSWER: Used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon.

DOCUMENT: September Weather in Paris. – min temperature : 11°C / 52°F. – max temperature : 21°C / 70°F. – average rainfall : 54 mm. Although the Parisian weather can be quite variable in September, it remains pleasant for the most part. September is generally a sunny month with one of the lowest average rainfalls in the year.
QUESTION: average temperature in paris in september in fahrenheit
ANSWER: Min temperature: 52°F and max temperature: 70°F.

DOCUMENT: {passage}
QUESTION: {question}
Answer the user's QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
Keep your answer within one short sentence.
ANSWER: {answer}'''

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # each token's id in answer, [1:] is to exclude the <s> token in the beginning
    answer_ids = tokenizer.encode(answer, return_tensors='pt')[0][1:].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_softmax = torch.nn.LogSoftmax(dim=-1)
    probs = log_softmax(logits)

    # exclude the last logit 
    answer_probability = probs[0, :-1]

    # only select probabilities for each token in the answer
    answer_probability = answer_probability[-len(answer_ids):]

    # gather the probabilities for each token in the answer
    actual_token_probs = answer_probability.gather(1, answer_ids.unsqueeze(-1)).squeeze(-1)

    average_probability = actual_token_probs.sum().item() / len(answer_ids)

    return average_probability

def query_likelihood(shots, question, passage, answer, model, tokenizer, device):
    input_text = ""
    if shots == 0:
        input_text = f'''PASSAGE: {passage}
Please write a question based on this passage.
QUESTION: {question}'''
    else: 
        print("UPR only supports 0 shot prompt")
        x=1/0

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    question_ids = tokenizer.encode(question, return_tensors='pt')[0][1:].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_softmax = torch.nn.LogSoftmax(dim=-1)
    probs = log_softmax(logits)
    question_probability = probs[0, :-1]
    question_probability = question_probability[-len(question_ids):]
    actual_token_probs = question_probability.gather(1, question_ids.unsqueeze(-1)).squeeze(-1)
    average_probability = actual_token_probs.sum().item() / len(question_ids)
    return average_probability

def generate_corpus_queries_qrels(dataset_name: str = "msmarco_qa_v2", 
             from_split: str = "train",
             sample_corpus_size: int = 500_000,
             num_question: int = 1_200, # 1200 is roughly how long until the 2 day limit on WAVE
             num_val_question: int = 10_000, 
             passage_id_start: int = 0, # when we need more weak labels, this controls where passage id pick up from to form the corpus file 
             out_root_dir: str = "/WAVE/users2/unix/jnian/WeakLabelForRAG/weak_label_gen"
            ): 
    '''
    Based on function inputs, sample some qustions that will be used to generate weak labels, 
    sample some questions for validation, and further sample a corpus that encloses all the associated passages as well as some random ones. 
    In the created directory, save 5 files: 
    corpus.jsonl: {"_id": "xxx"(our defined passage_id), "title": "", "text": "xxx", "metadata": {}}
    queries.jsonl: {"_id": "xxx"(the official query_id), "text": "xxx", "metadata": {}}
    answers.jsonl: {"_id": "xxx"(the official query_id), "text": "xxx"}
    qrels/val_groundtruth.tsv: query-id \t corpus-id \t score (1s and 0s)
    qrels/train_groundtruth.tsv: query-id \t corpus-id \t score (1s and 0s)
    
    Returns: directory_name
    '''
    
    directory_name = f"{out_root_dir}/{dataset_name}_{from_split}_corpus{sample_corpus_size}_weakTrainQ{num_question}_ValQ{num_val_question}"
    if not os.path.exists(directory_name): 
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    else: 
        print(f"Directory '{directory_name}' already exists.")

    pid = passage_id_start
    corpus = {} # pid: "passage_string"
    corpus_rev = {} # "passage_string": pid
    qid2qrel_train = {} # qid: ("question_string", [pid], "answer_string")
    qid2qrel_val = {} # qid: ("question_string", [pid], "answer_string")
    train_groundtruth_qrel = [] # (qid, pid, rel)
    val_groundtruth_qrel = [] # (qid, pid, rel)
    if dataset_name == "msmarco_qa_v2": 
        dataset = load_dataset("microsoft/ms_marco", "v2.1")
        df = dataset[from_split].to_pandas()
        df = df.sample(frac=1).reset_index(drop=True) # shuffle 
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
            if row["answers"][0] == "No Answer Present.":
                continue
            question_string, qid = row["query"], row["query_id"]
            answer_string = row["answers"][0] if len(row["wellFormedAnswers"]) == 0 else row["wellFormedAnswers"][0]
            # iterate through this sample, fill corpus, select the relevant passage
            for p, rel in zip(row['passages']['passage_text'], row['passages']['is_selected']): 
                corpus[pid] = p 
                corpus_rev[p] = pid
                pid += 1 

            # 0~1119: gather train set, later generate weak label
            if len(qid2qrel_train) < num_question: 
                qid2qrel_train[qid] = (question_string, answer_string)
                for p, rel in zip(row['passages']['passage_text'], row['passages']['is_selected']):
                    train_groundtruth_qrel.append((qid, corpus_rev[p], rel))
            # 1200~11200: gather val set
            elif len(qid2qrel_val) < num_val_question: 
                qid2qrel_val[qid] = (question_string, answer_string)
                for p, rel in zip(row['passages']['passage_text'], row['passages']['is_selected']):
                    val_groundtruth_qrel.append((qid, corpus_rev[p], rel))
            elif len(corpus) >= sample_corpus_size: 
                break
    
    # pid = passage_id_start
    # corpus = {} # pid: "passage_string"
    # corpus_rev = {} # "passage_string": pid
    # qid2qrel_train = {} # qid: ("question_string", [pid], "answer_string")
    # qid2qrel_val = {} # qid: ("question_string", [pid], "answer_string")
    # train_groundtruth_qrel = [] # (qid, pid, rel)
    # val_groundtruth_qrel = [] # (qid, pid, rel)
    else: 
        if dataset_name == "trivia":
            file_name = "/trivia-train.json"
        if dataset_name == "nq":
            file_name = "/nq-train.json"
        if dataset_name == "squad":
            file_name = "/squad1-train.json"
        if dataset_name == "wq":
            file_name = "/webq-train.json"
        file_path = "/WAVE/datasets/yfang_lab/QA_datasets/dpr/retriever" + file_name
        df = pd.read_json(file_path)
        
        qid = -1
        pid = -1
        
        all_text = set()
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
            if len(row['positive_ctxs']) == 0:
                continue 
            # There are always multiple answers, but we only take the first one (row['answers'][0]) for generating weak labels. 
            # During evaluation, we may look up entity by the question string and evaluate if LLM generates any of the answers
            question_string, answer_string, qid = row['question'], row['answers'][0], qid + 1 
            
            for entity in row['positive_ctxs']:
                # pid = int(entity['passage_id']) if dataset_name =='nq' else int(entity['psg_id'])
                text = entity['title'] + " " + entity['text']
                pid += 1
                corpus[pid] = text
                all_text.add(text)
                if len(qid2qrel_train) < num_question: 
                    train_groundtruth_qrel.append((qid, pid, 1)) 
                elif len(qid2qrel_val) < num_val_question: 
                    val_groundtruth_qrel.append((qid, pid, 1))
                    
            # if dataset_name == "wq":
            #     first_few_negative_ctxs = row['negative_ctxs']
            #     first_few_hard_negative_ctxs = row['hard_negative_ctxs']
            # else: 
            #     first_few_negative_ctxs = row['negative_ctxs'][:30]
            #     first_few_hard_negative_ctxs = row['hard_negative_ctxs'][:30]
                
            # for entity in (first_few_negative_ctxs + first_few_hard_negative_ctxs): 
            #     pid = int(entity['passage_id']) if dataset_name =='nq' else int(entity['psg_id'])
            #     text = entity['title'] + " " + entity['text']
            #     corpus[pid] = text
            #     if len(qid2qrel_train) < num_question: 
            #         train_groundtruth_qrel.append((qid, pid, 0)) 
            #     elif len(qid2qrel_val) < num_val_question: 
            #         val_groundtruth_qrel.append((qid, pid, 0))
                
            if len(qid2qrel_train) < num_question: 
                qid2qrel_train[qid] = (question_string, answer_string)
            elif len(qid2qrel_val) < num_val_question:     
                qid2qrel_val[qid] = (question_string, answer_string)
            elif len(corpus) >= sample_corpus_size: 
                print(f"Hey, corpus exceeded {sample_corpus_size}, early stop negative collection!")
                x=1/0
            else: 
                break 
            
        # gather all passages, fill to corpus until reach sample_corpus_size, throw out too similar passages 
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
            for entity in (row['negative_ctxs'] + row['hard_negative_ctxs']):
                text = entity['title'] + " " + entity['text']
                all_text.add(text)

        # Shuffle the unique texts
        unique_texts = list(all_text)
        random.shuffle(unique_texts)

        for text in tqdm(unique_texts, desc="Iterating through Shuffled Texts"):
            if len(corpus) == sample_corpus_size: 
                break 
            pid += 1
            corpus[pid] = text 
    
    assert len(qid2qrel_train) == num_question
    assert len(qid2qrel_val) == num_val_question
    assert len(corpus) >= sample_corpus_size and len(corpus) <= sample_corpus_size + 10
    combined_dict = {**qid2qrel_train, **qid2qrel_val}
    assert len(combined_dict) == num_question + num_val_question
    
    corpus_data = {pid: {"_id": str(pid), "title": "", "text": text, "metadata": {}} for pid, text in corpus.items()}
    queries_data = {qid: {"_id": str(qid), "text": item[0], "metadata": {}} for qid, item in combined_dict.items()}
    answers_data = {qid: {"_id": str(qid), "text": item[1]} for qid, item in combined_dict.items()}
    save_to_jsonl(corpus_data, f"{directory_name}/corpus.jsonl")
    save_to_jsonl(queries_data, f"{directory_name}/queries.jsonl")
    save_to_jsonl(answers_data, f"{directory_name}/answers.jsonl")
    os.makedirs(f"{directory_name}/qrels")
    save_to_tsv(train_groundtruth_qrel, f"{directory_name}/qrels/train_groundtruth.tsv")
    save_to_tsv(val_groundtruth_qrel, f"{directory_name}/qrels/val_groundtruth.tsv")
    
    return directory_name



def generate_weak_labels(use_upr: str=False,
                         llm_name: str = "llama3", 
                         first_stage_ret_count: int = 100, 
                         how_many_shot_prompt: int = 0,
                         data_path: str = "",
                         hf_token: str = ""): 
    """
    Load from jsonl: questions = {qid: "question"}      corpus = {pid: "passage"}     answers = {qid: "answer"}     
    Load from qrels: qrel = {qid: pid}     from train_groundtruth.tsv (take all 1200 unique query ids)
    Construct BM25 using corpus (500_000 passages)
    Load LLM 
    iterate through qrel: 
        weak_and_ground_truth_labels[qid] = {}
        Fill weak_and_ground_truth_labels[pid]["rel_passage"] = qrel[pid]
        BM25 retrieve 100 passages, add to bm25_list
        LLM rerank 100 passages, add to weak_label_list
    !!!!!! Save weak_and_ground_truth_labels to a json !!!!!!!!!

    Parameters:
    llm_name (str): llama3/gemma2/phi3/mistral/llama2
    first_stage_ret_count (int): how many passages per question to retrieve using bm25 
    how_namy_shot_prompt (int): 0/1/2 
    data_path (str): where to find data (corpus, queries, answers, qrels)

    Returns:
    weak_and_ground_truth_labels, A DICT:  
                {   
                    qid: {  
                            "rel_passage": [pid, pid, ...],              # 1 passage for msmarco, 1-4 for hotpotqa
                            "llm_ranked_list":  [(pid, log_likelihood), ...],
                            "bm25_ranked_list": [(pid, bm25_score), ...], 
                        }
                }
    """ 
    if how_many_shot_prompt != 0 and how_many_shot_prompt != 1 and how_many_shot_prompt != 2: 
        print(f"Only 0/1/2 shots prompts are supported. You entered {how_many_shot_prompt}")
        return 
    if llm_name != "llama3" and llm_name != "llama3.1" and llm_name != "gemma2" and llm_name != "phi3" and llm_name != "mistral" and llm_name != "llama2":
        print(f"Only llama3/llama3.1/gemma2/phi3/mistral/llama2 are supported. You entered {llm_name}")
        return
    print(f"Working on {how_many_shot_prompt}shot using {llm_name}, {data_path}, using UPR?{use_upr}. and BM25 retrieve top {first_stage_ret_count}")
    
    # Set up, load stuff
    questions, corpus, answers, qrel, weak_and_ground_truth_labels = {}, {}, {}, {}, {}
    questions = load_jsonl(questions, f"{data_path}/queries.jsonl")
    corpus = load_jsonl(corpus, f"{data_path}/corpus.jsonl")
    answers = load_jsonl(answers, f"{data_path}/answers.jsonl")
    qrel_path = f"{data_path}/qrels/train_groundtruth.tsv"
    with open(qrel_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            qid = int(row['query-id'])
            if row['score'] == '1': 
                if qid not in qrel:
                    qrel[qid] = []  # Initialize an empty list for this query-id if it doesn't exist
                qrel[qid].append(int(row['corpus-id']))
    bm25 = load_or_create_bm25(corpus, f"{data_path}/bm25_{len(corpus)}.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using", device)
    llm, llm_tokenizer = load_llm(llm_name, device, False, hf_token)
    
    # Retrieve and rerank 
    weak_labels = []
    weak_labels_full = []
    # output_data = []
    # count = 0
    for qid, rel_pids in tqdm(qrel.items(), desc="Processing questions"):
        question, answer = questions[qid], answers[qid]
        ### (1) bm25 retrieve top 100 
        question_tokens = word_tokenize(question.lower())
        passage_scores = bm25.get_scores(question_tokens)
        sorted_indices = np.argsort(passage_scores)[::-1][:first_stage_ret_count]
        top_passages_with_scores = [(list(corpus.keys())[idx], corpus[list(corpus.keys())[idx]], passage_scores[idx]) for idx in sorted_indices]
    #     output_data.append(f"Question: {question}")
    #     output_data.append(f"Answer: {answer}")
    #     for pid, text, score in top_passages_with_scores[:20]:
    #         output_data.append(f"PID: {pid}, Score: {score:.5f}, Passage: {text}...")
    #     output_data.append("")  # Add a blank line for separation between questions
    #     count += 1
    #     if count == 20: 
    #         break 
    # with open("/WAVE/users2/unix/jnian/WeakLabelForRAG/weak_label_gen/bm25_output.txt", 'w') as file:
    #     for line in output_data:
    #         file.write(line + '\n')
        weak_and_ground_truth_labels[qid] = {}
        weak_and_ground_truth_labels[qid]["rel_passage"] = rel_pids # why list of rel_pid you may ask, because nq, trivia, wq, squad has multiple relevant passages
        weak_and_ground_truth_labels[qid]["bm25_ranked_list"] = [(pid, score) for pid, _, score in top_passages_with_scores]
        ### (2) rerank by infer LLM 1 by 1, record score
        llm_score_list = []
        if use_upr: 
            for pid, passage, _ in top_passages_with_scores:
                log_likelihood = query_likelihood(how_many_shot_prompt, question, passage, answer, llm, llm_tokenizer, device) 
                llm_score_list.append((pid, log_likelihood))
        else: # U-RAG
            for pid, passage, _ in top_passages_with_scores:
                log_likelihood = llm_calculate_sequence_log_likelihood(how_many_shot_prompt, question, passage, answer, llm, llm_tokenizer, device) 
                llm_score_list.append((pid, log_likelihood))
        llm_ranked_list = sorted(llm_score_list, key=lambda x: x[1], reverse=True)
        weak_and_ground_truth_labels[qid]["llm_ranked_list"] = llm_ranked_list 
        for index, (pid, likelihood) in enumerate(llm_ranked_list): 
            # print(f"Score: {likelihood:.5f}, PID: {pid}, Passage: {corpus[pid][:100]}...")
            if index == 0:
                weak_labels.append((qid, pid, 1)) 
                weak_labels_full.append((qid, pid, 1))
            else: 
                weak_labels_full.append((qid, pid, 0))

            
    # Save weak labels and the full rankings 
    
    if use_upr:
        folder_name = f"{data_path}/upr_{llm_name}_{how_many_shot_prompt}shot_prompt_top{first_stage_ret_count}"
    else: 
        folder_name = f"{data_path}/{llm_name}_{how_many_shot_prompt}shot_prompt_top{first_stage_ret_count}"
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)
        print(f"Directory '{folder_name}' created successfully.")
    else: 
        print(f"Directory '{folder_name}' already exists.")
        
    file_path = f"{folder_name}/full_rank_lists{len(weak_and_ground_truth_labels)}.jsonl"
    if os.path.exists(file_path):
        print(f"File {file_path} already exists.")
        return
    with open(file_path, 'w', encoding='utf-8') as f:
        for qid, item in weak_and_ground_truth_labels.items():
            json_object = json.dumps({"qid": str(qid), **item})
            f.write(json_object + '\n')
    print(f"Successfully written {file_path}")
    save_to_tsv(weak_labels, f"{folder_name}/train_weak.tsv")
    save_to_tsv(weak_labels_full, f"{folder_name}/train_weak_full.tsv")
            
        
    '''
    {   
        qid: {  
                "rel_passage": [pid, pid, ...],              # 1 passage for msmarco, 1-4 for hotpotqa
                "llm_ranked_list":  [(pid, log_likelihood), ...],
                "bm25_ranked_list": [(pid, bm25_score), ...], 
            }
    }
    '''
        

        
def produce_all_answers_file(dataset_name, num_question, num_val_question, save_folder):
    corpus = {} # pid: "passage_string"
    corpus_rev = {} # "passage_string": pid
    qid2qrel_train = {} # qid: ("question_string", [pid], "answer_string")
    qid2qrel_val = {} # qid: ("question_string", [pid], "answer_string")
    train_groundtruth_qrel = [] # (qid, pid, rel)
    val_groundtruth_qrel = [] # (qid, pid, rel)
    if dataset_name == "trivia":
        file_name = "/trivia-train.json"
    if dataset_name == "nq":
        file_name = "/nq-train.json"
    if dataset_name == "squad":
        file_name = "/squad1-train.json"
    if dataset_name == "wq":
        file_name = "/webq-train.json"
    file_path = "/WAVE/datasets/yfang_lab/QA_datasets/dpr/retriever" + file_name
    df = pd.read_json(file_path)
    
    qid = -1
    pid = -1
    
    all_text = set()
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        if len(row['positive_ctxs']) == 0:
            continue 
        # There are always multiple answers, but we only take the first one (row['answers'][0]) for generating weak labels. 
        # During evaluation, we may look up entity by the question string and evaluate if LLM generates any of the answers
        question_string, answer_list, qid = row['question'], row['answers'], qid + 1 
        
        for entity in row['positive_ctxs']:
            # pid = int(entity['passage_id']) if dataset_name =='nq' else int(entity['psg_id'])
            text = entity['title'] + " " + entity['text']
            pid += 1
            corpus[pid] = text
            all_text.add(text)
            if len(qid2qrel_train) < num_question: 
                train_groundtruth_qrel.append((qid, pid, 1)) 
            elif len(qid2qrel_val) < num_val_question: 
                val_groundtruth_qrel.append((qid, pid, 1))
            
        if len(qid2qrel_train) < num_question: 
            qid2qrel_train[qid] = (question_string, answer_list)
        elif len(qid2qrel_val) < num_val_question:     
            qid2qrel_val[qid] = (question_string, answer_list)
        else: 
            break 
        
    
    assert len(qid2qrel_train) == num_question
    assert len(qid2qrel_val) == num_val_question
    combined_dict = {**qid2qrel_train, **qid2qrel_val}
    assert len(combined_dict) == num_question + num_val_question
    

    answers_data = {qid: {"_id": str(qid), "text": item[1]} for qid, item in combined_dict.items()}

    save_to_jsonl(answers_data, f"{save_folder}/all_answers.jsonl")

        
        
        
def recall_at_k(ranked_list, relevant_set, k):
    relevant_retrieved = 0
    for pid, _ in ranked_list[:k]:
        if pid in relevant_set:
            relevant_retrieved += 1
    return relevant_retrieved / len(relevant_set)

def mean_reciprocal_rank(ranked_list, relevant_set):
    for index, (pid, _) in enumerate(ranked_list, start=1):
        if pid in relevant_set:
            return 1 / index
    return 0

def study_different_llm_on_msmarco(first_stage_ret_count, llm_name, levels, hf_token): 
    
    data_path = "data/msmarco_qa_v2_train_corpus500000_weakTrainQ2048_ValQ10000"
    
    questions, corpus, answers, qrel, weak_and_ground_truth_labels = {}, {}, {}, {}, {}
    questions = load_jsonl(questions, f"{data_path}/queries.jsonl")
    corpus = load_jsonl(corpus, f"{data_path}/corpus.jsonl")
    answers = load_jsonl(answers, f"{data_path}/answers.jsonl")
    qrel_path = f"{data_path}/qrels/val.tsv"
    with open(qrel_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            qid = int(row['query-id'])
            if row['score'] == '1': 
                if qid not in qrel:
                    qrel[qid] = []  # Initialize an empty list for this query-id if it doesn't exist
                qrel[qid].append(int(row['corpus-id']))
    
    bm25 = load_or_create_bm25(corpus, f"{data_path}/bm25_{len(corpus)}.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using", device)
    llm, llm_tokenizer = load_llm(llm_name, device, False. hf_token)
        
    for qid, rel_pids in tqdm(qrel.items(), desc="Processing questions"):
        question, answer = questions[qid], answers[qid]
        ### (1) bm25 retrieve top 100 
        question_tokens = word_tokenize(question.lower())
        passage_scores = bm25.get_scores(question_tokens)
        sorted_indices = np.argsort(passage_scores)[::-1][:first_stage_ret_count]
        top_passages_with_scores = [(list(corpus.keys())[idx], corpus[list(corpus.keys())[idx]], passage_scores[idx]) for idx in sorted_indices]

        weak_and_ground_truth_labels[qid] = {}
        weak_and_ground_truth_labels[qid]["rel_passage"] = rel_pids # why list of rel_pid you may ask, because nq, trivia, wq, squad has multiple relevant passages
        weak_and_ground_truth_labels[qid]["bm25_ranked_list"] = [(pid, score) for pid, _, score in top_passages_with_scores]
        ### (2) rerank by infer LLM 1 by 1, record score
        llm_score_list = []
        for pid, passage, _ in top_passages_with_scores:         # 0 here means 0 shot prompt
            log_likelihood = llm_calculate_sequence_log_likelihood(0, question, passage, answer, llm, llm_tokenizer, device) 
            llm_score_list.append((pid, log_likelihood))
        llm_ranked_list = sorted(llm_score_list, key=lambda x: x[1], reverse=True)
        weak_and_ground_truth_labels[qid]["llm_ranked_list"] = llm_ranked_list 
    
    # Use weak_and_ground_truth_labels to evaluate the quality of weak labels and bm25 performance
    results = {
        'recall': {model: {k: [] for k in levels} for model in ['llm', 'bm25']},
        'mrr': {model: [] for model in ['llm', 'bm25']},
    }

    # Process each query and calculate metrics
    for qid, info in weak_and_ground_truth_labels.items():
        relevant_pids = set(info['rel_passage'])
        for model in ['llm_ranked_list', 'bm25_ranked_list']:
            model_key = 'llm' if model == 'llm_ranked_list' else 'bm25'
            ranked_list = sorted(info[model], key=lambda x: x[1], reverse=True)
            
            for k in levels:
                results['recall'][model_key][k].append(recall_at_k(ranked_list, relevant_pids, k))
            results['mrr'][model_key].append(mean_reciprocal_rank(ranked_list, relevant_pids))

    # Print out the results and perform statistical tests
    for metric in results:
        if metric == 'recall':
            for k in levels:
                avg_recall_llm = np.mean(results['recall']['llm'][k])
                avg_recall_bm25 = np.mean(results['recall']['bm25'][k])
                print(f"Average Recall@{k} for LLM: {avg_recall_llm:.4f}, BM25: {avg_recall_bm25:.4f}")
        else:
            avg_metric_llm = np.mean(results[metric]['llm'])
            avg_metric_bm25 = np.mean(results[metric]['bm25'])
            print(f"Average {metric.upper()} for LLM: {avg_metric_llm:.4f}, BM25: {avg_metric_bm25:.4f}")
    
    
    
if __name__ == '__main__': 
    set_random_seed(42)
    hf_token = "xx" # Put yours here
    path = generate_corpus_queries_qrels("msmarco", "train", 500_000, 2000, 3000, 0, "WeakLabelForRAG/data")
    # specifically for WebQ:
    # path = generate_corpus_queries_qrels("wq", "train", 163683, 2000, 474, 0, "WeakLabelForRAG/data")
    generate_weak_labels(False, "llama3", 100, 0, "data_path here", hf_token)

    ######## utility function
    # produce_all_answers_file("wq", 2000, 474, "/WeakLabelForRAG/data/wq_train_corpus163683_weakTrainQ2000_ValQ474", hf_token)
    
    
    
    # # Below are not for generating weak label dataset
    # ###################################################################################
    # # supported llm names: llama3   llama3.1    gemma2     phi3    mistral     llama2
    # #                      bm25 results will come out together
    # # first_stage_ret_count: how many documents BM25 retrieves for a given query
    # # levels: the Ks in recall@k and overall mrr
    # study_different_llm_on_msmarco(first_stage_ret_count=100, llm_name="llama3", levels=[1, 5, 10, 20, 50, 100], hf_token=hf_token)
    #####################################################################################