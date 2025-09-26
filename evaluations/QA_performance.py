import torch, argparse, json, os, logging, pickle, time
os.environ['HF_HOME'] = '/local/scratch'
# os.environ['TRANSFORMERS_CACHE'] = '/local/scratch'
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import eval_util
from transformers import pipeline

logging.getLogger("transformers").setLevel(logging.ERROR)


parser = argparse.ArgumentParser(description="Script for training a retriever")
parser.add_argument('--model_path', type=str, help='Path to the model (the parent folder of state dict), or just type bm25')
parser.add_argument('--top_k', type=int, help="Retrieve top k to add to prompt for QA")
parser.add_argument('--num_shot', type=int, help="How many shots you want the prompt to have")
parser.add_argument('--llm_name', type=str, help='Choose a llm among llama3/llama3.1/gemma2/phi3/mistral/llama2')
parser.add_argument('--hf_checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--hf_token', type=str, help='Add your HuggingFace token here')
parser.add_argument('--new_token', type=int, help='number of new tokens to generate')
parser.add_argument('--use_gt_passage', type=str, help='Use ground truth passage to QA')
parser.add_argument('--data', type=str, help='Which dataset to evaluate on')
args = parser.parse_args()

data_path = ""
if args.data == "msmarco":
    data_path = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/msmarco_qa_v2_train_corpus500000_weakTrainQ2048_ValQ10000"
elif args.data == "nq":
    data_path = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/nq_train_corpus500000_weakTrainQ2000_ValQ3000"
elif args.data == "squad":
    data_path = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/squad_train_corpus500000_weakTrainQ2000_ValQ3000"
elif args.data == "trivia":
    data_path = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/trivia_train_corpus500000_weakTrainQ2000_ValQ3000"
elif args.data == "wq":
    data_path = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/wq_train_corpus163683_weakTrainQ2000_ValQ474"
else: 
    print(f"{args.data} is not supported")
    x=1/0

output_file_name = ""
if args.top_k == 0 and args.num_shot == 0:
    output_file_name = f"evaluations/qa/{args.data}/no_passage"
elif args.use_gt_passage == "y":
    output_file_name = f"evaluations/qa/{args.data}/gt_passage"
else: 
    output_file_name = f"evaluations/qa/{args.data}/{args.model_path.split('/')[-1]}"
    
    
# Which set to use, test/val/train maybe
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="test")
if args.data == "msmarco":
    answers = eval_util.load_jsonl({}, f"{data_path}/answers.jsonl")
else: 
    answers = eval_util.load_jsonl({}, f"{data_path}/all_answers.jsonl")
'''
dev_corpus: {id: xxx, text: {xxx}, title: {xxx}}
dev_queries: {'619943': 'what did the colossal stone heads of the olmecs'}
dev_qrels: {'619943': {'docid': 1, 'docid': 0}}
answers: {'619943': "colossal stone heads are xxx"}
'''

corpus = [item['title']+' '+item['text'] for _, item in dev_corpus.items()]
pqa_list = []

if args.use_gt_passage == "y":
    for qid, rels in dev_queries.items():
        query, answer = dev_queries[qid], answers[qid]
        rel_passage = [dev_corpus[doc_id]['text'] for doc_id, value in dev_qrels[qid].items() if value == 1]
        if len(rel_passage) > 0:
            pqa_list.append((rel_passage, query, answer, qid))
    print(f"Experiment Info: {args.llm_name} ground truth passage {args.num_shot}shot prompt, {len(pqa_list)}questions. Corpus is {args.data}")
elif args.model_path == "bm25":
    save_path = f"/WAVE/users2/unix/jnian/WeakLabelForRAG/evaluations/bm25_retrieval_results/{args.data}_top{args.top_k}.pkl"
    if os.path.exists(save_path):
        print(f"Loading saved BM25 results from {save_path}")
        with open(save_path, "rb") as f:
            pqa_list = pickle.load(f)
    else:
        print("BM25 results not found. Running retrieval and saving results...")
        corpus = eval_util.load_jsonl_int_key({}, f"{data_path}/corpus.jsonl")  # corpus: dict
        bm25 = eval_util.load_or_create_bm25(corpus, f"{data_path}/bm25_{len(corpus)}.pkl")

        pqa_list = []
        for qid, rels in tqdm(dev_queries.items(), desc="BM25 Retrieving"):
            rel_passage = [dev_corpus[doc_id]['text'] for doc_id, value in dev_qrels[qid].items() if value == 1]
            if len(rel_passage) > 0:
                query, answer = dev_queries[qid], answers[qid]
                passages = eval_util.bm25_retrieve(bm25, corpus, query, args.top_k)
                pqa_list.append((passages, query, answer, qid))
        
        # Save results
        with open(save_path, "wb") as f:
            pickle.dump(pqa_list, f)
        print(f"BM25 results saved to {save_path}")
    print(f"Experiment Info: {args.llm_name} {args.top_k}passage {args.num_shot}shot prompt, {len(pqa_list)}questions. Retriever is BM25. Corpus is {args.data}")
elif args.top_k == 0 and args.num_shot == 0:  
    for qid, rels in dev_queries.items():
        rel_passage = [dev_corpus[doc_id]['text'] for doc_id, value in dev_qrels[qid].items() if value == 1] # just for consistency
        if len(rel_passage) > 0:
            query, answer = dev_queries[qid], answers[qid]
            pqa_list.append(([], query, answer, qid))
    print(f"Experiment Info: {args.llm_name} no passage {args.num_shot}shot prompt, {len(pqa_list)}questions. Corpus is {args.data}")
else: 
    model = SentenceTransformer(args.model_path)
    model_folder = args.model_path.split("/")[-1]
    corpus_embeddings_file_name = "evaluations/corpus_embeddings/" + model_folder + "_" + args.data + ".pt" # evaluations/corpus_embeddings/model_folder_name.pt
    if os.path.exists(corpus_embeddings_file_name):
        corpus_embeddings = eval_util.load_embeddings(corpus_embeddings_file_name)
    else: 
        corpus = [item['title']+' '+item['text'] for _, item in dev_corpus.items()] # a list
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        eval_util.save_embeddings(corpus_embeddings, corpus_embeddings_file_name)

    for qid, rels in dev_queries.items():
        rel_passage = [dev_corpus[doc_id]['text'] for doc_id, value in dev_qrels[qid].items() if value == 1]
        if len(rel_passage) > 0:
            query, answer = dev_queries[qid], answers[qid]
            query_embedding = model.encode(query, convert_to_tensor=True) 
            cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cosine_scores, k=args.top_k)
            # for score, idx in zip(top_results[0], top_results[1]):
            #     print(corpus[idx], "(Score: {:.4f})".format(score))
            # break 
            passages = [corpus[idx] for idx in top_results[1]]
            pqa_list.append((passages, query, answer, qid))
    print(f"Experiment Info: {args.llm_name} {args.top_k}passage {args.num_shot}shot prompt, {len(pqa_list)}questions. Retriever is {model_folder}. Corpus is {args.data}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# llm, tokenizer = eval_util.load_llm(args.llm_name, device, quantize=False, hf_token=args.hf_token)
generator = pipeline("text-generation",model=args.hf_checkpoint,device="cuda",torch_dtype=torch.float16,token=args.hf_token)

prompt_func_selector = {20:{0: eval_util.twenty_passage_0shot_prompt,},
                        10:{0: eval_util.ten_passage_0shot_prompt,},
                        5: {0: eval_util.five_passage_0shot_prompt, 
                            1: eval_util.five_passage_1shot_prompt,
                            2: eval_util.five_passage_2shot_prompt,
                            3: eval_util.five_passage_3shot_prompt},
                        3: {0: eval_util.three_passage_0shot_prompt,},
                        1: {0: eval_util.one_passage_0shot_prompt,
                            1: eval_util.one_passage_1shot_prompt,
                            2: eval_util.one_passage_2shot_prompt,
                            3: eval_util.one_passage_3shot_prompt},
                        0: {0: eval_util.no_passage_prompt},
                        }


answers = []
total_time = 0
for passages, question, answer, qid in tqdm(pqa_list, desc="Generating Answers"): 
    # In here we can show if the ground truth passage is retrieved or not, etc. 
    if args.use_gt_passage == 'y': 
        messages = eval_util.one_passage_0shot_prompt(passages, question)
    else:
        messages = prompt_func_selector[args.top_k][args.num_shot](passages, question)
    start = time.time()
    generated_output = generator(messages, max_new_tokens=args.new_token, return_full_text=False)
    end = time.time()
    total_time += (end - start)
    answer_gen = generated_output[0]["generated_text"]
    if "assistant\n" in answer_gen:
        answer_gen = answer_gen.split("assistant")[-1]
    print("MESSAGES:", messages)
    print("GENERATED ANSWER:", answer_gen)
    print("TRUE ANSWER:", answer)
    print("=========================================================")
    
    if isinstance(answer, str):
        alist = [eval_util.normalize_answer(answer)]
    elif isinstance(answer, list):
        alist = [eval_util.normalize_answer(ans) for ans in answer]
    else:
        raise ValueError(f"Unexpected answer format: {type(answer)}")
    g = eval_util.normalize_answer(answer_gen)
    answers.append((alist, g, qid))

eval_util.evaluate(answers, output_file_name)
avg_time_ms = (total_time / len(pqa_list)) * 1000
print(f"Average time per question: {avg_time_ms:.2f} milliseconds")
