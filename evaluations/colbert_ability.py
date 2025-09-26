from ragatouille import RAGPretrainedModel
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm 
import argparse, logging, json, os, torch, eval_util
from transformers import pipeline

parser = argparse.ArgumentParser(description="Script for training a retriever")
parser.add_argument('--data', type=str, help='msmarco or nq or squad or trivia or wq')
parser.add_argument('--model_path', type=str, help='Path to the model that you want to evaluate')
parser.add_argument('--hf_checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--hf_token', type=str, help='Add your HuggingFace token here')
parser.add_argument('--new_token', type=int, help="Number of new token to allow llm to generate")
parser.add_argument('--index_name', type=str, help="Give the index a name, or try to load an existing index")
parser.add_argument('--top_k', type=int, help="Number of documents in the prompt")
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

# Either load index or create index from a trained ColBERT
path_to_index = f"/WAVE/users2/unix/jnian/WeakLabelForRAG/.ragatouille/colbert/indexes/{args.index_name}"
if os.path.exists(path_to_index):
    print(f"Loading index: {path_to_index}")
    RAG = RAGPretrainedModel.from_index(path_to_index)
else:
    RAG = RAGPretrainedModel.from_pretrained(args.model_path)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    all_documents = [f"{item['text']} {item['title']}" for item in corpus.values()]
    print(f"Building index: {args.index_name}")
    RAG.index(
        collection=all_documents, 
        index_name=args.index_name, 
        max_document_length=350, 
        use_faiss=True # doesn't use gpu for some reason. I heard if you create conda env and install faiss-gpu it works and it's a lot faster. 
        )


##########################################################
# k = 3 
# # ground truth passage should be: Cross-Country Skiing Burns More. Burning about 381 calories in 60 minutes of snowboarding provides a slower caloric burn than some other forms of winter exercise. A 160-pound person burns about 533 calories in an hour of slow-paced cross-country skiing and about 419 calories in 60 minutes of light ice skating. The caloric burn from light snowboarding is equivalent to that of light downhill skiing. Related Reading: How to Estimate the Total Calories You Burned While Running.
# colbert_results = RAG.search(query="how many calories does skiing virn", k=k) 
# print(colbert_results)
'''
colbert_results = [{'content': "the passage", 'score': 21.25, 'rank': 1, 'document_id': 'some hash', 'passage_id': 20450}]
'''
########################################################

'''
Prepare "results" into BEIR format and we are good to go 
results = {qid: {doc_id: score}, ...}
'''
def find_doc_id_by_text(corpus, search_text):
    return text_to_doc_id.get(search_text, None)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

text_to_doc_id = {item['text']: doc_id for doc_id, item in corpus.items()}

answers = eval_util.load_jsonl({}, f"{data_path}/answers.jsonl")
results_file = f"evaluations/colbert_retrieval_results/{args.index_name}_retrieval_resuls.json"
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    pqa_list = []
    for qid, rank_dict in results.items():
        query = queries[qid]
        answer = answers[qid]
        sorted_docs = sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)
        if sorted_docs:
            doc_list = []
            for i in range(args.top_k):
                doc_id = sorted_docs[i][0]
                doc_text = corpus[doc_id]['text']
                doc_list.append(doc_text)
            pqa_list.append((doc_list, query, answer, qid))
else:
    results, pqa_list = {}, []

    for qid, rel_doc in tqdm(qrels.items(), desc="ColBERT retrieving"): 
        results[qid] = {}
        query, answer = queries[qid], answers[qid]
        colbert_results = RAG.search(query=query, index_name=args.index_name, k=100)
        topk = args.top_k
        for doc_dict in colbert_results:
            doc, score = doc_dict['content'], doc_dict['score']
            doc_id = find_doc_id_by_text(corpus, doc)  # I know this is super slow. 
            if doc_id is None:
                continue 
            results[qid][doc_id] = score
            if topk > 0:
                pqa_list.append(([doc], query, answer, qid))        
                topk -= 1
    # Save results to a JSON file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

# Evaluate Retrieval ability
evaluator = EvaluateRetrieval(k_values=[1,3,5,10,100])
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, evaluator.k_values)
mrr = evaluator.evaluate_custom(qrels, results, evaluator.k_values, metric="mrr")

print(f"Retrieval results of {args.index_name}")
print(recall)
print(mrr)


# DOING QA and then evaluate 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = pipeline("text-generation",model=args.hf_checkpoint,device="cuda",torch_dtype=torch.float16,token=args.hf_token)
answers_tobe_eval = []

for passages, question, answer, qid in tqdm(pqa_list, desc="Generating Answers"):
    if args.top_k == 1:
        messages = eval_util.one_passage_0shot_prompt(passages, question)
    elif args.top_k == 5:
        messages = eval_util.five_passage_0shot_prompt(passages, question)
    generated_output = generator(messages, max_new_tokens=args.new_token, return_full_text=False)
    answer_gen = generated_output[0]["generated_text"]
    if "assistant\n" in answer_gen:
        answer_gen = answer_gen.split("assistant")[-1]
    print("MESSAGES:", messages)
    print("GENERATED ANSWER:", answer_gen)
    print("TRUE ANSWER:", answer)
    print("=========================================================")
    a, g = eval_util.normalize_answer(answer), eval_util.normalize_answer(answer_gen)
    answers_tobe_eval.append(([a], g, qid))

output_file_name = f"evaluations/qa/{args.data}/colbert_{args.index_name}"
eval_util.evaluate(answers_tobe_eval, output_file_name, args.new_token)