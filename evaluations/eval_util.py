import os, torch, json, logging, pickle, datetime 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # pipeline,
)
import nltk
import numpy as np 
from rouge import Rouge
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from scipy.stats import ttest_ind
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from nltk.tokenize import word_tokenize

nltk.download('punkt')
logging.getLogger("transformers").setLevel(logging.ERROR)

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

def load_or_create_bm25_results(data_path, corpus, qrels, queries, bm25_topk=100):
    results_file = os.path.join(data_path, f'bm25_top{bm25_topk}_on_test.json')

    if os.path.isfile(results_file):
        print("Load existing results from the JSON file", results_file)
        with open(results_file, 'r') as f:
            results = json.load(f)
    else: 
        print("No Results file found, creating one right now. ")
        results = {}
        
        bm25 = load_or_create_bm25(corpus, f"{data_path}/bm25_{len(corpus)}.pkl")
        
        for query_id, relevant_docs in tqdm(qrels.items(), desc="BM25 retrieving"):
            query = queries[query_id]
            scores = bm25.get_scores(query)
            ranked_indices = np.argsort(scores)[::-1][:bm25_topk]
            results[query_id] = {idx: float(scores[idx]) for idx in ranked_indices}
            
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print("BM25 Results saved at", results_file)

    return results

def bm25_retrieve(bm25, corpus, question, top_k):
    question_tokens = word_tokenize(question.lower())
    passage_scores = bm25.get_scores(question_tokens)
    sorted_indices = np.argsort(passage_scores)[::-1][:top_k]
    # top_passages_with_scores = [(list(corpus.keys())[idx], corpus[list(corpus.keys())[idx]], passage_scores[idx]) for idx in sorted_indices]
    top_passages_with_scores = [(idx, corpus[idx], passage_scores[idx]) for idx in sorted_indices]
    top_passages = [corpus[pid] for pid, _, score in top_passages_with_scores]
    return top_passages

def load_jsonl(data_dict, file_path): 
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            data_dict[data['_id']] = data['text'] 
    return data_dict 

def load_jsonl_int_key(data_dict, file_path): 
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            data_dict[int(data['_id'])] = data['text'] # cast "1183942" into 1183942 as the keys 
    return data_dict 
    
# Function to save embeddings
def save_embeddings(embeddings, file_path):
    torch.save(embeddings, file_path)
    print(f"Embeddings saved to {file_path}")

# Function to load embeddings
def load_embeddings(file_path):
    embeddings = torch.load(file_path)
    print(f"Embeddings loaded from {file_path}")
    return embeddings

def test():
    print("Hi I am eval util")

def no_passage_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Question: {question}"},
        {"role": "assistant", "content": ""}
    ]
    return messages 

def one_passage_0shot_prompt(passages, question): # use "passages" just for consistency, there should only be 1 sentence in this list
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: {passages[0]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""}
    ]
    return messages 

# def five_passage_0shot_prompt(passages, question): # This is old 
#     prompt = f'''DOCUMENT: {passages[0]}
# DOCUMENT: {passages[1]}
# DOCUMENT: {passages[2]}
# DOCUMENT: {passages[3]}
# DOCUMENT: {passages[4]}
# QUESTION: {question}
# The passages above may or may not be relevant to the question. If you find any one of them relevant to the question, then please use the text in the passage to answer. Otherwise, use your own knowledge to answer.
# Keep the answer within one short sentence.
# ANSWER: '''
#     return prompt 

def three_passage_0shot_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: {passages[0]}\nPassage: {passages[1]}\nPassage: {passages[2]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""}
    ]
    return messages 

def five_passage_0shot_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: {passages[0]}\nPassage: {passages[1]}\nPassage: {passages[2]}\nPassage: {passages[3]}\nPassage: {passages[4]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""}
    ]
    return messages 

def ten_passage_0shot_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: {passages[0]}\nPassage: {passages[1]}\nPassage: {passages[2]}\nPassage: {passages[3]}\nPassage: {passages[4]}\nPassage: {passages[5]}\nPassage: {passages[6]}\nPassage: {passages[7]}\nPassage: {passages[8]}\nPassage: {passages[9]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""}
    ]
    return messages

def twenty_passage_0shot_prompt(passages, question):
    passage_block = "\n".join([f"Passage: {p}" for p in passages])
    user_content = f"{passage_block}\nQuestion: {question}"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions using the provided passages. "
                "If the passages do not contain the necessary information, you will answer the question directly "
                "based on your general knowledge. Only provide the final answer, no explanations."
            )
        },
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": ""}
    ]
    return messages

def one_passage_1shot_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: September Weather in Paris. – min temperature : 11°C / 52°F. – max temperature : 21°C / 70°F. – average rainfall : 54 mm. Although the Parisian weather can be quite variable in September, it remains pleasant for the most part. September is generally a sunny month with one of the lowest average rainfalls in the year.\nQuestion: average temperature in paris in september in fahrenheit"},
        {"role": "assistant", "content": "Min temperature: 52°F and max temperature: 70°F."},
        {"role": "user", "content": f"Passage: {passages[0]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""},
    ]
    return messages

def one_passage_2shot_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: September Weather in Paris. – min temperature : 11°C / 52°F. – max temperature : 21°C / 70°F. – average rainfall : 54 mm. Although the Parisian weather can be quite variable in September, it remains pleasant for the most part. September is generally a sunny month with one of the lowest average rainfalls in the year.\nQuestion: average temperature in paris in september in fahrenheit"},
        {"role": "assistant", "content": "Min temperature: 52°F and max temperature: 70°F."},
        {"role": "user", "content": f"Passage: If you have arthritis, you may have considered a cortisone shot as part of your treatment plan. These shots are not pain relievers. Cortisone is a type of steroid, a drug that lowers inflammation, which is something that can lead to less pain. Cortisone injections can be used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon. They can also treat inflammation that is widespread throughout the body, such as with allergic reactions, asthma, and rheumatoid arthritis, which affects many joints.\nQuestion: what is a cortisone injection used for"},
        {"role": "assistant", "content": "Used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon."},
        {"role": "user", "content": f"Passage: {passages[0]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""},
    ]
    return messages

def one_passage_3shot_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: September Weather in Paris. – min temperature : 11°C / 52°F. – max temperature : 21°C / 70°F. – average rainfall : 54 mm. Although the Parisian weather can be quite variable in September, it remains pleasant for the most part. September is generally a sunny month with one of the lowest average rainfalls in the year.\nQuestion: average temperature in paris in september in fahrenheit"},
        {"role": "assistant", "content": "Min temperature: 52°F and max temperature: 70°F."},
        {"role": "user", "content": f"Passage: Big Bus Tours London. The hop-on, hop-off bus tour of London includes a cruise along the River Thames; a selection of guided walking tours and a Big Bus voucher booklet that offers a range of discounts at attractions, shops and restaurants.ur hop-on, hop-off bus tours of London allow you to explore the sights at your own pace. There are more than 50 locations where you can get off the bus to visit attractions or explore places of interest.\nQuestion: does the london hop on off bus price inlcude the cruise"},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": f"Passage: If you have arthritis, you may have considered a cortisone shot as part of your treatment plan. These shots are not pain relievers. Cortisone is a type of steroid, a drug that lowers inflammation, which is something that can lead to less pain. Cortisone injections can be used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon. They can also treat inflammation that is widespread throughout the body, such as with allergic reactions, asthma, and rheumatoid arthritis, which affects many joints.\nQuestion: what is a cortisone injection used for"},
        {"role": "assistant", "content": "Used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon."},
        {"role": "user", "content": f"Passage: {passages[0]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""},
    ]
    return messages 

def five_passage_1shot_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: September Weather in Paris. – min temperature : 11°C / 52°F. – max temperature : 21°C / 70°F. – average rainfall : 54 mm. Although the Parisian weather can be quite variable in September, it remains pleasant for the most part. September is generally a sunny month with one of the lowest average rainfalls in the year.\nQuestion: average temperature in paris in september in fahrenheit"},
        {"role": "assistant", "content": "Min temperature: 52°F and max temperature: 70°F."},
        {"role": "user", "content": f"Passage: {passages[0]}\nPassage: {passages[1]}\nPassage: {passages[2]}\nPassage: {passages[3]}\nPassage: {passages[4]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""},
    ]
    return messages 

def five_passage_2shot_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: September Weather in Paris. – min temperature : 11°C / 52°F. – max temperature : 21°C / 70°F. – average rainfall : 54 mm. Although the Parisian weather can be quite variable in September, it remains pleasant for the most part. September is generally a sunny month with one of the lowest average rainfalls in the year.\nQuestion: average temperature in paris in september in fahrenheit"},
        {"role": "assistant", "content": "Min temperature: 52°F and max temperature: 70°F."},
        {"role": "user", "content": f"Passage: If you have arthritis, you may have considered a cortisone shot as part of your treatment plan. These shots are not pain relievers. Cortisone is a type of steroid, a drug that lowers inflammation, which is something that can lead to less pain. Cortisone injections can be used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon. They can also treat inflammation that is widespread throughout the body, such as with allergic reactions, asthma, and rheumatoid arthritis, which affects many joints.\nQuestion: what is a cortisone injection used for"},
        {"role": "assistant", "content": "Used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon."},
        {"role": "user", "content": f"Passage: {passages[0]}\nPassage: {passages[1]}\nPassage: {passages[2]}\nPassage: {passages[3]}\nPassage: {passages[4]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""},
    ]
    return messages 

def five_passage_3shot_prompt(passages, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using the provided passage. If the passage does not contain the necessary information, you will answer the question directly based on your general knowledge. Only provide the final answer, no explanations"},
        {"role": "user", "content": f"Passage: September Weather in Paris. – min temperature : 11°C / 52°F. – max temperature : 21°C / 70°F. – average rainfall : 54 mm. Although the Parisian weather can be quite variable in September, it remains pleasant for the most part. September is generally a sunny month with one of the lowest average rainfalls in the year.\nQuestion: average temperature in paris in september in fahrenheit"},
        {"role": "assistant", "content": "Min temperature: 52°F and max temperature: 70°F."},
        {"role": "user", "content": f"Passage: Big Bus Tours London. The hop-on, hop-off bus tour of London includes a cruise along the River Thames; a selection of guided walking tours and a Big Bus voucher booklet that offers a range of discounts at attractions, shops and restaurants.ur hop-on, hop-off bus tours of London allow you to explore the sights at your own pace. There are more than 50 locations where you can get off the bus to visit attractions or explore places of interest.\nQuestion: does the london hop on off bus price inlcude the cruise"},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": f"Passage: If you have arthritis, you may have considered a cortisone shot as part of your treatment plan. These shots are not pain relievers. Cortisone is a type of steroid, a drug that lowers inflammation, which is something that can lead to less pain. Cortisone injections can be used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon. They can also treat inflammation that is widespread throughout the body, such as with allergic reactions, asthma, and rheumatoid arthritis, which affects many joints.\nQuestion: what is a cortisone injection used for"},
        {"role": "assistant", "content": "Used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon."},
        {"role": "user", "content": f"Passage: {passages[0]}\nPassage: {passages[1]}\nPassage: {passages[2]}\nPassage: {passages[3]}\nPassage: {passages[4]}\nQuestion: {question}"},
        {"role": "assistant", "content": ""},
    ]
    return messages 

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
        if llm_name == "phi3":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, # if using 16 bit
                # quantization_config=quant_config, # if using 4 bit
                # device_map={"": 0}, # more advanced way to map different layers to different device
                trust_remote_code=True,
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

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re, string
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punct(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punct(lower(s)))

def calculate_bleu(reference, candidate, n_grams=4):
    """ Calculate BLEU score for the candidate sentence using the reference sentence. """
    smoothing_function = SmoothingFunction().method1
    weights = {
        1: (1.0, 0, 0, 0),  # BLEU-1
        2: (0.5, 0.5, 0, 0),  # BLEU-2 (not requested but shown for example)
        4: (0.25, 0.25, 0.25, 0.25)  # BLEU
    }
    return sentence_bleu([word_tokenize(reference)], word_tokenize(candidate), weights=weights[n_grams], smoothing_function=smoothing_function)

def calculate_f1(precision, recall):
    """ Calculate F1 score given precision and recall. """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_against_no_passage(answers):
    rouge = Rouge()
    answer_gen_scores = {'f1': [], 'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu': [], 'bleu-1': [], 'meteor': []}
    blind_answer_scores = {'f1': [], 'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu': [], 'bleu-1': [], 'meteor': []}

    for answer, answer_gen, blind_answer in answers:
        # Calculate scores for answer_gen
        scores = rouge.get_scores(answer_gen, answer)[0]
        answer_gen_scores['rouge-1'].append(scores['rouge-1']['f'])
        answer_gen_scores['rouge-2'].append(scores['rouge-2']['f'])
        answer_gen_scores['rouge-l'].append(scores['rouge-l']['f'])
        answer_gen_scores['meteor'].append(meteor_score([word_tokenize(answer)], word_tokenize(answer_gen)))
        answer_gen_scores['bleu'].append(calculate_bleu(answer, answer_gen))
        answer_gen_scores['bleu-1'].append(calculate_bleu(answer, answer_gen, n_grams=1))
        f1_gen = calculate_f1(scores['rouge-1']['p'], scores['rouge-1']['r'])
        answer_gen_scores['f1'].append(f1_gen)

        # Calculate scores for blind_answer
        scores = rouge.get_scores(blind_answer, answer)[0]
        blind_answer_scores['rouge-1'].append(scores['rouge-1']['f'])
        blind_answer_scores['rouge-2'].append(scores['rouge-2']['f'])
        blind_answer_scores['rouge-l'].append(scores['rouge-l']['f'])
        blind_answer_scores['meteor'].append(meteor_score([word_tokenize(answer)], word_tokenize(blind_answer)))
        blind_answer_scores['bleu'].append(calculate_bleu(answer, blind_answer))
        blind_answer_scores['bleu-1'].append(calculate_bleu(answer, blind_answer, n_grams=1))
        f1_blind = calculate_f1(scores['rouge-1']['p'], scores['rouge-1']['r'])
        blind_answer_scores['f1'].append(f1_blind)

    # After loop, calculate average scores and statistical significance
    for metric in answer_gen_scores:
        gen_avg = sum(answer_gen_scores[metric]) / len(answer_gen_scores[metric])
        blind_avg = sum(blind_answer_scores[metric]) / len(blind_answer_scores[metric])
        print(f"{metric} - Answer Gen Avg: {gen_avg}, Blind Answer Avg: {blind_avg}")

        # T-test for significance
        if answer_gen_scores[metric] and blind_answer_scores[metric]:
            t_stat, p_val = ttest_ind(answer_gen_scores[metric], blind_answer_scores[metric])
            print(f"{metric} - T-test result: T-Stat={t_stat}, P-Value={p_val}")
        else:
            print(f"{metric} - Insufficient data for T-test")
        print()
     
     
# # old code, kept here just in case   
# def evaluate(answers):
#     rouge = Rouge()
#     answer_gen_scores = {'f1': [], 'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu': [], 'bleu-1': [], 'meteor': []}

#     for answer, answer_gen, qid in answers:
#         # Calculate scores for answer_gen
#         scores = rouge.get_scores(answer_gen, answer)[0]
#         answer_gen_scores['rouge-1'].append(scores['rouge-1']['f'])
#         answer_gen_scores['rouge-2'].append(scores['rouge-2']['f'])
#         answer_gen_scores['rouge-l'].append(scores['rouge-l']['f'])
#         answer_gen_scores['meteor'].append(meteor_score([word_tokenize(answer)], word_tokenize(answer_gen)))
#         answer_gen_scores['bleu'].append(calculate_bleu(answer, answer_gen))
#         answer_gen_scores['bleu-1'].append(calculate_bleu(answer, answer_gen, n_grams=1))
#         f1_gen = calculate_f1(scores['rouge-1']['p'], scores['rouge-1']['r'])
#         answer_gen_scores['f1'].append(f1_gen)

#     # After loop, calculate average scores and statistical significance
#     for metric in answer_gen_scores:
#         gen_avg = sum(answer_gen_scores[metric]) / len(answer_gen_scores[metric])
#         print(f"{metric}: {gen_avg}")


def evaluate(answers, file_name=None, token_num=None):
    rouge = Rouge()
    results = {}
    bad = 0
    for answer_list, answer_gen, qid in answers:
        best_scores = {
            'f1': 0,
            'rouge-1': 0,
            'rouge-2': 0,
            'rouge-l': 0,
            'bleu': 0,
            'bleu-1': 0,
            'meteor': 0,
            'sub_em': 0,
            'em': 0,
        }
        evaluated_at_least_once = False 
        print("Answer:", answer_list)
        print("Generated Answer:", answer_gen)
        print("==========================================================")
        for answer in answer_list:
            if answer is None or len(answer) == 0 or len(answer_gen) == 0:
                bad = bad + 1
                continue 
            
            scores = rouge.get_scores(answer_gen, answer)[0]
            f1_gen = calculate_f1(scores['rouge-1']['p'], scores['rouge-1']['r'])
            
            bleu_score = calculate_bleu(answer, answer_gen)
            bleu_1_score = calculate_bleu(answer, answer_gen, n_grams=1)
            meteor = meteor_score([word_tokenize(answer)], word_tokenize(answer_gen))
            sub_em = 1 if answer in answer_gen else 0
            em = 1 if answer == answer_gen else 0
            # Update best scores if current scores are higher
            best_scores['f1'] = max(best_scores['f1'], f1_gen)
            best_scores['em'] = max(best_scores['em'], em)
            best_scores['bleu-1'] = max(best_scores['bleu-1'], bleu_1_score)
            best_scores['rouge-1'] = max(best_scores['rouge-1'], scores['rouge-1']['f'])
            best_scores['rouge-2'] = max(best_scores['rouge-2'], scores['rouge-2']['f'])
            best_scores['rouge-l'] = max(best_scores['rouge-l'], scores['rouge-l']['f'])
            best_scores['bleu'] = max(best_scores['bleu'], bleu_score)
            best_scores['meteor'] = max(best_scores['meteor'], meteor)
            best_scores['sub_em'] = max(best_scores['sub_em'], sub_em)
            
            evaluated_at_least_once = True 
        if evaluated_at_least_once:
            results[qid] = best_scores

    avg_scores = {metric: sum(scores[metric] for scores in results.values()) / len(results) for metric in results[next(iter(results))]}
    results["avg"] = avg_scores

    for metric, avg_score in avg_scores.items():
        print(f"{metric}: {avg_score}")
    
    
    if file_name is not None and file_name != "":
        current_time = datetime.datetime.now()
        time_string = current_time.strftime('%m_%d_%H:%M')
        with open(f"{file_name}__{time_string}.json", 'w') as f:
            json.dump(results, f, indent=4)
        print(f"{bad} answers are bad")
        print(f"Individual results written to {file_name}__{time_string}.json")
    