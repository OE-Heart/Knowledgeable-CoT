import re
import json
import argparse
import warnings
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import util, SentenceTransformer

warnings.filterwarnings("ignore")


def extract_explanation(text):
    text = re.sub(r"The answer is [A-Z]. BECAUSE: ", "", text)
    return text


########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def caculate_bleu(results, data, gram):
    bleus = []
    for qid, outputs in results.items():
        bleu = []
        for output in outputs:
            prediction = re.sub(r"The answer is [A-Z]. BECAUSE: ", "", output)
            target = data[qid]["lecture"] + " " + data[qid]["solution"]
            target = target.strip()
            if prediction == "" or target == "":
                if len(bleu) == 0:
                    bleu.append(0)
                bleu.append(sum(bleu) / len(bleu))
            else:
                bleu.append(bleu_score(target, prediction, gram))
        bleus.append(sum(bleu) / len(bleu))

    avg_bleu = sum(bleus) / len(bleus)
    return avg_bleu


########################
## Rouge-L
########################
def rouge_score(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


def caculate_rouge(results, data):
    rouges = []
    for qid, outputs in results.items():
        rouge = []
        for output in outputs:
            prediction = re.sub(r"The answer is [A-Z]. BECAUSE: ", "", output)
            target = data[qid]["lecture"] + " " + data[qid]["solution"]
            target = target.strip()
            if prediction == "" or target == "":
                if len(rouge) == 0:
                    rouge.append(0)
                rouge.append(sum(rouge) / len(rouge))
            else:
                rouge.append(rouge_score(target, prediction))
        rouges.append(sum(rouge) / len(rouge))

    avg_rouge = sum(rouges) / len(rouges)
    return avg_rouge


########################
## Sentence Similarity
########################
def similariry_score(str1, str2, model):
    # compute embedding for both lists
    embedding_1 = model.encode(str1, convert_to_tensor=True)
    embedding_2 = model.encode(str2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    return score


def caculate_similariry(results, data, model):
    scores = []
    for qid, outputs in results.items():
        score = []
        for output in outputs:
            prediction = re.sub(r"The answer is [A-Z]. BECAUSE: ", "", output)
            target = data[qid]["lecture"] + " " + data[qid]["solution"]
            target = target.strip()
            if prediction == "" or target == "":
                if len(score) == 0:
                    score.append(0)
                score.append(sum(score) / len(score))
            else:
                score.append(similariry_score(target, prediction, model))
        scores.append(sum(score) / len(score))

    avg_score = sum(scores) / len(scores)
    return avg_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default="data/scienceqa/problems.json")
    parser.add_argument('--result_file', type=str, default="results/chatgpt/test/QCM-ALE/visual_clues.json/2shots_seed_1.json")
    args = parser.parse_args()

    data = json.load(open(args.data_file))
    results = json.load(open(args.result_file))["outputs"]

    print("Data file: ", args.data_file)
    print("Result file: ", args.result_file)

    ## BLEU
    bleu1 = caculate_bleu(results, data, gram=1)
    bleu4 = caculate_bleu(results, data, gram=4)
    print("BLEU-1: %.3f" % bleu1)
    print("BLEU-4: %.3f" % bleu4)

    ## Rouge-L
    rouge = caculate_rouge(results, data)
    print("ROUGE-L: %.3f" % rouge)

    ## Similarity
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
    similariry = caculate_similariry(results, data, model)
    print("Similariry: %.3f" % similariry)
