import re
import json
import argparse
from tqdm import tqdm
from sentence_transformers import util, SentenceTransformer

from cot_evaluation import bleu_score, rouge_score, similariry_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default="data/scienceqa/problems.json")
    parser.add_argument('--result_file', type=str, default="results/chatgpt/test/QCM-ALE/visual_clues_ocrs.json/paths_10/2shots_seed_3.json")
    parser.add_argument('--knowledge_file', type=str, default="data/knowledge.json")

    args = parser.parse_args()

    print("Data file: ", args.data_file)
    print("Result file: ", args.result_file)

    results = json.load(open(args.result_file))
    outputs = json.load(open(args.result_file))["outputs"]
    knowledge = json.load(open(args.knowledge_file))["knowledge"]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()

    results["bleu1s"] = {}
    results["bleu4s"] = {}
    results["rouges"] = {}
    results["similarities"] = {}

    for qid, output in tqdm(outputs.items()):
        bleu1s = []
        bleu4s = []
        rouges = []
        similarities = []

        target = re.sub('-', '', knowledge[qid]).strip()

        for reasoning_path in output:
            prediction = re.sub(r"The answer is [A-Z]. BECAUSE: ", "", reasoning_path)

            bleu1 = bleu_score(target, prediction, 1)
            bleu4 = bleu_score(target, prediction, 4)
            rouge = rouge_score(target, prediction)
            similarity = similariry_score(target, prediction, model)

            bleu1s.append(bleu1)
            bleu4s.append(bleu4)
            rouges.append(rouge)
            similarities.append(similarity)
        
        results["bleu1s"][qid] = bleu1s
        results["bleu4s"][qid] = bleu4s
        results["rouges"][qid] = rouges
        results["similarities"][qid] = similarities

    with open(args.result_file, "w") as f:
        json.dump(results, f, indent=2)