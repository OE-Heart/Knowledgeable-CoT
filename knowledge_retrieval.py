import os
import re
import json
import spacy
import argparse
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

from utils import *


def retrieval(query, k=5):
    hits = ssearcher.search(query, k)
    paragraphs = []
    for i in range(len(hits)):
        doc = ssearcher.doc(hits[i].docid)
        json_doc = json.loads(doc.raw())
        paragraphs.append(json_doc["contents"])
    return paragraphs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/scienceqa/problems.json")
    parser.add_argument(
        "--result_file",
        type=str,
        default="results/chatgpt/test/QCM-ALE/visual_clues.json/paths_10/2shots_seed_3.json",
    )
    parser.add_argument("--knowledge_store", type=str, default="wikipedia-dpr")
    parser.add_argument(
        "--knowledge_store_path",
        type=str,
        default="indexes/index-wikipedia-dpr-20210120-d1b9e6",
    )
    parser.add_argument("--k_hits", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    results = json.load(open(args.result_file))["results"]
    outputs = json.load(open(args.result_file))["outputs"]

    # ssearcher = LuceneSearcher.from_prebuilt_index(args.knowledge_store)
    ssearcher = LuceneSearcher(args.knowledge_store_path)

    nlp = spacy.load("en_core_web_md")

    output_path = os.path.dirname(args.result_file)
    output_name = os.path.basename(args.result_file).replace(
        ".json", "_external_knowledge.json"
    )
    output_file = os.path.join(output_path, output_name)

    # load the check point
    if os.path.exists(output_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(output_file))

        external_knowledge = check_point["external_knowledge"]
        if len(external_knowledge) >= len(results):
            print("## The result file is complete! We will exit!!!")
            exit()
        else:
            print("## The result file is not complete! We will continue!!!")
    else:
        external_knowledge = {}

    ## Retrieve external knowledge
    print("Retrieving external knowledge!")
    for i, pid in enumerate(tqdm(outputs)):
        if pid in external_knowledge:
            print(f"## The external knowledge of {pid} is already retrieved!")
            continue

        external_knowledge[pid] = []
        for reasoning_path in outputs[pid]:
            explanation = re.sub(r"The answer is [A-Z]. BECAUSE: ", "", reasoning_path)
            nlp_text = nlp(explanation).sents
            sent_paragraphs = []
            for sent in nlp_text:
                sent = str(sent)
                try:
                    paragraphs = retrieval(sent, args.k_hits)
                    sent_paragraphs.append((sent, paragraphs))
                except Exception as e:
                    print(sent)
                    print(e)
            external_knowledge[pid].append(sent_paragraphs)

        ## Save the external knowledge
        if (i + 1) % args.save_every == 0 or (i + 1) == len(outputs):
            print(f"Saved to {output_file}")

            knowledge = {
                "knowledge_store": args.knowledge_store,
                "external_knowledge": external_knowledge,
            }

            with open(output_file, "w") as f:
                json.dump(knowledge, f, indent=2, separators=(",", ": "))
