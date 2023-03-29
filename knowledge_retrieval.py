from pyserini.search.lucene import LuceneSearcher
import os
import json
from tqdm import tqdm
from utils import *


def retrieval(query):
    hits = ssearcher.search(query, 5)
    paragraphs = []
    for i in range(len(hits)):
        doc = ssearcher.doc(hits[i].docid)
        json_doc = json.loads(doc.raw())
        paragraphs.append(json_doc['contents'])
    return paragraphs


if __name__ == "__main__":
    input_path = "data/scienceqa"
    output_path = "data"
    output_name = "knowledge.json"

    problems = json.load(open(os.path.join(input_path, 'problems.json')))
    pid_splits = json.load(open(os.path.join(input_path, 'pid_splits.json')))
    visual_clues = json.load(open(os.path.join(output_path, 'visual_clues.json')))["visual_clues"]

    knowledge_store = "wikipedia-dpr"
    # ssearcher = LuceneSearcher.from_prebuilt_index(knowledge_store)

    knowledge_store_path = "indexes/index-wikipedia-dpr-20210120-d1b9e6"
    ssearcher = LuceneSearcher(knowledge_store_path)
    
    ## Retrieve external knowledge
    external_knowledge = {}

    print(f"Retrieving external knowledge!")
    for pid in tqdm(problems):
        problems[pid]["visual_clues"] = visual_clues[pid] if pid in visual_clues else ""
        
        question = get_question_text(problems[pid])
        hint = get_hint_text(problems[pid])
        visual_clues = get_visual_clues(problems[pid])

        sent_paragraphs = []
        for sent in [question, hint, visual_clues]:
            sent = str(sent)
            if sent == "": 
                continue
            try:
                paragraphs = retrieval(sent)
                sent_paragraphs.append((sent, paragraphs))
            except Exception as e:
                print(sent)
                print(e)
        external_knowledge[pid] = sent_paragraphs
        # print(external_knowledge[pid])

    ## Save the external knowledge
    output_file = os.path.join(output_path, output_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"Saved to {output_file}")

    results = {
        "knowledge_store": knowledge_store,
        "external_knowledge": external_knowledge
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, separators=(',', ': '))
