from pyserini.search.lucene import LuceneSearcher
import os
import json
import spacy
from tqdm import tqdm
from utils import *


def retrieval(query):
    hits = ssearcher.search(query, 10)
    # for i in range(0, 10):
    #     print(f'{i + 1:2} {hits[i].docid:7} {hits[i].score:.5f}')
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
    nlp = spacy.load("en_core_web_md")
    ssearcher = LuceneSearcher.from_prebuilt_index(knowledge_store)

    ## Retrieve external knowledge
    external_knowledge = {}

    print(f"Retrieving external knowledge!")
    for pid in tqdm(problems):
        problems[pid]["visual_clues"] = visual_clues[pid] if pid in visual_clues else ""
        
        question = get_question_text(problems[pid])
        hint = get_hint_text(problems[pid])
        visual_clues = get_visual_clues(problems[pid])

        query = " ".join([question, hint, visual_clues]).strip()

        import ipdb; ipdb.set_trace()

        nlp_text = nlp(query).sents
        sent_pages = []
        for sent in nlp_text:
            sent = str(sent)
            try:
                paragraphs = retrieval(sent)
                external_knowledge[pid] = paragraphs
                sent_pages.append((sent, paragraphs))
                external_knowledge[pid] = sent_pages
                # print(external_knowledge[pid])
            except Exception as e:
                print(sent)
                print(e)

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
