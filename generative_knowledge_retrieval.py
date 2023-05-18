import os
import json
import argparse
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from easyinstruct.utils.api import set_openai_key, set_proxy
from easyinstruct.prompts import BasePrompt

from utils import *


set_openai_key("")
set_proxy("http://127.0.0.1:7890")


demo_prompt = """
Read the following question, generate the background knowledge as the context information that could be helpful for answering the question.

Question: Which property do these three objects have in common? 

Options: (A) hard (B) soft (C) yellow

Metadata: {'pid': 43, 'has_image': True, 'grade': 4, 'subject': 'natural science', 'topic': 'physics', 'category': 'Materials', 'skill': 'Compare properties of objects'} 

Detected text in the image: ['handkerchief', 'slippers', 'leisure suit']

Knowledge:
- This question is about comparing the properties of three objects: a handkerchief, slippers, and a leisure suit.
- The objects are related to the topic of physics and the skill of comparing properties of objects.
- Properties of objects can include physical characteristics such as color, texture, shape, size, weight, and material.

Question: The diagrams below show two pure samples of gas in identical closed, rigid containers. Each colored ball represents one gas particle. Both samples have the same number of particles. 

Options: (A) neither; the samples have the same temperature (B) sample A (C) sample B

Metadata: {'pid': 19, 'has_image': True, 'grade': 8, 'subject': 'natural science', 'topic': 'physics', 'category': 'Particle motion and energy', 'skill': 'Identify how particle motion affects temperature and pressure'}

Knowledge:
- The temperature of a substance depends on the average kinetic energy of the particles in the substance. 
- The higher the average kinetic energy of the particles, the higher the temperature of the substance.
- The kinetic energy of a particle is determined by its mass and speed. 
- For a pure substance, the greater the mass of each particle in the substance and the higher the average speed of the particles, the higher their average kinetic energy. 

Question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?

Context: The images below show two pairs of magnets. The magnets in different pairs do not affect each other. All the magnets shown are made of the same material, but some of them are different shapes.  

Options: (A) The magnitude of the magnetic force is greater in Pair 1. (B) The magnitude of the magnetic force is greater in Pair 2. (C) The magnitude of the magnetic force is the same in both pairs.

Metadata: {'pid': 270, 'has_image': True, 'grade': 6, 'subject': 'natural science', 'topic': 'physics', 'category': 'Velocity, acceleration, and forces', 'skill': 'Compare magnitudes of magnetic forces'}

Knowledge:
- Magnets can pull or push on each other without touching. When magnets attract, they pull together. When magnets repel, they push apart. These pulls and pushes between magnets are called magnetic forces.
- The strength of a force is called its magnitude. The greater the magnitude of the magnetic force between two magnets, the more strongly the magnets attract or repel each other.
- You can change the magnitude of a magnetic force between two magnets by changing the distance between them. The magnitude of the magnetic force is greater when there is a smaller distance between the magnets. 

Read the following question, generate the background knowledge as the context information that could be helpful for answering the question.
"""


def load_data(args):
    problems = json.load(open(os.path.join(args.data_root, "problems.json")))
    pid_splits = json.load(open(os.path.join(args.data_root, "pid_splits.json")))
    visual_clues = json.load(open(args.visual_clues_file))["visual_clues"]

    for qid in problems:
        problems[qid]["visual_clues"] = visual_clues[qid] if qid in visual_clues else ""

    qids = pid_splits["%s" % (args.test_split)]
    qids = qids[: args.test_number] if args.test_number > 0 else qids
    print(f"number of test problems: {len(qids)}\n")

    return problems, qids


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=2, max=120))
def generative_retieval(problems, test_qid, args):
    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_visual_clues)
    choice = get_choice_text(problems[test_qid], args.options)
    metadata = get_metadata(problems[test_qid])

    test_prompt = f"Question: {question}\nContext: {context}\nOptions: {choice}\n\nMetadata: {metadata}\n\nKnowledge:\n"

    # create the prompt input
    prompt_input = BasePrompt()
    prompt_input.build_prompt(demo_prompt+test_prompt)

    # generate prediction
    Knowledge = prompt_input.get_openai_result(
        engine=args.engine,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )

    return Knowledge

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/scienceqa")
    parser.add_argument("--output_root", type=str, default="results")
    parser.add_argument(
        "--visual_clues_file", type=str, default="data/visual_clues.json"
    )
    parser.add_argument("--model", type=str, default="chatgpt")
    parser.add_argument("--options", type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument(
        "--test_split", type=str, default="val", choices=["test", "val", "minival"]
    )
    parser.add_argument(
        "--test_number",
        type=int,
        default=10,
        help="GPT-3 is expensive. -1 for whole val/test set",
    )
    parser.add_argument(
        "--use_visual_clues", action="store_true", help="Use visual clues or not"
    )
    parser.add_argument(
        "--multiple_api_keys", action="store_true", help="Use multiple API keys"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save the result with every n examples.",
    )
    # GPT-3 settings
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="The maximum number of tokens allowed for the generated answer.",
    )
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    output_file = "data/knowledge.json"

    if args.multiple_api_keys:
        api_list = json.load(open("openai_keys.json"))

    problems, qids = load_data(args)

    results = {
        "args": vars(args),
        "knowledge": {}
    }

    for i, qid in enumerate(tqdm(qids)):

        if args.multiple_api_keys:
            set_openai_key(api_list[i % len(api_list)])

        choices = problems[qid]["choices"]
        answer = problems[qid]["answer"]  # 0, 1, ..., 4
        label = args.options[answer]  # 'A', ..., 'E'

        knowledge = generative_retieval(problems, qid, args)

        results["knowledge"][qid] = knowledge

        if (i + 1) % args.save_every == 0 or (i + 1) == len(qids):
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)