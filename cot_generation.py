import os
import re
import json
import argparse
import random
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from easyinstruct.utils.api import set_openai_key, set_proxy
from easyinstruct.utils.log import setup_logger
from easyinstruct.prompts import FewshotCoTPrompt

from utils import *


set_openai_key("")
set_proxy("http://127.0.0.1:7890")


prompt_cot = """

Given the question (and the context), select the answer from the options ["A", "B", "C", "D", "E"]. You should give consice and step-by-step solutions. Finally, conclude the answer in the format of "the answer is [ANSWER]", where [ANSWER] is one from the options ["A", "B", "C", "D", "E"]. For example, "the answer is A", "the answer is B", "the answer is C", "the answer is D", or "the answer is E". If the answer is not in the options, select the most possible option.
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

    # pick up shot examples from the training set
    shot_qids = args.shot_qids
    train_qids = pid_splits["train"]
    if shot_qids == None:
        assert args.n_shots >= 0 and args.n_shots <= 32
        shot_qids = random.sample(train_qids, args.n_shots)  # random sample
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            assert qid in train_qids  # check shot_qids
    print("training question ids for prompting: ", shot_qids, "\n")

    return problems, qids, shot_qids


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=2, max=120))
def get_instruct_result(problems, shot_qids, test_qid, args):
    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_visual_clues)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(
            args.prompt_format,
            question,
            context,
            choice,
            answer,
            lecture,
            solution,
            test_example=False,
        )
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_visual_clues)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(
        args.prompt_format,
        question,
        context,
        choice,
        answer,
        lecture,
        solution,
        test_example=True,
    )

    # create the prompt input
    prompt_input = FewshotCoTPrompt()
    prompt_input.build_prompt(
        prompt=test_example, in_context_examples=examples, n_shots=args.n_shots
    )

    prompt_input.prompt = prompt_cot + prompt_input.prompt

    if args.debug:
        print(prompt_input.prompt)

    # generate prediction
    output = prompt_input.get_openai_result(
        engine=args.engine,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )

    # extract the answer
    pattern = re.compile(r"[Tt]he answer is ([A-Z])")
    res = pattern.findall(output)
    if len(res) > 0:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"

    return answer, output


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[: len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


def get_result_path_and_file(args):
    result_path = os.path.join(
        args.output_root, args.model, args.test_split, args.prompt_format, os.path.basename(args.visual_clues_file), "paths_{}".format(args.n_paths)
    )
    result_file = os.path.join(
        result_path, "{}shots_seed_{}.json".format(args.n_shots, args.seed)
    )

    return result_path, result_file


def save_results(result_file, acc, correct, count, shot_qids, args, results, outputs):
    data = {}
    data["acc"] = acc
    data["correct"] = correct
    data["count"] = count
    data["shot_qids"] = shot_qids
    data["args"] = vars(args)
    data["results"] = results
    data["outputs"] = outputs

    with open(result_file, "w") as f:
        json.dump(data, f, indent=2, separators=(",", ": "))


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
        "--n_paths", type=int, default=1, help="Number of reasoning paths"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save the result with every n examples.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="CQM-A",
        choices=[
            "CQM-A",
            "CQM-LA",
            "CQM-EA",
            "CQM-LEA",
            "CQM-ELA",
            "CQM-AL",
            "CQM-AE",
            "CQM-ALE",
            "QCM-A",
            "QCM-LA",
            "QCM-EA",
            "QCM-LEA",
            "QCM-ELA",
            "QCM-AL",
            "QCM-AE",
            "QCM-ALE",
            "QCML-A",
            "QCME-A",
            "QCMLE-A",
            "QCLM-A",
            "QCEM-A",
            "QCLEM-A",
            "QCML-AE",
        ],
        help="prompt format template",
    )
    parser.add_argument(
        "--n_shots", type=int, default=2, help="Number of n-shot training examples."
    )
    parser.add_argument(
        "--shot_qids", type=list, default=None, help="Question indexes of shot examples"
    )
    parser.add_argument("--seed", type=int, default=10, help="random seed")
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
    result_path, result_file = get_result_path_and_file(args)
    setup_logger(result_path)

    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    if args.multiple_api_keys:
        api_list = json.load(open("openai_keys.json"))

    problems, qids, shot_qids = load_data(
        args
    )  # probelms, test question ids, shot example ids

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(result_file))

        acc = check_point["acc"]
        correct = check_point["correct"]
        count = check_point["count"]
        results = check_point["results"]
        outputs = check_point["outputs"]

        if count >= len(qids):
            print("## The result file is complete! We will exit!!!")
            print(
                f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%"
            )
            exit()
        else:
            print("## The result file is not complete! We will continue!!!")
            qids = qids[count:]
    else:
        correct = 0
        count = 0
        results = {}
        outputs = {}
    
    for _ in range(args.n_paths):
        for i, qid in enumerate(tqdm(qids)):

            if args.multiple_api_keys:
                set_openai_key(api_list[i % len(api_list)])

            choices = problems[qid]["choices"]
            answer = problems[qid]["answer"]  # 0, 1, ..., 4
            label = args.options[answer]  # 'A', ..., 'E'

            prediction, output = get_instruct_result(problems, shot_qids, qid, args)

            pred_idx = get_pred_idx(prediction, choices, args.options)  # 0, 1, ..., 4

            if qid not in results:
                results[qid] = [pred_idx]
            else:
                results[qid].append(pred_idx)

            if qid not in outputs:
                outputs[qid] = [output]
            else:
                outputs[qid].append(output)

            if max(results[qid], key=results[qid].count) == answer:
                correct += 1

            acc = correct / (count + i + 1) * 100

            if args.debug or i < 3:
                print("# labeled answer:", label)
                print("# predicted answer:", prediction)
                print("# predicted index:", pred_idx)
                print("# predicted output:", output)
                print("\n######################################################\n")

            if (i + 1) % args.save_every == 0 or (i + 1) == len(qids):
                print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
                save_results(result_file, acc, correct, count + i + 1, shot_qids, args, results, outputs)
