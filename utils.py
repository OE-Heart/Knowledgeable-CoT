def get_question_text(problem):
    question = problem["question"]
    return question


def get_visual_clues(problem):
    visual_clues = problem["visual_clues"]
    return visual_clues


def get_hint_text(problem):
    hint = problem["hint"]
    return hint


def get_context_text(problem, use_visual_clues):
    txt_context = get_hint_text(problem)
    img_context = get_visual_clues(problem) if use_visual_clues else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm["choices"]
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    # print(choice_txt)
    return choice_txt


def get_answer(problem, options):
    return options[problem["answer"]]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem["lecture"].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem["solution"].replace("\n", "\\n")
    return solution


def create_one_example(
    format, question, context, choice, answer, lecture, solution, test_example=True
):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == "A":
        output = f"Answer: The answer is {answer}."

    elif output_format == "AL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == "AE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == "ALE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == "AEL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == "LA":
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == "EA":
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == "LEA":
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == "ELA":
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text
