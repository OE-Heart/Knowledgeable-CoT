import os
import json
import torch
import warnings
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from easyinstruct.utils.api import set_openai_key

from chat.chatgpt import AskQuestions, summarize_chat
from chat.blip2 import Blip2


warnings.filterwarnings("ignore")


def get_visual_clues(processor, model, image, device):
    prompt = "a photo of"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        device, torch.float16
    )

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text


def get_visual_clues_with_chat(processor, model, image, device):
    blip2 = Blip2(processor, model, device)
    chat = AskQuestions(image, blip2, model="gpt-3.5-turbo")

    questions, answers, n_token_chat = chat.chatting(n_rounds=10, print_mode="no")
    summary, summary_prompt, n_token_sum = summarize_chat(
        questions, answers, model="gpt-3.5-turbo"
    )

    result = {
        "summary": summary,
        "chat": summary_prompt,
        "n_token": n_token_chat + n_token_sum,
    }

    return result["summary"]


if __name__ == "__main__":
    ## Arguments
    input_path = "data/scienceqa"
    output_path = "data"
    output_name = "visual_clues_with_chat.json"
    plm_path = "/data/oyx/PLM"
    device_id = 2
    multiple_api_keys = False

    # model_name = "blip2-opt-2.7b"
    model_name = "blip2-flan-t5-xxl"
    model_path = os.path.join(plm_path, model_name)

    ## Read data
    problems = json.load(open(os.path.join(input_path, "problems.json")))
    pids = [
        pid for pid in list(problems.keys()) if problems[pid]["image"] == "image.png"
    ]

    print("number of images: ", len(pids))

    ## Prepare the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        {"torch_dtype": torch.float16}
        if model_name == "blip2-opt-2.7b"
        else {"load_in_8bit": True}
    )

    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path, device_map={'': device_id}, **dtype
    )

    if multiple_api_keys:
        api_list = json.load(open("openai_keys.json"))

    output_file = os.path.join(output_path, output_name)
    if os.path.exists(output_file):
        check_point = json.load(open(output_file))
        visual_clues = check_point["visual_clues"]
        count = len(visual_clues)

        if count >= len(pids):
            print("# The result file is complete! We will exit!!!")
            exit()
        else:
            print("# The result file is not complete! We will continue!!!")
            pids = pids[count:]
    else:
        visual_clues = {}

    ## Generate image visual_clues
    print(f"Generating visual_clues!")
    for i, pid in enumerate(tqdm(pids)):
        image_file = os.path.join(
            input_path, "images", problems[pid]["split"], pid, "image.png"
        )
        image = Image.open(image_file)

        if multiple_api_keys:
            set_openai_key(api_list[i % len(api_list)])

        try:
            clues = get_visual_clues_with_chat(processor, model, image, device)
            visual_clues[pid] = clues.capitalize() + "."
            # print(visual_clues[pid])
        except Exception as e:
            print(image_file)
            print(e)

        ## Save the visual_clues
        if (i + 1) % 5 == 0 or (i + 1) == len(pids):
            print(f"Saved to {output_file}")

            results = {"model": model_name, "visual_clues": visual_clues}

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, separators=(",", ": "))
