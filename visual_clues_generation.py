import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from chat.chatgpt import caption_image
from chat.blip2 import Blip2

import warnings

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

    result = caption_image(blip2, image, model="chatgpt", print_mode="chat")
    return result


if __name__ == "__main__":
    ## Arguments
    input_path = "data/scienceqa"
    output_path = "data"
    output_name = "visual_clues.json"
    plm_path = "/data/oyx/PLM"

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
        model_path, **dtype, device_map="auto"
    )
    model.to(device)

    ## Generate image visual_clues
    visual_clues = {}

    print(f"Generating visual_clues!")
    for pid in tqdm(pids):
        image_file = os.path.join(
            input_path, "images", problems[pid]["split"], pid, "image.png"
        )
        image = Image.open(image_file)
        try:
            clues = get_visual_clues(processor, model, image, device)
            visual_clues[pid] = clues.capitalize() + "."
            # print(visual_clues[pid])
        except Exception as e:
            print(image_file)
            print(e)

    ## Save the visual_clues
    output_file = os.path.join(output_path, output_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"Saved to {output_file}")

    results = {"model": model_name, "visual_clues": visual_clues}

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, separators=(",", ": "))
