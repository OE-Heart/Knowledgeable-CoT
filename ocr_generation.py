import easyocr
import json
from tqdm import tqdm

if __name__ == "__main__":

    data_file = "data/scienceqa/problems.json"
    image_dir = "data/scienceqa/images"
    visual_clues_file = "data/visual_clues_with_chat.json"
    ocrs_file = "data/ocrs.json"
    output_file = "data/visual_clues_ocrs.json"

    data = json.load(open(data_file, 'r'))

    visual_clues = json.load(open(visual_clues_file, 'r'))

    reader = easyocr.Reader(['en']) # ocr reader

    img_pids = [pid for pid in data.keys() if data[pid]["image"]]

    results = {
        "model": "easyocr",
        "url": "https://github.com/JaidedAI/EasyOCR",
        "version": "1.1.8",
        "texts": {}
    }


    for pid in tqdm(img_pids):
        split = data[pid]["split"]
        image_file = f"{image_dir}/{split}/{pid}/image.png"

        result = reader.readtext(image_file)
        results["texts"][pid] = str(result)

        if len(result) > 0:
            texts = [(t[0], t[1]) for t in result] # (coordiantes, text)

            visual_clues["visual_clues"][pid] += f"\n\nDetected text in the image: {texts}"
            visual_clues["visual_clues"][pid] = visual_clues["visual_clues"][pid].strip()

    with open(ocrs_file, 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_file, 'w') as f:
        json.dump(visual_clues, f, indent=2)
