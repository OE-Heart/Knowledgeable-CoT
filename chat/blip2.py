import torch


class Blip2:
    def __init__(self, blip2_processor, blip2, device):
        self.blip2_processor = blip2_processor
        self.blip2 = blip2
        self.device = device

    def ask(self, raw_image, question):
        inputs = self.blip2_processor(raw_image, question, return_tensors="pt").to(
            self.device, torch.float16
        )
        out = self.blip2.generate(**inputs)
        answer = self.blip2_processor.batch_decode(out, skip_special_tokens=True)[0].strip()

        return answer

    def caption(self, raw_image):
        # starndard way to caption an image in the blip2 paper
        caption = self.ask(raw_image, "a photo of")
        caption = caption.replace("\n", " ").strip()  # trim caption
        return caption

    def call_llm(self, prompts):
        prompts_temp = self.blip2_processor(None, prompts, return_tensors="pt")
        input_ids = prompts_temp["input_ids"].to(self.device)
        attention_mask = prompts_temp["attention_mask"].to(self.device, torch.float16)

        prompts_embeds = self.blip2.language_model.get_input_embeddings()(input_ids)

        outputs = self.blip2.language_model.generate(
            inputs_embeds=prompts_embeds, attention_mask=attention_mask
        )

        outputs = self.blip2_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        return outputs
