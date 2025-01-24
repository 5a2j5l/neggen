import argparse
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

def generate_caption(image_path):
    model_id = "meta-llama/Llama-3.2-11B-Vision"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    image = Image.open(image_path)

    prompt = '''<|image|><|begin_of_text|>You are a descriptive writer who excels at capturing the
 essence and details of items in clear language. Generate natural, detailed
 description of the item shown in the given image.'''
    inputs = processor(image, prompt, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(output[0])

    return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess")
    parser.add_argument("--data", type=str, required=True, help="dataset name")
    args = parser.parse_args()

    data_path_dict = {
        "baby": "../dataset/baby/images",
        "beauty": "../dataset/beauty/images",
        "clothing": "../dataset/clothing/images",
        "sports": "../dataset/sports/images"
    }

    if args.data in data_path_dict:
        real_path = data_path_dict[args.data]
        caption = generate_caption(real_path)
        with open('visual.txt', 'rw') as f:
            f.write(caption+'\n')
    else:
        print(f"Not supported dataset: {args.data}")
