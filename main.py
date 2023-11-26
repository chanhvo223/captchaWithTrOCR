from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import time
import torch

model_path = "asia-captcha-model1"

if 'processor' not in locals():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(model_path)

    #Pick GPU or CPU
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Running in device:", device)
# model.to(device)

    # load image from the IAM dataset
image = Image.open("dataset_captcha_asiacode/train/cDk1pS.JPG").convert("RGB")
timeini = time.time()
pixel_values = processor(image, return_tensors="pt").pixel_values

    # inference
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


timeend = time.time() - timeini
print("Result: ", generated_text)
print("Execution time: ", timeend)
