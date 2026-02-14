
import os
import torch
import cv2
import numpy as np
import requests
import random
from PIL import Image
from ultralytics import YOLO
from moviepy.editor import ImageSequenceClip, AudioFileClip
from diffusers import StableDiffusionPipeline
from rembg import remove
from io import BytesIO
import gradio as gr
from dotenv import load_dotenv
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OUTPUT_FOLDER = "output_ad"
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
AUDIO_FILE = os.path.join(OUTPUT_FOLDER, "voiceover.mp3")
FINAL_VIDEO = os.path.join(OUTPUT_FOLDER, "final_ad.mp4")

os.makedirs(IMAGE_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")

# Load Stable Diffusion once
sd_model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    sd_model_id,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
pipe.to(DEVICE)
def extract_object(image_path):
    with open(image_path, "rb") as f:
        output_bytes = remove(f.read())
        obj_image = Image.open(BytesIO(output_bytes)).convert("RGBA")

    obj_path = os.path.join(IMAGE_FOLDER, f"object_{os.path.basename(image_path)}")
    obj_image.save(obj_path)

    image_cv = cv2.imread(image_path)
    results = model(image_cv)

    label = "product"
    for r in results:
        if len(r.boxes.cls) > 0:
            cls_id = int(r.boxes.cls[0])
            label = model.names[cls_id]

    return obj_path, label


def generate_ai_backgrounds(prompts):
    generated_backgrounds = []

    for prompt in prompts:
        image = pipe(prompt).images[0]
        bg_save_path = os.path.join(
            IMAGE_FOLDER,
            f"generated_bg_{random.randint(0, 9999)}.png"
        )
        image.save(bg_save_path)
        generated_backgrounds.append(bg_save_path)

    return generated_backgrounds


def generate_voiceover(script_text):
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY not set in environment variables.")

    VOICE_ID = "pNInz6obpgDQGcFmaJgB"

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": script_text,
        "model_id": "eleven_monolingual_v1"
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        with open(AUDIO_FILE, "wb") as f:
            f.write(response.content)
        return AUDIO_FILE
    else:
        raise RuntimeError(f"Voiceover failed: {response.text}")

def generate_ad(image):
    temp_path = os.path.join(IMAGE_FOLDER, "input.png")
    image.save(temp_path)

    object_path, label = extract_object(temp_path)

    prompts = [
        f"Luxury studio background for {label}",
        f"Minimal aesthetic background for {label}"
    ]

    bg_paths = generate_ai_backgrounds(prompts)

    script = f"Introducing our premium {label}. Experience innovation like never before."

    audio_path = generate_voiceover(script)

    clip = ImageSequenceClip(bg_paths, fps=1)
    audio = AudioFileClip(audio_path)
    clip = clip.set_audio(audio)
    clip.write_videofile(FINAL_VIDEO, codec="libx264")

    return FINAL_VIDEO

interface = gr.Interface(
    fn=generate_ad,
    inputs=gr.Image(type="pil"),
    outputs=gr.Video(),
    title=" AdVisio - AI Advertisement Generator",
    description="Upload a product image to generate an AI-powered advertisement."
)

if __name__ == "__main__":
    interface.launch()
