# 필요한 패키지 설치
!pip install nest_asyncio pyngrok uvicorn fastapi pydantic torch pillow google-generativeai diffusers deep-translator boto3 botocore

import io
import nest_asyncio
from pyngrok import ngrok, conf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import torch
from PIL import Image
import google.generativeai as genai
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from typing import Optional
import logging
import asyncio
from deep_translator import GoogleTranslator
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ngrok 및 FastAPI 설정
conf.get_default().monitor_thread = False
app = FastAPI()

# API 및 클라우드 키 설정
GOOGLE_API_KEY = ""
NGROK_AUTH_TOKEN = ""
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = ""
S3_BUCKET_NAME = ""
S3_BUCKET_PATH = ""

# ngrok 설정 및 URL 출력
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(8000)
logger.info(f'NGrok URL: {public_url}')
print(public_url)

# Stable Diffusion 설정
pipe = None

# S3 클라이언트 초기화
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def initialize_stable_diffusion():
    global pipe
    if pipe is None:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16
            )
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                pipe.enable_attention_slicing()
                if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except ImportError:
                        pass
            else:
                pipe = pipe.to("cpu")
        except Exception as e:
            logger.error(str(e))
            raise HTTPException(status_code=500, detail="Init failed")
    return pipe

class TextRequest(BaseModel):
    text: str

class ImageResponse(BaseModel):
    image_url: Optional[str]
    created_at: str

def upload_to_s3(image: Image.Image, filename: str) -> str:
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        s3_key = f"{S3_BUCKET_PATH}/{filename}"
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=img_byte_arr,
            ContentType='image/png'
        )

        url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        return url

    except ClientError as e:
        logger.error(f"S3 upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload image to S3")

def transform_prompt(original_text: str) -> str:
    translator = GoogleTranslator(source='ko', target='en')
    translated = translator.translate(original_text)
    base_prompt = (
        "Highly detailed and photorealistic Korean greenhouse farmland, showcasing vibrant and healthy tomato plants in full growth. "
        "A diligent farmer is inspecting soil moisture with precision tools, conveying the dedication to daily agricultural tasks. "
        "Warm, diffused sunlight streams through the greenhouse’s plastic covering, illuminating rows of meticulously maintained lush green plants. "
        "Modern farm equipment, irrigation systems, and organized plant beds are clearly visible, emphasizing a harmonious blend of tradition and technology in farming. "
        "The environment is realistic and immersive, with a subtle morning haze adding depth and richness to the scene. "
        "Vibrant and natural earthy tones dominate, creating a serene and productive atmosphere. "
        "Inspired by professional documentary photography, with incredible texture, lifelike lighting, and a shallow depth of field for emphasis on detail."
    )
    return base_prompt

@app.post("/generate_image/", response_model=ImageResponse)
async def generate_image(req: TextRequest):
    try:
        pipe = initialize_stable_diffusion()
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Empty text")

        prompt = transform_prompt(req.text.strip())

        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=25,
                guidance_scale=8.0,
                width=512,
                height=512
            ).images[0]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ai_image_{timestamp}.png"
        image_url = upload_to_s3(image, filename)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ImageResponse(
            image_url=image_url,
            created_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def run_server():
    try:
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(str(e))
    finally:
        try:
            ngrok.disconnect(public_url.public_url)
        except:
            pass

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(run_server())
