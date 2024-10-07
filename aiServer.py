# 1. 필요한 패키지 설치
!pip install pyngrok fastapi uvicorn torch diffusers transformers google-generativeai pillow nest-asyncio xformers

import nest_asyncio
from pyngrok import ngrok
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Smart Farm Diary API",
    description="농업 일지 생성 및 이미지 생성을 위한 API",
    version="1.0.0"
)

# 환경 변수 설정
GOOGLE_API_KEY = "AIzaSyCXhDJgQBldkqjGUKLvZHkJELq25ntgImw"  
NGROK_AUTH_TOKEN = "2j27vD2VtJOyWNLlG1Hhe6aUTVl_782M5FWdcUq833RhR4ZhE"
SAVE_PATH = "./farm_diary_outputs"
os.makedirs(SAVE_PATH, exist_ok=True)

# NGrok 설정
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Gemini API 설정
genai.configure(api_key=GOOGLE_API_KEY)

# Stable Diffusion 모델
pipe = None

class DiaryEntry(BaseModel):
    crop: str
    weather: str
    temperature: float
    humidity: float
    issues: str
    work: str

class DiaryResponse(BaseModel):
    diary_content: str
    image_path: Optional[str]
    created_at: str

def initialize_stable_diffusion():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            pipe.enable_xformers_memory_efficient_attention()
    return pipe

def generate_enhanced_prompt(entry: DiaryEntry) -> str:
    return f"""
    A professional agricultural photograph of a {entry.crop} farm.
    {entry.weather} weather conditions with {entry.temperature}°C temperature.
    
    Photorealistic style with the following specific details:
    - Crystal clear 8K resolution
    - Professional agricultural photography
    - Golden hour lighting
    - Perfect exposure and composition
    - Vibrant natural colors
    - Shallow depth of field where appropriate
    
    Must show:
    - Detailed view of {entry.crop} plants
    - Evidence of {entry.work} activities
    - {entry.weather} conditions in the sky
    - Subtle signs of {entry.issues}
    - Modern farming equipment
    
    Additional specifications:
    - Hyperrealistic detail
    - Professional agricultural setting
    - Environmental context
    """

def generate_diary_content(entry: DiaryEntry) -> str:
    prompt = f"""
    전문 농업 컨설턴트의 관점에서 다음 정보를 바탕으로 상세한 농가 일지를 작성해주세요:
    
    작성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시')}
    
    기본 정보:
    - 작물: {entry.crop}
    - 기상 상황: {entry.weather}
    - 온도: {entry.temperature}°C
    - 습도: {entry.humidity}%
    
    주요 사항:
    - 특이사항: {entry.issues}
    - 작업내용: {entry.work}
    
    다음 항목들을 포함해 전문적으로 작성해주세요:
    1. 작물 생육 상태 분석
    2. 환경 요인 영향 평가
    3. 수행된 작업의 적절성 평가
    4. 향후 1주일간의 관리 권장사항
    5. 예상되는 문제점과 대응 방안
    6. 수확량 예측 및 품질 전망
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"일지 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="일지 생성 실패")

async def simulate_long_task(duration: int):
    """장시간 작업 시뮬레이션"""
    await asyncio.sleep(duration)
    return "작업 완료"

@app.post("/generate_diary/", response_model=DiaryResponse)
async def create_diary(entry: DiaryEntry):
    try:
        # 작업이 진행 중임을 시뮬레이션
        await simulate_long_task(5)

        # 일지 내용 생성
        diary_content = generate_diary_content(entry)
        
        # 이미지 생성
        pipe = initialize_stable_diffusion()
        prompt = generate_enhanced_prompt(entry)
        
        try:
            with torch.cuda.amp.autocast():
                image = pipe(
                    prompt,
                    num_inference_steps=75,    # 더 높은 세밀도
                    guidance_scale=10.0,       # 더 높은 가이던스 스케일
                    width=768,                 # 해상도 증가
                    height=768
                ).images[0]
        except Exception as img_error:
            logger.error(f"이미지 생성 중 오류 발생: {str(img_error)}")
            raise HTTPException(status_code=500, detail="이미지 생성 중 오류가 발생했습니다.")
        
        # 결과물 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"farm_image_{timestamp}.png"
        diary_filename = f"farm_diary_{timestamp}.txt"
        
        image_path = os.path.join(SAVE_PATH, image_filename)
        diary_path = os.path.join(SAVE_PATH, diary_filename)
        
        # 이미지 저장
        try:
            image.save(image_path)
        except Exception as save_error:
            logger.error(f"이미지 저장 중 오류 발생: {str(save_error)}")
            raise HTTPException(status_code=500, detail="이미지 저장 중 오류가 발생했습니다.")
        
        # 일지 저장
        try:
            with open(diary_path, "w", encoding="utf-8") as f:
                f.write(diary_content)
        except Exception as txt_error:
            logger.error(f"일지 저장 중 오류 발생: {str(txt_error)}")
            raise HTTPException(status_code=500, detail="일지 저장 중 오류가 발생했습니다.")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        return DiaryResponse(
            diary_content=diary_content,
            image_path=image_path,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="처리 중 문제가 발생했습니다.")

def run_server():
    # NGrok 터널 생성
    public_url = ngrok.connect(8000)
    print(f'NGrok 터널이 생성되었습니다. Public URL: {public_url.public_url}')
    
    # FastAPI 서버 설정
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    # Jupyter/Colab 환경에서 비동기 실행을 위한 설정
    nest_asyncio.apply()
    
    try:
        # 서버 실행
        server.run()
    except KeyboardInterrupt:
        print("서버가 중단되었습니다.")
    finally:
        # NGrok 터널 제거
        ngrok.disconnect(public_url.public_url)

if __name__ == "__main__":
    run_server()
