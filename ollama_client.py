import logging
import requests
from functools import lru_cache

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
CONNECT_TIMEOUT = 5    # ลดลงจาก 10
READ_TIMEOUT = 30      # ลดลงจาก 60
MAX_RETRIES = 2
CACHE_SIZE = 100      # จำนวนคำตอบที่จะเก็บใน cache

@lru_cache(maxsize=CACHE_SIZE)
def cached_generate(prompt: str) -> str:
    """Cache wrapper for generate_response"""
    return _generate_response(prompt)

def generate_response(question: str, context: str = None) -> str:
    try:
        if context:
            # ลดขนาด context โดยเลือกเฉพาะส่วนที่เกี่ยวข้องที่สุด
            context_parts = context.split('\n\n')
            relevant_parts = context_parts[:2]  # เลือก 2 ส่วนแรกที่เกี่ยวข้องที่สุด
            shortened_context = '\n\n'.join(relevant_parts)
            
            prompt = f"""คุณเป็น AI ที่ช่วยตอบคำถามโดยใช้ข้อมูลที่ให้มา
ตอบให้กระชับ ตรงประเด็น

ข้อมูลอ้างอิง:
{shortened_context}

คำถาม: {question}

คำตอบ:"""
        else:
            prompt = f"""คุณเป็น AI ผู้ช่วยตอบคำถาม
ตอบให้กระชับ ตรงประเด็น

คำถาม: {question}

คำตอบ:"""

        return cached_generate(prompt)

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "ขออภัย เกิดข้อผิดพลาดในการประมวลผล"

def _generate_response(prompt: str) -> str:
    """Internal function to generate response from Ollama"""
    for attempt in range(MAX_RETRIES):
        try:
            session = requests.Session()
            session.mount('http://', requests.adapters.HTTPAdapter(max_retries=1))
            
            response = session.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,     # ลดลงจาก 0.7
                    "num_predict": 256,      # ลดลงจาก 512
                    "num_ctx": 512,          # จำกัด context window
                    "num_thread": 4,         # ใช้ multi-threading
                    "top_k": 10,            # จำกัดตัวเลือกคำตอบ
                    "top_p": 0.9            # ช่วยให้ตอบตรงประเด็นขึ้น
                }
            }, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))

            if response.status_code == 200:
                return response.json()["response"].strip()
                
            logger.error(f"Ollama API error: Status={response.status_code}, Response={response.text[:200]}")
            if attempt < MAX_RETRIES - 1:
                continue
            return "ขออภัย ระบบยังไม่พร้อมให้บริการ กรุณาติดต่อผู้ดูแลระบบ"

        except (requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            logger.error(f"Connection error on attempt {attempt + 1}/{MAX_RETRIES}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                continue
            return "ขออภัย ระบบ AI ไม่สามารถตอบคำถามได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
            
    return "ขออภัย เกิดข้อผิดพลาดในการประมวลผล"
