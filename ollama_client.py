import logging
import requests
import time
from functools import lru_cache
import json 

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
CONNECT_TIMEOUT = 3 
READ_TIMEOUT = 15 
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
CACHE_SIZE = 100

@lru_cache(maxsize=CACHE_SIZE)
def cached_generate(prompt: str) -> str:
    """Cache wrapper for generate_response"""
    return _generate_response(prompt)

def generate_response(question: str, context: str = None) -> str:
    try:
        if context:
            context_parts = context.split('\n\n')
            relevant_parts = []

            if not relevant_parts:
                relevant_parts = context_parts[:2]
                
            shortened_context = '\n\n'.join(relevant_parts)

            # เพิ่มบุคลิก "ผู้หญิงร่าเริง"
            prompt = f"""คุณชื่อ DMC Chatbot เป็น AI ผู้ช่วยผู้หญิง นิสัยร่าเริง พูดจาน่ารัก เป็นกันเอง 
คุณชอบช่วยเหลือผู้อื่นและให้คำแนะนำด้วยภาษาที่เข้าใจง่าย มีความสุภาพแต่ไม่ทางการเกินไป

ข้อมูลอ้างอิง:
{shortened_context}

คำถาม: {question}

คำแนะนำในการตอบ:
- ตอบคำถามด้วยภาษาที่เข้าใจง่ายและเป็นมิตร
- หากเป็นเรื่องลิงค์หรือ URL ให้แสดง URL เต็ม
- หากมีขั้นตอนให้แสดงเป็นข้อๆ สรุปให้ชัดเจน
- ใช้คำพูดที่เป็นมิตร เช่น ค่ะ, น้าา, นะคะ, เย้!
- ตอบให้เข้าใจง่าย กระชับ ตรงประเด็น
"""
        return cached_generate(prompt)

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้งในภายหลัง"

def _generate_response(prompt: str) -> str:
    """Internal function to generate response from Ollama"""
    for attempt in range(MAX_RETRIES):
        try:
            # Add exponential backoff
            if attempt > 0:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                
            session = requests.Session()
            session.mount('http://', requests.adapters.HTTPAdapter(max_retries=1))
            
            response = session.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,  
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,  
                    "num_ctx": 1024,    
                    "num_thread": 4,
                    "top_k": 10,
                    "top_p": 0.9
                }
            }, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True)

            if response.status_code == 200:
                # อ่านทีละบรรทัดและรวม response
                result = ""
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            result += data["response"]
                        if data.get("done"):
                            break
                    except Exception as e:
                        logger.error(f"Error parsing streaming line: {e}")
                        continue
                return result.strip()
                
            logger.error(f"Ollama API error: Status={response.status_code}, Response={response.text[:200]}")
            if attempt < MAX_RETRIES - 1:
                continue
            return "ขออภัยค่ะ ระบบยังไม่พร้อมให้บริการ กรุณาติดต่อผู้ดูแลระบบ"

        except (requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            logger.error(f"Connection error on attempt {attempt + 1}/{MAX_RETRIES}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                continue
            return "ขออภัยค่ะ ระบบ AI ไม่สามารถตอบคำถามได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
            
    return "ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผล"
