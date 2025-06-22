import os
import json
import logging
from rag import RAGSystem
from ollama_client import generate_response

logger = logging.getLogger(__name__)
rag_system = None

def initialize_rag():
    global rag_system
    try:
        rag_system = RAGSystem()
        success = rag_system.load_documents(None)  # No need to pass json_dir anymore
        if success:
            logger.info("RAG system initialized successfully")
            return True
        logger.error("Failed to initialize RAG system")
        return False
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return False

def search_from_documents(question):
    try:
        global rag_system
        if rag_system is None:
            if not initialize_rag():
                return "ขออภัย ระบบยังไม่พร้อมใช้งาน", False, None

        qa_candidates = []
        content_candidates = []

        # ค้นหาข้อมูลจากเอกสาร
        base_results = rag_system.search(question, k=5)

        # แยก Q&A และ content พร้อมเก็บ score
        if base_results:
            for r in base_results:
                if isinstance(r, dict):
                    if 'question' in r and 'answer' in r:
                        qa_candidates.append({
                            'type': 'qa',
                            'question': r['question'],
                            'answer': r['answer'],
                            'score': r['score']
                        })
                    elif 'content' in r:
                        content_candidates.append({
                            'type': 'content',
                            'text': r.get('content', ''),
                            'score': r['score']
                        })

        # ถ้าไม่มีอะไรเลย
        if not qa_candidates and not content_candidates:
            return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้อง", False, None

        # หา Q&A และ content ที่ score สูงสุด
        best_qa = max(qa_candidates, key=lambda x: x['score']) if qa_candidates else None
        best_content = max(content_candidates, key=lambda x: x['score']) if content_candidates else None

        # เปรียบเทียบ score และเลือกแบบที่สูงสุด
        if best_qa and (not best_content or best_qa['score'] >= best_content['score']):
            logger.info(f"Found question-answer in document: {best_qa['question']}")
            return best_qa['answer'], True, {
                'question': question,
                'contexts': [best_qa['answer']],
                'score': best_qa['score']
            }
        elif best_content:
            logger.info(f"Query: {question}")
            logger.info(f"Top content result: {best_content['text'][:30]} (score: {best_content['score']:.4f})")
            # ถ้า match ดี (score >= 0.8)
            if best_content['score'] >= 0.8:
                return generate_response(question, best_content['text']), True, {
                    'question': question,
                    'contexts': [best_content['text']],
                    'score': best_content['score']
                }
            if best_content['score'] >= 0.3:
                # รวม context ที่ score >= 0.2
                sorted_contents = sorted(content_candidates, key=lambda x: x['score'], reverse=True)
                contexts = [r['text'] for r in sorted_contents[:3] if r['score'] >= 0.2]
                combined_context = "\n\n".join(contexts)
                try:
                    return generate_response(question, combined_context), True, {
                        'question': question,
                        'contexts': contexts,
                        'score': best_content['score']
                    }
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    return best_content['text'], True, {
                        'question': question,
                        'contexts': contexts,
                        'score': best_content['score']
                    }
            return "ขออภัย ไม่พบข้อมูลที่ตรงกับคำถามของคุณ", False, None

        return "ขออภัย ไม่พบข้อมูลที่ตรงกับคำถามของคุณ", False, None

    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}")
        return "เกิดข้อผิดพลาดในการค้นหา", False, None
