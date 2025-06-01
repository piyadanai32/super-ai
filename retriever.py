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

        results = []
        found_matches = []
        
        # ค้นหาข้อมูลจากเอกสาร
        base_results = rag_system.search(question, k=5) 
        
        if base_results:
            for r in base_results:
                if isinstance(r, dict):
                    if 'question' in r and 'answer' in r:
                        found_matches.append(f"Found question-answer in document: {r['question']}")
                        results.append({
                            'type': 'qa',
                            'text': r['answer'], 
                            'score': r['score']
                        })
                    elif 'content' in r:
                        results.append({
                            'type': 'content',
                            'text': r.get('content', ''),
                            'score': r['score']
                        })

        if not results:
            return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้อง", False, None

        # Log found matches
        for match in found_matches:
            logger.info(match)

        # Sort by score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        best_match = results[0]

        logger.info(f"Query: {question}")
        logger.info(f"Top result: {found_matches[0]} (score: {best_match['score']:.4f})")
        logger.info(f"คำถาม: {question}")

        if best_match['score'] >= 0.3:
            contexts = []
            for r in results[:3]:
                if r['score'] >= 0.2:
                    contexts.append(r['text'])

            combined_context = "\n\n".join(contexts)
            try:
                return generate_response(question, combined_context), True, {
                    'question': question,
                    'contexts': contexts,
                    'score': best_match['score']
                }
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return best_match['text'], True, {
                    'question': question,
                    'contexts': contexts,
                    'score': best_match['score']
                }

        return "ขออภัย ไม่พบข้อมูลที่ตรงกับคำถามของคุณ", False, None

    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}")
        return "เกิดข้อผิดพลาดในการค้นหา", False, None
