import json
import logging
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-base'):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dimension = None
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Initialized RAG system with model: {model_name}")

        # ตรวจสอบว่ามี GPU หรือไม่
        self.use_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                self.use_gpu = True
                logger.info("GPU acceleration enabled")
        except:
            pass

    def load_documents(self, json_path: str):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_dir = os.path.join(base_dir, 'data', 'json')
            cache_file = os.path.join(self.cache_dir, 'faiss_index.cache')
            
            # ตรวจสอบ cache
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        self.documents = cache_data['documents']
                        self.index = cache_data['index']
                        self.dimension = cache_data['dimension']
                        logger.info("Loaded index from cache")
                        return True
                except Exception as e:
                    logger.warning(f"Could not load cache: {e}")

            # ถ้าไม่มี cache หรือโหลดไม่สำเร็จ ให้โหลดจากไฟล์
            if not os.path.exists(json_dir):
                logger.error(f"Directory not found: {json_dir}")
                return False

            processed_docs = []
            json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

            if not json_files:
                logger.warning(f"No JSON files found in {json_dir}")
                return False

            # โหลดเอกสารแบบ batch
            for json_file in tqdm(json_files, desc="Loading documents"):
                try:
                    file_path = os.path.join(json_dir, json_file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for item in data:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            text = f"{item['question']} {item['answer']}"
                            processed_docs.append({
                                'text': text,
                                'question': item['question'],
                                'answer': item['answer'],
                                'source': json_file
                            })
                except Exception as e:
                    logger.error(f"Error loading file {json_file}: {str(e)}")
                    continue
            
            self.documents = processed_docs
            self._build_index()

            # บันทึก cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'index': self.index,
                        'dimension': self.dimension
                    }, f)
                logger.info("Saved index to cache")
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")

            return True
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False

    def _build_index(self):
        if not self.documents:
            logger.warning("No documents to index")
            return

        # Encode documents in batches
        batch_size = 32
        texts = [doc['text'] for doc in self.documents]
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding documents"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encoder.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Initialize IVF index for faster search
        self.dimension = embeddings.shape[1]
        n_lists = min(int(np.sqrt(len(self.documents))), 100)  # จำนวน clusters
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_lists, faiss.METRIC_INNER_PRODUCT)
        
        # ใช้ GPU ถ้ามี
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU for FAISS index")
            except Exception as e:
                logger.warning(f"Could not use GPU: {e}")

        # Train and add vectors
        self.index.train(embeddings)
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Built FAISS IVF index with {len(self.documents)} documents")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        try:
            if self.index is None:
                return []

            # Set number of probes for better recall
            if isinstance(self.index, faiss.IndexIVFFlat):
                self.index.nprobe = min(16, self.index.nlist)

            query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores, indices = self.index.search(query_embedding.astype('float32'), k)

            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'question': doc['question'],
                        'answer': doc['answer'],
                        'score': float(score),
                        'text': doc['text']
                    })

            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"Query: {query}")
            logger.info(f"Top result: {results[0]['question']} (score: {results[0]['score']:.4f})")
            
            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
