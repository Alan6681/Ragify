from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
from PIL import Image
import os
import tempfile

load_dotenv()

class AIService:
    def __init__(self, files):
        self.files = files
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.easyocr_reader = None 

    def _init_easyocr(self):
        """Initialize EasyOCR reader once and reuse"""
        if self.easyocr_reader is None:
            import easyocr
            try:
                import torch
                gpu = torch.cuda.is_available()
                if gpu:
                    print("🚀 GPU detected - using GPU acceleration")
                else:
                    print("💻 Using CPU for OCR")
            except:
                gpu = False
                print("💻 Using CPU for OCR")
            
            print("⏳ Initializing EasyOCR (this may take a moment)...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
            print("✅ EasyOCR initialized")
        
        return self.easyocr_reader

    def _is_scanned_pdf(self, pdf_path):
        """Check if PDF is scanned"""
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            total_text = "".join([doc.page_content for doc in docs])
            return len(total_text.strip()) < 100
        except Exception as e:
            print(f"⚠️ Error checking PDF type: {e}")
            return True

    def _ocr_with_easyocr(self, pdf_path, file_name):
        """Fast OCR with single shared reader and batch processing"""
        try:
            import pypdfium2 as pdfium
            import numpy as np
            import time
            
            print(f"🔄 Starting OCR for {file_name}...")
            start_time = time.time()
            
        
            reader = self._init_easyocr()
            
            # Convert PDF to images
            print(f"   Converting PDF to images...")
            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)
            
            # Prepare all images
            images = []
            for page_index in range(total_pages):
                page = pdf[page_index]
                pil_image = page.render(scale=1.5).to_pil()  
                
                # Resize if needed
                max_dimension = 1800
                if max(pil_image.size) > max_dimension:
                    ratio = max_dimension / max(pil_image.size)
                    new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
                images.append((page_index, pil_image))
                page.close()
            
            pdf.close()
            
            print(f"📄 Processing {total_pages} pages with optimized OCR...")
            
            documents = []

            for i, image in images:
                try:
                    img_array = np.array(image)
                    
                    # Optimized OCR settings
                    results = reader.readtext(
                        img_array,
                        paragraph=True,
                        detail=0,
                        batch_size=10,
                        text_threshold=0.7,  
                        low_text=0.4,  
                        link_threshold=0.4,  
                        canvas_size=1800,  
                        mag_ratio=1.0  
                    )
                    
                    if isinstance(results, list):
                        text = '\n'.join(results)
                    else:
                        text = str(results)
                    
                    if text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': file_name,
                                'page': i,
                                'ocr_method': 'easyocr_optimized'
                            }
                        )
                        documents.append(doc)
                    
                    # Progress tracking
                    if (i + 1) % 5 == 0 or (i + 1) == total_pages:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / (i + 1)
                        eta = avg_time * (total_pages - (i + 1))
                        print(f"   Progress: {i+1}/{total_pages} pages | "
                              f"{elapsed:.1f}s elapsed | "
                              f"ETA: {eta:.1f}s | "
                              f"Avg: {avg_time:.1f}s/page")
                
                except Exception as e:
                    print(f"   ⚠️ Error on page {i+1}: {e}")
            
            elapsed_total = time.time() - start_time
            avg_per_page = elapsed_total / total_pages if total_pages > 0 else 0
            print(f"✅ OCR completed: {total_pages} pages in {elapsed_total:.1f}s ({avg_per_page:.1f}s/page)")
            
            return documents
            
        except Exception as e:
            print(f"❌ EasyOCR failed for {file_name}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _ocr_with_tesseract_fallback(self, pdf_path, file_name):
        """Tesseract fallback (if available)"""
        try:
            import pytesseract
            import pypdfium2 as pdfium
            import time
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            print(f"🔄 Using Tesseract OCR for {file_name}...")
            start_time = time.time()
            
            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)
            
            images = []
            for page_index in range(total_pages):
                page = pdf[page_index]
                pil_image = page.render(scale=1.5).to_pil()
                images.append((page_index, pil_image))
                page.close()
            
            pdf.close()
            
            print(f"📄 Processing {total_pages} pages with Tesseract...")
            
            def ocr_page(page_data):
                i, image = page_data
                try:
                    # Fast Tesseract config
                    text = pytesseract.image_to_string(image, config='--psm 3 --oem 1')
                    return (i, text) if text.strip() else None
                except Exception as e:
                    print(f"   ⚠️ Error on page {i+1}: {e}")
                    return None
            
            documents = []
            processed = 0
            
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(ocr_page, img_data): img_data[0] 
                          for img_data in images}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        i, text = result
                        doc = Document(
                            page_content=text,
                            metadata={'source': file_name, 'page': i, 'ocr_method': 'tesseract'}
                        )
                        documents.append(doc)
                    
                    processed += 1
                    if processed % 5 == 0 or processed == total_pages:
                        elapsed = time.time() - start_time
                        print(f"   Progress: {processed}/{total_pages} pages ({elapsed:.1f}s)")
            
            documents.sort(key=lambda x: x.metadata['page'])
            
            elapsed_total = time.time() - start_time
            print(f"✅ Tesseract completed: {total_pages} pages in {elapsed_total:.1f}s")
            
            return documents
            
        except ImportError:
            print("⚠️ Tesseract not available")
            return []
        except Exception as e:
            print(f"❌ Tesseract failed: {e}")
            return []

    def load_pdf(self):
        """Load PDFs with OCR for scanned documents"""
        documents = []
        
        for file in self.files:
            file.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            try:
                is_scanned = self._is_scanned_pdf(tmp_path)
                
                if not is_scanned:
                    print(f"📄 Loading regular PDF: {file.name}")
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"✅ Successfully loaded {file.name} ({len(docs)} pages)")
                else:
                    print(f"📸 Detected scanned PDF: {file.name}")
                    
                    # Try Tesseract first (faster)
                    docs = self._ocr_with_tesseract_fallback(tmp_path, file.name)
                    
                    # Fallback to EasyOCR
                    if not docs or len(docs) == 0:
                        print(f"🔄 Using EasyOCR for {file.name}...")
                        docs = self._ocr_with_easyocr(tmp_path, file.name)
                    
                    if docs and len(docs) > 0:
                        documents.extend(docs)
                        print(f"✅ OCR successful for {file.name} ({len(docs)} pages)")
                    else:
                        print(f"❌ OCR failed for {file.name}")
                        error_doc = Document(
                            page_content=f"Unable to extract text from {file.name}.",
                            metadata={"source": file.name, "error": True}
                        )
                        documents.append(error_doc)
                        
            except Exception as e:
                print(f"❌ Error processing {file.name}: {e}")
                import traceback
                traceback.print_exc()
                error_doc = Document(
                    page_content=f"Error: {str(e)}",
                    metadata={"source": file.name, "error": True}
                )
                documents.append(error_doc)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

        if not documents:
            raise ValueError("No documents loaded.")

        successful_docs = [doc for doc in documents if not doc.metadata.get("error", False)]
        if successful_docs:
            documents = successful_docs
            print(f"\n📚 Total: {len(documents)} pages from {len(self.files)} file(s)")

        print(f"✂️ Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)
        print(f"✅ Created {len(splits)} chunks")

        return splits

    def create_vectorstore(self):
        if self.vectorstore is None:
            print(f"🔍 Creating vector store...")
            docs = self.load_pdf()
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            print(f"✅ Vector store created")
        return self.vectorstore

    def create_retriever(self):
        return self.create_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

    def load_llm(self):
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0)