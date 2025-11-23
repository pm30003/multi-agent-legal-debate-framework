import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Vector store and embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredFileLoader
)

class LegalRAGSystem:
    """
    Advanced RAG system for legal case law and statutes.
    Supports multiple document types and specialized legal embeddings.
    """
    
    def __init__(
        self,
        case_law_path: str = "./legal_documents/case_law",
        statutes_path: str = "./legal_documents/statutes",
        regulations_path: str = "./legal_documents/regulations",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the Legal RAG System
        
        Args:
            case_law_path: Path to case law documents
            statutes_path: Path to statutory documents  
            regulations_path: Path to regulatory documents
            embedding_model: Embedding model for vector similarity
        """
        self.case_law_path = Path(case_law_path)
        self.statutes_path = Path(statutes_path)
        self.regulations_path = Path(regulations_path)
        
        # Initialize embeddings - consider law-specific models
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter optimized for legal documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Vector stores for different document types
        self.case_law_store: Optional[FAISS] = None
        self.statutes_store: Optional[FAISS] = None
        self.regulations_store: Optional[FAISS] = None
        
        # Initialize stores
        self._initialize_vector_stores()
    
    def _initialize_vector_stores(self):
        """Initialize vector stores for different legal document types"""
        try:
            # Load case law documents
            if self.case_law_path.exists():
                case_law_docs = self._load_documents(self.case_law_path)
                if case_law_docs:
                    case_law_chunks = self.text_splitter.split_documents(case_law_docs)
                    self.case_law_store = FAISS.from_documents(case_law_chunks, self.embeddings)
            
            # Load statutes
            if self.statutes_path.exists():
                statutes_docs = self._load_documents(self.statutes_path)
                if statutes_docs:
                    statutes_chunks = self.text_splitter.split_documents(statutes_docs)
                    self.statutes_store = FAISS.from_documents(statutes_chunks, self.embeddings)
            
            # Load regulations
            if self.regulations_path.exists():
                regulations_docs = self._load_documents(self.regulations_path)
                if regulations_docs:
                    regulations_chunks = self.text_splitter.split_documents(regulations_docs)
                    self.regulations_store = FAISS.from_documents(regulations_chunks, self.embeddings)
                    
        except Exception as e:
            print(f"Error initializing vector stores: {e}")
    
    def _load_documents(self, path: Path) -> List[Document]:
        """Load documents from a directory"""
        documents = []
        
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == '.txt':
                        loader = TextLoader(str(file_path))
                    else:
                        loader = UnstructuredFileLoader(str(file_path))
                    
                    docs = loader.load()
                    # Add metadata
                    for doc in docs:
                        doc.metadata.update({
                            'source': str(file_path),
                            'document_type': self._get_document_type(path),
                            'file_name': file_path.name
                        })
                    documents.extend(docs)
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
        return documents
    
    def _get_document_type(self, path: Path) -> str:
        """Determine document type based on path"""
        if "case_law" in str(path):
            return "case_law"
        elif "statutes" in str(path):
            return "statute"
        elif "regulations" in str(path):
            return "regulation"
        return "unknown"
    
    def retrieve_case_law(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant case law documents"""
        if not self.case_law_store:
            return []
        return self.case_law_store.similarity_search(query, k=k)
    
    def retrieve_statutes(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant statutory provisions"""
        if not self.statutes_store:
            return []
        return self.statutes_store.similarity_search(query, k=k)
    
    def retrieve_regulations(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant regulatory provisions"""
        if not self.regulations_store:
            return []
        return self.regulations_store.similarity_search(query, k=k)
    
    def retrieve_comprehensive(
        self, 
        query: str, 
        case_law_k: int = 3,
        statutes_k: int = 2,
        regulations_k: int = 2
    ) -> Dict[str, List[Document]]:
        """Retrieve from all document types for comprehensive legal research"""
        results = {
            'case_law': self.retrieve_case_law(query, case_law_k),
            'statutes': self.retrieve_statutes(query, statutes_k),
            'regulations': self.retrieve_regulations(query, regulations_k)
        }
        return results
    
    def get_retriever(self, document_type: str = "all") -> BaseRetriever:
        """Get retriever for specific document type"""
        if document_type == "case_law" and self.case_law_store:
            return self.case_law_store.as_retriever()
        elif document_type == "statutes" and self.statutes_store:
            return self.statutes_store.as_retriever()
        elif document_type == "regulations" and self.regulations_store:
            return self.regulations_store.as_retriever()
        else:
            # Return combined retriever (implement ensemble retriever)
            return self._create_ensemble_retriever()
    
    def _create_ensemble_retriever(self):
        """Create ensemble retriever combining all document types"""
        # This is a simplified version - you might want to use 
        # langchain's EnsembleRetriever for more sophisticated combination
        from langchain.retrievers import EnsembleRetriever
        
        retrievers = []
        weights = []
        
        if self.case_law_store:
            retrievers.append(self.case_law_store.as_retriever())
            weights.append(0.5)  # Higher weight for case law
            
        if self.statutes_store:
            retrievers.append(self.statutes_store.as_retriever())
            weights.append(0.3)
            
        if self.regulations_store:
            retrievers.append(self.regulations_store.as_retriever())
            weights.append(0.2)
        
        if retrievers:
            return EnsembleRetriever(retrievers=retrievers, weights=weights)
        return None

class LegalCitationExtractor:
    """Extract and validate legal citations from text"""
    
    def __init__(self):
        # Common legal citation patterns
        self.citation_patterns = {
            'case_citation': r'\d+\s+\w+\.?\s+\d+',  # e.g., "123 F.3d 456"
            'statute_citation': r'\d+\s+U\.S\.C\.?\s+§?\s*\d+',  # e.g., "42 U.S.C. § 1983"
            'regulation_citation': r'\d+\s+C\.F\.R\.?\s+§?\s*\d+',  # e.g., "29 C.F.R. § 1630"
        }
    
    def extract_citations(self, text: str) -> Dict[str, List[str]]:
        """Extract citations from legal text"""
        import re
        
        citations = {}
        for citation_type, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations[citation_type] = matches
            
        return citations
    
    def validate_citations(self, citations: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate legal citations (simplified implementation)"""
        # In a real implementation, you would check against legal databases
        # like Westlaw, LexisNexis, or free alternatives like Justia
        return citations