#!/usr/bin/env python3
"""
Test script to verify legal framework installation
"""

def test_imports():
    """Test all critical imports"""
    try:
        # Core LangChain
        from langchain_core.messages import HumanMessage
        from langgraph.graph import StateGraph
        print("✓ LangChain/LangGraph imports successful")
        
        # Vector stores
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✓ Vector store imports successful")
        
        # LLM providers (will fail without API keys, but imports should work)
        from langchain_openai import ChatOpenAI
        from langchain_groq import ChatGroq
        print("✓ LLM provider imports successful")
        
        # FastAPI
        from fastapi import FastAPI
        import uvicorn
        print("✓ FastAPI imports successful")
        
        # Document processing
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        print("✓ Document loader imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_embeddings():
    """Test embeddings initialization"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Test embedding
        test_text = "This is a legal document about contracts."
        embedding = embeddings.embed_query(test_text)
        
        print(f"✓ Embeddings working - dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return False

def test_vector_store():
    """Test FAISS vector store"""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        
        # Create sample documents
        docs = [
            Document(page_content="Contract law governs agreements between parties.", 
                    metadata={"source": "contract_law.txt"}),
            Document(page_content="Tort law deals with civil wrongs and damages.",
                    metadata={"source": "tort_law.txt"})
        ]
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Test search
        results = vector_store.similarity_search("What is contract law?", k=1)
        
        print(f"✓ Vector store working - found {len(results)} results")
        print(f"  Sample result: {results[0].page_content[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Legal Framework Installation")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Embeddings", test_embeddings), 
        ("Vector Store", test_vector_store)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n📋 Testing {name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"✅ {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Installation successful! Ready to run legal framework.")
        return True
    else:
        print("❌ Some tests failed. Check error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)