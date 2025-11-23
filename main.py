from lstm_memory import LSTMLongTermMemory
import os
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Legal framework imports
from legal_types import LegalDebateState, ConsensusMetrics, LegalMemory
from consensus_calculator import ConsensusCalculator

# Try to import components with fallbacks
try:
    from legal_rag import LegalRAGSystem
except ImportError:
    from legal_rag import LegalRAGSystem

try:
    from advanced_memory import LegalMemoryManager
except ImportError:
    print("⚠ Advanced memory not found - using basic memory")

# Groq import (primary LLM)
GROQ_AVAILABLE = False
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
    print("Groq LangChain integration available\n")
except ImportError:
    print("Groq not available - install with: pip install langchain-groq")

# Environment setup
from dotenv import load_dotenv
load_dotenv()

class LegalDebateRequest(BaseModel):
    """Request model for legal debate initiation"""
    case_description: str
    legal_question: str
    jurisdiction: str = "indian"
    max_rounds: int = 6
    session_id: Optional[str] = None

class GroqLegalFramework:
    """
    Legal framework optimized for Groq API usage.
    Prioritizes Groq for all LLM operations.
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Legal Judgment Framework (Groq-Powered)", 
            version="1.0.0",
            description="Legal debate system powered by Groq's fast LLM inference"
        )
        
        # Initialize components
        self._setup_groq_llms()
        self._setup_components()
        self._setup_routes()
        
        print("\nGroq-Powered Legal Framework initialized!")
    
    def _setup_groq_llms(self):
        """Initialize Groq LLMs with different models for different tasks"""
        
        groq_api_key = os.getenv('GROQ_API_KEY')
        
        if not groq_api_key:
            print("GROQ_API_KEY not found in .env file!")
            print("Please add: GROQ_API_KEY=your_groq_api_key_here")
            self.llms = {}
            return
        
        if not GROQ_AVAILABLE:
            print("Groq not available. Install with: pip install langchain-groq")
            self.llms = {}
            return
        
        # Initialize different Groq models for different tasks
        try:
            # Primary model for complex legal reasoning
            self.llms = {
                'primary': ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    api_key=groq_api_key,
                    max_tokens=4000
                ),
                'fast': ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    api_key=groq_api_key,
                    max_tokens=2000
                ),
                'creative': ChatGroq(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    temperature=0.4,
                    api_key=groq_api_key,
                    max_tokens=3000
                )
            }

            print(" Groq LLMs initialized:")
            print(f"  - Primary: llama-3.3-70b-versatile")
            print(f"  - Fast: llama-3.1-8b-instant") 
            print(f"  - Creative: meta-llama/llama-4-scout-17b-16e-instruct")
            
        except Exception as e:
            print(f" Groq initialization failed: {e}")
            self.llms = {}
    
    def _setup_components(self):
        """Initialize other components"""
        
        # Setup RAG system
        try:
            legal_docs_path = Path("./legal_documents")
            if legal_docs_path.exists():
                self.rag_system = LegalRAGSystem(
                    case_law_path=str(legal_docs_path / "case_law"),
                    statutes_path=str(legal_docs_path / "statutes"),
                    regulations_path=str(legal_docs_path / "regulations")
                )
                print(" Legal RAG system initialized")
            else:
                print(" Legal documents folder not found - run setup_legal_documents.py")
                self.rag_system = None
        except Exception as e:
            print(f" RAG system failed: {e}")
            self.rag_system = None
        
        # Setup memory (simplified)
        try:
            self.memory_manager = LegalMemoryManager()
            print(" Memory system initialized")
        except:
            self.memory_manager = None
            print(" Using basic memory")
    
        try:
            self.lstm_memory = LSTMLongTermMemory()
            print("✓ LSTM Memory system initialized")
        except Exception as e:
            print(f"⚠ LSTM Memory system failed: {e}")
            self.lstm_memory = None

    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Legal Judgment Framework API (Groq-Powered)",
                "version": "1.0.0",
                "status": "running",
                "llm_provider": "Groq",
                "available_models": list(self.llms.keys()) if self.llms else [],
                "components": {
                    "groq_llms": len(self.llms) > 0,
                    "rag": bool(self.rag_system),
                    "memory": bool(self.memory_manager)
                },
                "endpoints": [
                    "/debate/legal",
                    "/debate/multi_agent",
                    "/health",
                    "/test_groq"
                ]
            }
        
        @self.app.post("/debate/legal")
        async def legal_debate(request: LegalDebateRequest):
            """Run a comprehensive legal debate using Groq models"""
            try:
                if not self.llms:
                    raise HTTPException(
                        status_code=500, 
                        detail="Groq LLMs not available. Please check GROQ_API_KEY in .env file"
                    )
                
                # Run comprehensive legal debate
                debate_result = await self._run_legal_debate(request)
                
                return {
                    "session_id": request.session_id or "groq_legal_debate",
                    "status": "completed",
                    "case_description": request.case_description,
                    "legal_question": request.legal_question,
                    "jurisdiction": request.jurisdiction,
                    "llm_provider": "Groq",
                    "models_used": list(self.llms.keys()),
                    "result": debate_result
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/debate/multi_agent")
        async def multi_agent_debate(request: LegalDebateRequest):
            """Run multi-agent legal debate with different Groq models as agents"""
            try:
                if not self.llms:
                    raise HTTPException(
                        status_code=500, 
                        detail="Groq LLMs not available. Please check GROQ_API_KEY"
                    )
                
                # Run multi-agent debate
                debate_result = await self._run_multi_agent_debate(request)
                
                return {
                    "session_id": request.session_id or "multi_agent_debate",
                    "status": "completed",
                    "debate_type": "multi_agent",
                    "result": debate_result
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/test_groq")
        async def test_groq():
            """Test Groq API connectivity and models"""
            
            if not self.llms:
                return {
                    "status": "failed",
                    "error": "No Groq models initialized",
                    "recommendation": "Check GROQ_API_KEY in .env file"
                }
            
            test_results = {}
            
            for model_name, llm in self.llms.items():
                try:
                    response = await llm.ainvoke([
                        {"role": "user", "content": "What is contract law in one sentence?"}
                    ])
                    test_results[model_name] = {
                        "status": "success",
                        "response": response.content[:100] + "..." if len(response.content) > 100 else response.content
                    }
                except Exception as e:
                    test_results[model_name] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            return {
                "status": "completed",
                "groq_api_key_configured": bool(os.getenv('GROQ_API_KEY')),
                "models_tested": test_results,
                "legal_documents": self._check_legal_documents()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "llm_provider": "Groq",
                "models_available": len(self.llms),
                "components": {
                    "groq": len(self.llms) > 0,
                    "rag": bool(self.rag_system),
                    "memory": bool(self.memory_manager)
                }
            }
    
    async def _run_legal_debate(self, request: LegalDebateRequest) -> Dict[str, Any]:
        """Run comprehensive legal debate using Groq models"""
        
        # Get legal context from RAG
        legal_context = ""
        if self.rag_system:
            try:
                legal_docs = self.rag_system.retrieve_comprehensive(request.legal_question)
                legal_context = self._format_legal_context(legal_docs)
            except Exception as e:
                print(f" RAG retrieval failed: {e}")
        
        # Build comprehensive legal analysis prompt
        analysis_prompt = f"""You are an expert Indian legal advisor. Provide a comprehensive legal analysis for this case:

CASE DETAILS:
- Description: {request.case_description}
- Legal Question: {request.legal_question}
- Jurisdiction: {request.jurisdiction}

{legal_context}

ANALYSIS REQUIREMENTS:
1. **Legal Framework**: Identify applicable laws, sections, and constitutional provisions
2. **Precedent Analysis**: Reference relevant case law and judicial precedents  
3. **Arguments Analysis**: Present arguments from different perspectives (plaintiff, defendant)
4. **Evidence Evaluation**: Assess the strength of available evidence
5. **Legal Conclusion**: Provide reasoned conclusion with supporting rationale

Provide a structured, professional legal analysis with proper citations."""

        # Use primary Groq model for analysis
        try:
            analysis_response = await self.llms['primary'].ainvoke([
                {"role": "system", "content": "You are a senior Indian legal expert with deep knowledge of Indian law, constitution, and judicial precedents."},
                {"role": "user", "content": analysis_prompt}
            ])
            
            return {
                "comprehensive_analysis": analysis_response.content,
                "legal_context_used": bool(legal_context),
                "model_used": "llama-3.3-70b-versatile",
                "method": "comprehensive_groq_analysis"
            }
            
        except Exception as e:
            return {
                "error": f"Legal analysis failed: {e}",
                "method": "fallback"
            }
    
    async def _run_multi_agent_debate(self, request: LegalDebateRequest) -> Dict[str, Any]:
        """Run iterative multi-agent debate until consensus threshold is reached"""
        
        # Initialize consensus calculator
        consensus_calc = ConsensusCalculator(target_consensus=0.7)
        
        # Get legal context from RAG
        legal_context = ""
        if self.rag_system:
            try:
                legal_docs = self.rag_system.retrieve_comprehensive(request.legal_question)
                legal_context = self._format_legal_context(legal_docs)
            except Exception as e:
                print(f" RAG retrieval failed: {e}")
        
        base_context = f"""
CASE: {request.case_description}
LEGAL QUESTION: {request.legal_question}
JURISDICTION: {request.jurisdiction}

{legal_context}
"""
        
        # Store arguments for each round
        plaintiff_arguments = []
        defendant_arguments = []
        consensus_metrics = []
        
        max_rounds = request.max_rounds or 6
        current_round = 1
        
        print(f" Starting multi-round debate (target consensus: 0.7, max rounds: {max_rounds})")
        
        # Initial round - establish positions
        print(f" Round {current_round}: Initial Arguments")
        
        # Plaintiff opening argument
        plaintiff_prompt = f"""{base_context}

You are representing the PLAINTIFF/PETITIONER in Round {current_round}. Present your strongest legal arguments supporting your client's position. Include:
- Legal basis for your claims
- Supporting evidence and precedents  
- Constitutional and statutory provisions
- Case law citations

Focus on establishing your core legal position. Write in clear, structured paragraphs without markdown formatting."""
        
        # Defendant opening argument  
        defendant_prompt = f"""{base_context}

You are representing the DEFENDANT/RESPONDENT in Round {current_round}. Present strong legal arguments defending against the claims. Include:
- Legal defenses available
- Counter-arguments to plaintiff's position
- Distinguishing case law
- Procedural and substantive defenses

Focus on establishing your defensive position. Write in clear, structured paragraphs without markdown formatting."""
        
        try:
            # Run initial arguments in parallel
            plaintiff_task = self.llms['creative'].ainvoke([
                {"role": "system", "content": "You are an experienced advocate representing the plaintiff with expertise in Indian law. Provide clear, well-structured arguments without markdown formatting."},
                {"role": "user", "content": plaintiff_prompt}
            ])
            
            defendant_task = self.llms['primary'].ainvoke([
                {"role": "system", "content": "You are a senior advocate representing the defendant with deep legal knowledge. Provide clear, well-structured arguments without markdown formatting."},
                {"role": "user", "content": defendant_prompt}
            ])
            
            # Wait for initial arguments
            plaintiff_response, defendant_response = await asyncio.gather(plaintiff_task, defendant_task)
            
            # Store initial arguments
            plaintiff_arguments.append(plaintiff_response.content)
            defendant_arguments.append(defendant_response.content)
            
            # Calculate initial consensus
            consensus_result = consensus_calc.calculate_round_consensus(
                plaintiff_arguments, defendant_arguments, current_round
            )
            consensus_metrics.append(consensus_result)
            
            print(f" Round {current_round} consensus: {consensus_result['consensus_score']:.3f}")
            
            # Continue debate rounds until consensus or max rounds
            while (consensus_calc.should_continue_debate(consensus_result) and 
                   current_round < max_rounds):
                
                current_round += 1
                print(f" Round {current_round}: Counter-Arguments & Rebuttals")
                
                # Build context with previous arguments
                debate_history = ""
                for i, (p_arg, d_arg) in enumerate(zip(plaintiff_arguments, defendant_arguments), 1):
                    debate_history += f"\nROUND {i}:\n"
                    debate_history += f"Plaintiff argued: {p_arg[:200]}...\n"
                    debate_history += f"Defendant argued: {d_arg[:200]}...\n"
                
                # Plaintiff rebuttal
                plaintiff_rebuttal_prompt = f"""{base_context}

DEBATE HISTORY:
{debate_history}

You are the PLAINTIFF's advocate in Round {current_round}. The defendant has presented their arguments. Now:

1. Address the defendant's key counter-arguments
2. Strengthen your original position with additional evidence
3. Point out weaknesses in the defendant's reasoning
4. Introduce new legal precedents if needed
5. Show how the law favors your client's position

Latest defendant argument: {defendant_arguments[-1][:500]}...

Provide a focused rebuttal that addresses their points while reinforcing your case. Write in clear paragraphs without markdown."""
                
                # Defendant rebuttal
                defendant_rebuttal_prompt = f"""{base_context}

DEBATE HISTORY:
{debate_history}

You are the DEFENDANT's advocate in Round {current_round}. The plaintiff has reinforced their position. Now:

1. Counter the plaintiff's latest arguments
2. Reinforce your defensive position
3. Highlight procedural or substantive weaknesses in their case
4. Cite distinguishing precedents
5. Show why the law supports your client

Latest plaintiff argument: {plaintiff_arguments[-1][:500]}...

Provide a strong rebuttal that counters their arguments and defends your position. Write in clear paragraphs without markdown."""
                
                # Execute rebuttal round
                plaintiff_rebuttal_task = self.llms['creative'].ainvoke([
                    {"role": "system", "content": f"You are the plaintiff's advocate in round {current_round}. Focus on rebutting the defendant's arguments while strengthening your position. No markdown formatting."},
                    {"role": "user", "content": plaintiff_rebuttal_prompt}
                ])
                
                defendant_rebuttal_task = self.llms['primary'].ainvoke([
                    {"role": "system", "content": f"You are the defendant's advocate in round {current_round}. Counter the plaintiff's arguments effectively. No markdown formatting."},
                    {"role": "user", "content": defendant_rebuttal_prompt}
                ])
                
                # Wait for rebuttals
                plaintiff_rebuttal, defendant_rebuttal = await asyncio.gather(
                    plaintiff_rebuttal_task, defendant_rebuttal_task
                )
                
                # Store new arguments
                plaintiff_arguments.append(plaintiff_rebuttal.content)
                defendant_arguments.append(defendant_rebuttal.content)
                
                # Calculate new consensus
                consensus_result = consensus_calc.calculate_round_consensus(
                    plaintiff_arguments, defendant_arguments, current_round
                )
                consensus_metrics.append(consensus_result)
                
                print(f" Round {current_round} consensus: {consensus_result['consensus_score']:.3f}")
                
                # Check if consensus reached
                if consensus_result['target_reached']:
                    print(f" Consensus threshold reached: {consensus_result['consensus_score']:.3f}")
                    break
            
            # Final judicial decision after consensus/max rounds
            print(f"👩‍⚖️ Rendering final judicial decision...")
            
            # Build complete debate history for judge
            complete_debate_history = ""
            for i, (p_arg, d_arg) in enumerate(zip(plaintiff_arguments, defendant_arguments), 1):
                complete_debate_history += f"\n=== ROUND {i} ===\n"
                complete_debate_history += f"PLAINTIFF ARGUMENTS:\n{p_arg}\n\n"
                complete_debate_history += f"DEFENDANT ARGUMENTS:\n{d_arg}\n\n"
            
            judge_prompt = f"""{base_context}

COMPLETE DEBATE RECORD:
{complete_debate_history}

CONSENSUS ANALYSIS:
- Total rounds: {current_round}
- Final consensus score: {consensus_result['consensus_score']:.3f}
- Target consensus (0.7): {'REACHED' if consensus_result['target_reached'] else 'NOT REACHED'}

You are the presiding JUDGE. After {current_round} rounds of arguments, provide your final judicial decision:

1. Analyze the legal merits of each party's position across all rounds
2. Evaluate how arguments evolved and which side presented stronger case
3. Consider the consensus metrics and quality of legal reasoning
4. Apply relevant legal principles to reach a decision
5. Provide clear rationale for your judgment

Render a comprehensive judicial decision that considers the full debate record. Write in clear, authoritative paragraphs without markdown formatting."""
            
            judge_response = await self.llms['primary'].ainvoke([
                {"role": "system", "content": "You are an experienced High Court judge with expertise in Indian jurisprudence. Provide a well-reasoned judicial decision based on the complete debate record."},
                {"role": "user", "content": judge_prompt}
            ])
            
            # Get consensus summary
            consensus_summary = consensus_calc.get_consensus_summary(consensus_metrics)
            
            print(f"⚖️ Debate completed: {current_round} rounds, consensus: {consensus_result['consensus_score']:.3f}")
            
            # Store debate in LSTM memory for learning
            if self.lstm_memory:
                try:
                    debate_storage_data = {
                        'case_description': request.case_description,
                        'legal_question': request.legal_question,
                        'jurisdiction': request.jurisdiction,
                        'multi_agent_debate': {
                            'debate_rounds': current_round,
                            'all_plaintiff_arguments': plaintiff_arguments,
                            'all_defendant_arguments': defendant_arguments,
                            'consensus_progression': consensus_metrics,
                            'consensus_summary': consensus_summary,
                            'judicial_decision': judge_response.content
                        }
                    }
                    
                    debate_id = self.lstm_memory.store_debate(debate_storage_data)
                    print(f"🧠 Stored debate {debate_id} in LSTM memory")
                    
                    # Auto-train every 10 debates
                    memory_insights = self.lstm_memory.get_memory_insights()
                    total_debates = memory_insights['total_debates']
                    
                    if total_debates % 10 == 0 and total_debates >= 10:
                        print(f"🧠 Auto-training LSTM model with {total_debates} debates...")
                        self.lstm_memory.train_model(epochs=30, batch_size=8)
                        print("✅ LSTM model training completed")
                    
                except Exception as e:
                    print(f"⚠ LSTM storage failed: {e}")
            
            # Return successful debate results
            return {
                "multi_agent_debate": {
                    "plaintiff_arguments": plaintiff_arguments[-1],
                    "defendant_arguments": defendant_arguments[-1],  
                    "judicial_decision": judge_response.content,
                    "debate_rounds": current_round,
                    "all_plaintiff_arguments": plaintiff_arguments,
                    "all_defendant_arguments": defendant_arguments,
                    "consensus_progression": consensus_metrics,
                    "consensus_summary": consensus_summary
                },
                "models_used": {
                    "plaintiff": "meta-llama/llama-4-scout-17b-16e-instruct (creative)",
                    "defendant": "llama-3.3-70b-versatile (primary)", 
                    "judge": "llama-3.3-70b-versatile (primary)"
                },
                "legal_context_used": bool(legal_context),
                "method": "iterative_consensus_debate"
            }
            
        except Exception as e:
            return {
                "error": f"Multi-agent debate failed: {e}",
                "method": "fallback"
            }

        
    def _format_legal_context(self, legal_docs: Dict[str, Any]) -> str:
        """Format retrieved legal documents for context"""
        
        if not legal_docs:
            return ""
        
        context = "\nRELEVANT LEGAL DOCUMENTS:\n"
        
        for doc_type, documents in legal_docs.items():
            if documents:
                context += f"\n{doc_type.upper()}:\n"
                for i, doc in enumerate(documents[:2], 1):  # Limit to 2 docs per type
                    context += f"{i}. {doc.page_content[:300]}...\n"
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        context += f"   Source: {doc.metadata['source']}\n"
        
        return context
    
    def _check_legal_documents(self) -> Dict[str, Any]:
        """Check legal documents availability"""
        
        legal_docs_path = Path("./legal_documents")
        
        if not legal_docs_path.exists():
            return {
                "status": "not_found",
                "recommendation": "Run: python setup_legal_documents.py"
            }
        
        return {
            "status": "available",
            "case_law_files": len(list((legal_docs_path / "case_law").glob("*.txt"))) if (legal_docs_path / "case_law").exists() else 0,
            "statutes_files": len(list((legal_docs_path / "statutes").glob("*.txt"))) if (legal_docs_path / "statutes").exists() else 0,
            "regulations_files": len(list((legal_docs_path / "regulations").glob("*.txt"))) if (legal_docs_path / "regulations").exists() else 0
        }
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server"""
        print(f"\n-> Starting Groq-Powered Legal Framework on http://{host}:{port}")
        print("-> Access API docs at http://localhost:8000/docs")
        print("-> Test Groq connectivity at http://localhost:8000/test_groq")
        print("⚖️ Run legal debate at http://localhost:8000/debate/legal")
        
        uvicorn.run(self.app, host=host, port=port)

def main():
    """Main entry point"""
    
    print("🏛️ Legal Judgment Framework (Groq-Powered)")
    print("=" * 50)
    
    # Check for Groq API key
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        print("❌ GROQ_API_KEY not found!")
        print("\n🔧 Setup instructions:")
        print("1. Get free API key from: https://console.groq.com/")
        print("2. Add to .env file: GROQ_API_KEY=your_key_here")
        print("3. Run: python main.py")
        return
    
    print(f"-> Groq API key configured: {groq_key[:10]}...")
    
    # Check legal documents
    legal_docs_path = Path("./legal_documents")
    if not legal_docs_path.exists():
        print("⚠ Legal documents not found")
        print("📁 Run: python setup_legal_documents.py")
    else:
        print("✓ Legal documents found")
    
    # Initialize framework
    try:
        framework = GroqLegalFramework()
        framework.run_server()
    except KeyboardInterrupt:
        print("\n👋 Shutting down Legal Framework...")
    except Exception as e:
        print(f"\n❌ Error starting framework: {e}")

if __name__ == "__main__":
    main()