from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import AnyMessage, add_messages
import uuid
from datetime import datetime

class LegalPosition(BaseModel):
    """Represents a legal position/stance in the debate"""
    agent_id: str
    position: Literal["plaintiff", "defendant", "prosecution", "defense", "amicus"]
    stance: str
    evidence_citations: List[str] = []
    legal_precedents: List[str] = []

class EvidenceItem(BaseModel):
    """Structured evidence with citations"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: str
    citation: str
    credibility_score: float = 0.0
    legal_relevance: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

class LegalArgument(BaseModel):
    """A legal argument made by an agent"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    content: str
    round_number: int
    argument_type: Literal["opening", "rebuttal", "counter_rebuttal", "closing"]
    evidence_references: List[str] = []
    legal_citations: List[str] = []
    target_agent_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ConsensusMetrics(BaseModel):
    """Tracks convergence towards legal consensus"""
    agreement_score: float = 0.0
    evidence_strength: float = 0.0
    legal_precedent_alignment: float = 0.0
    transparency_score: float = 0.0
    confidence_level: float = 0.0

class LegalMemory(BaseModel):
    """Advanced memory system for legal context"""
    case_facts: List[str] = []
    legal_principles: List[str] = []
    precedent_cases: List[str] = []
    statutory_references: List[str] = []
    expert_opinions: List[str] = []
    
class LegalDebateState(TypedDict):
    """Complete state of the legal debate system"""
    # Core messaging
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Debate management
    current_round: int
    debate_phase: Literal["initialization", "opening_arguments", "evidence_presentation", 
                         "cross_examination", "rebuttals", "closing_arguments", 
                         "deliberation", "verdict"]
    active_agents: List[str]
    
    # Legal context
    case_description: str
    legal_question: str
    applicable_laws: List[str]
    jurisdiction: str
    
    # Agent roles and positions
    agent_positions: Dict[str, LegalPosition]
    
    # Evidence and citations
    evidence_pool: List[EvidenceItem]
    legal_citations: List[str]
    
    # Arguments and rebuttals
    arguments: List[LegalArgument]
    
    # Memory and context
    legal_memory: LegalMemory
    
    # RAG components
    retrieved_documents: List[Dict[str, Any]]
    search_queries: List[str]
    
    # Consensus tracking
    consensus_metrics: ConsensusMetrics
    
    # Orchestration
    next_agent: Optional[str]
    moderator_feedback: List[str]
    
    # Final judgment
    final_verdict: Optional[Dict[str, Any]]
    reasoning_chain: List[str]