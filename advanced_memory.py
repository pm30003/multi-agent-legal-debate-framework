from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from legal_types import LegalArgument, EvidenceItem, LegalMemory

# Optional imports for advanced features
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Using in-memory storage")

try:
    from langchain_postgres import PostgresChatMessageHistory
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("Using in-memory storage")

class AdvancedLegalMemory:
    """
    Advanced memory system for legal debates with optional database backends.
    Falls back to in-memory storage if databases are not available.
    """
    
    def __init__(
        self,
        postgres_connection_string: Optional[str] = None,
        redis_connection: Optional[str] = None,
        session_id: str = "legal_debate_session"
    ):
        self.session_id = session_id
        
        # Initialize storage backends
        self.postgres_connection = postgres_connection_string
        self.redis_client = self._init_redis(redis_connection) if REDIS_AVAILABLE else None
        
        # Memory components
        self.working_memory = ConversationBufferWindowMemory(k=10, return_messages=True)
        self.long_term_memory = self._init_postgres_memory() if POSTGRES_AVAILABLE and postgres_connection_string else None
        
        # Legal-specific memory structures (fallback to in-memory)
        self.evidence_memory: Dict[str, EvidenceItem] = {}
        self.precedent_memory: Dict[str, Dict] = {}
        self.argument_chains: Dict[str, List[str]] = {}
        self.consensus_history: List[Dict] = []
        
        # In-memory fallback for database operations
        self.memory_store: Dict[str, Any] = {}
        
        print(f"✓ Legal memory initialized (Redis: {bool(self.redis_client)}, PostgreSQL: {bool(self.long_term_memory)})")
    
    def _init_redis(self, redis_connection: Optional[str]):
        """Initialize Redis connection with error handling"""
        try:
            if redis_connection:
                return redis.from_url(redis_connection)
            else:
                return redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        except Exception as e:
            print(f"⚠ Redis connection failed: {e}")
            return None
    
    def _init_postgres_memory(self):
        """Initialize PostgreSQL-based persistent memory with error handling"""
        try:
            if POSTGRES_AVAILABLE and self.postgres_connection:
                return PostgresChatMessageHistory(
                    connection_string=self.postgres_connection,
                    session_id=self.session_id
                )
        except Exception as e:
            print(f"⚠ PostgreSQL connection failed: {e}")
        return None
    
    def store_argument(self, argument: LegalArgument) -> None:
        """Store legal argument in available storage"""
        
        # Add to working memory
        self.working_memory.chat_memory.add_message(
            HumanMessage(content=argument.content, name=argument.agent_id)
        )
        
        # Store in Redis if available
        if self.redis_client:
            try:
                argument_key = f"argument:{argument.id}"
                self.redis_client.hset(argument_key, mapping={
                    "agent_id": argument.agent_id,
                    "content": argument.content,
                    "round_number": argument.round_number,
                    "argument_type": argument.argument_type,
                    "evidence_references": json.dumps(argument.evidence_references),
                    "legal_citations": json.dumps(argument.legal_citations),
                    "timestamp": argument.timestamp.isoformat()
                })
            except Exception as e:
                print(f"⚠ Redis storage failed: {e}")
        
        # Fallback to in-memory storage
        self.memory_store[f"argument:{argument.id}"] = {
            "agent_id": argument.agent_id,
            "content": argument.content,
            "round_number": argument.round_number,
            "argument_type": argument.argument_type,
            "evidence_references": argument.evidence_references,
            "legal_citations": argument.legal_citations,
            "timestamp": argument.timestamp.isoformat()
        }
        
        # Build argument chains for reasoning tracking
        agent_chain_key = f"chain:{argument.agent_id}"
        if agent_chain_key not in self.argument_chains:
            self.argument_chains[agent_chain_key] = []
        self.argument_chains[agent_chain_key].append(argument.id)
        
        # Store in PostgreSQL if available
        if self.long_term_memory:
            try:
                self.long_term_memory.add_message(
                    AIMessage(
                        content=f"Legal argument by {argument.agent_id}: {argument.content}",
                        additional_kwargs={
                            "argument_type": argument.argument_type,
                            "round": argument.round_number,
                            "evidence_refs": argument.evidence_references
                        }
                    )
                )
            except Exception as e:
                print(f"⚠ PostgreSQL storage failed: {e}")
    
    def store_evidence(self, evidence: EvidenceItem) -> None:
        """Store evidence item with metadata and scoring"""
        
        # Store in memory dictionary
        self.evidence_memory[evidence.id] = evidence
        
        # Store in Redis if available
        if self.redis_client:
            try:
                evidence_key = f"evidence:{evidence.id}"
                evidence_data = {
                    "content": evidence.content,
                    "source": evidence.source,
                    "citation": evidence.citation,
                    "credibility_score": evidence.credibility_score,
                    "legal_relevance": evidence.legal_relevance,
                    "timestamp": evidence.timestamp.isoformat()
                }
                
                self.redis_client.hset(evidence_key, mapping=evidence_data)
                self.redis_client.expire(evidence_key, timedelta(days=7))
                
                # Index by credibility score
                self.redis_client.zadd(
                    "evidence:by_credibility", 
                    {evidence.id: evidence.credibility_score}
                )
            except Exception as e:
                print(f"⚠ Redis evidence storage failed: {e}")
        
        # Fallback to in-memory storage
        self.memory_store[f"evidence:{evidence.id}"] = {
            "content": evidence.content,
            "source": evidence.source,
            "citation": evidence.citation,
            "credibility_score": evidence.credibility_score,
            "legal_relevance": evidence.legal_relevance,
            "timestamp": evidence.timestamp.isoformat()
        }
    
    def store_legal_precedent(self, precedent_id: str, precedent_data: Dict[str, Any]) -> None:
        """Store legal precedent for future reference"""
        
        self.precedent_memory[precedent_id] = precedent_data
        
        # Store in Redis if available
        if self.redis_client:
            try:
                precedent_key = f"precedent:{precedent_id}"
                self.redis_client.hset(precedent_key, mapping={
                    k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    for k, v in precedent_data.items()
                })
                self.redis_client.expire(precedent_key, timedelta(days=30))
            except Exception as e:
                print(f"⚠ Redis precedent storage failed: {e}")
        
        # Fallback to in-memory storage
        self.memory_store[f"precedent:{precedent_id}"] = precedent_data
    
    def update_consensus_tracking(self, consensus_metrics: Dict[str, float]) -> None:
        """Update consensus progression tracking"""
        
        consensus_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": consensus_metrics,
            "round": len(self.consensus_history) + 1
        }
        
        self.consensus_history.append(consensus_entry)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                consensus_key = f"consensus:{self.session_id}"
                self.redis_client.lpush(consensus_key, json.dumps(consensus_entry))
                self.redis_client.ltrim(consensus_key, 0, 50)  # Keep last 50 entries
            except Exception as e:
                print(f"⚠ Redis consensus storage failed: {e}")
    
    def get_argument_chain(self, agent_id: str) -> List[str]:
        """Get argument chain for an agent"""
        return self.argument_chains.get(f"chain:{agent_id}", [])
    
    def get_recent_arguments(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get n most recent arguments"""
        recent_args = []
        
        # Try Redis first
        if self.redis_client:
            try:
                for key in self.redis_client.scan_iter(match="argument:*"):
                    arg_data = self.redis_client.hgetall(key)
                    if arg_data:
                        recent_args.append(arg_data)
            except Exception as e:
                print(f"⚠ Redis retrieval failed: {e}")
        
        # Fallback to in-memory storage
        if not recent_args:
            for key, value in self.memory_store.items():
                if key.startswith("argument:"):
                    recent_args.append(value)
        
        # Sort by timestamp and return n most recent
        recent_args.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return recent_args[:n]
    
    def get_best_evidence(self, n: int = 5, by: str = "credibility") -> List[EvidenceItem]:
        """Get best evidence items by score"""
        
        # Try Redis first
        if self.redis_client:
            try:
                index_key = f"evidence:by_{by}"
                evidence_ids = self.redis_client.zrevrange(index_key, 0, n-1)
                
                evidence_items = []
                for evidence_id in evidence_ids:
                    if evidence_id in self.evidence_memory:
                        evidence_items.append(self.evidence_memory[evidence_id])
                
                if evidence_items:
                    return evidence_items
            except Exception as e:
                print(f"⚠ Redis evidence retrieval failed: {e}")
        
        # Fallback to in-memory storage
        evidence_items = list(self.evidence_memory.values())
        if by == "credibility":
            evidence_items.sort(key=lambda x: x.credibility_score, reverse=True)
        elif by == "relevance":
            evidence_items.sort(key=lambda x: x.legal_relevance, reverse=True)
        
        return evidence_items[:n]
    
    def get_relevant_precedents(self, query_keywords: List[str], n: int = 3) -> List[Dict]:
        """Get relevant precedents based on keywords"""
        
        relevant_precedents = []
        
        for precedent_id, precedent_data in self.precedent_memory.items():
            # Simple keyword matching
            precedent_text = json.dumps(precedent_data).lower()
            relevance_score = sum(1 for keyword in query_keywords if keyword.lower() in precedent_text)
            
            if relevance_score > 0:
                precedent_with_score = precedent_data.copy()
                precedent_with_score['relevance_score'] = relevance_score
                precedent_with_score['precedent_id'] = precedent_id
                relevant_precedents.append(precedent_with_score)
        
        # Sort by relevance and return top n
        relevant_precedents.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_precedents[:n]
    
    def get_consensus_progression(self) -> List[Dict]:
        """Get consensus progression over time"""
        return self.consensus_history.copy()
    
    def get_contextual_memory(self, context_type: str = "recent") -> Dict[str, Any]:
        """Get contextual memory for agent prompts"""
        
        if context_type == "recent":
            return {
                "recent_arguments": self.get_recent_arguments(3),
                "best_evidence": self.get_best_evidence(3, "credibility"),
                "working_memory": self.working_memory.buffer_as_messages,
                "consensus_trend": self.consensus_history[-3:] if self.consensus_history else []
            }
        
        elif context_type == "comprehensive":
            return {
                "all_arguments": self.get_recent_arguments(10),
                "all_evidence": self.get_best_evidence(10, "relevance"),
                "precedent_summary": list(self.precedent_memory.keys()),
                "full_consensus_history": self.consensus_history,
                "argument_chains": self.argument_chains
            }
        
        return {}
    
    def clear_session(self) -> None:
        """Clear current session memory"""
        
        # Clear working memory
        self.working_memory.clear()
        
        # Clear Redis keys if available
        if self.redis_client:
            try:
                for key in self.redis_client.scan_iter(match=f"*:{self.session_id}*"):
                    self.redis_client.delete(key)
            except Exception as e:
                print(f"⚠ Redis cleanup failed: {e}")
        
        # Clear local memory structures
        self.evidence_memory.clear()
        self.precedent_memory.clear()
        self.argument_chains.clear()
        self.consensus_history.clear()
        self.memory_store.clear()
        
    def export_session_data(self) -> Dict[str, Any]:
        """Export all session data for analysis or archival"""
        
        return {
            "session_id": self.session_id,
            "export_timestamp": datetime.now().isoformat(),
            "arguments": self.get_recent_arguments(100),  # All arguments
            "evidence": [item.dict() for item in self.evidence_memory.values()],
            "precedents": self.precedent_memory,
            "argument_chains": self.argument_chains,
            "consensus_history": self.consensus_history,
            "working_memory": [msg.dict() for msg in self.working_memory.buffer_as_messages]
        }

class LegalMemoryManager:
    """Manager class for coordinating different memory systems"""
    
    def __init__(self):
        self.active_sessions: Dict[str, AdvancedLegalMemory] = {}
    
    def create_session(self, session_id: str, **kwargs) -> AdvancedLegalMemory:
        """Create new legal memory session"""
        
        memory_system = AdvancedLegalMemory(session_id=session_id, **kwargs)
        self.active_sessions[session_id] = memory_system
        return memory_system
    
    def get_session(self, session_id: str) -> Optional[AdvancedLegalMemory]:
        """Get existing memory session"""
        return self.active_sessions.get(session_id)
    
    def close_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Close session and return export data"""
        
        if session_id in self.active_sessions:
            memory_system = self.active_sessions[session_id]
            export_data = memory_system.export_session_data()
            memory_system.clear_session()
            del self.active_sessions[session_id]
            return export_data
        
        return None