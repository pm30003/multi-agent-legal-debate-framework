from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from legal_types import LegalDebateState, ConsensusMetrics, LegalMemory
from legal_agents import PlaintiffAgent, DefendantAgent, ModeratorAgent
from legal_rag import LegalRAGSystem

class LegalDebateOrchestrator:
    """
    LangGraph-based orchestration system for legal multi-agent debates.
    Manages workflow, consensus building, and transparent judgment.
    """
    
    def __init__(
        self,
        plaintiff_agent: PlaintiffAgent,
        defendant_agent: DefendantAgent,
        moderator_agent: ModeratorAgent,
        rag_system: LegalRAGSystem
    ):
        self.plaintiff = plaintiff_agent
        self.defendant = defendant_agent
        self.moderator = moderator_agent
        self.rag_system = rag_system
        
        # Memory for persistence
        self.memory = MemorySaver()
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for legal debate"""
        
        # Initialize graph with legal debate state
        workflow = StateGraph(LegalDebateState)
        
        # Add agent nodes
        workflow.add_node("plaintiff_agent", self._plaintiff_node)
        workflow.add_node("defendant_agent", self._defendant_node) 
        workflow.add_node("moderator_agent", self._moderator_node)
        workflow.add_node("rag_retrieval", self._rag_node)
        workflow.add_node("consensus_check", self._consensus_node)
        workflow.add_node("final_judgment", self._judgment_node)
        
        # Define workflow edges and conditional routing
        workflow.add_edge(START, "moderator_agent")  # Begin with moderator setup
        
        # Moderator decides next action
        workflow.add_conditional_edges(
            "moderator_agent",
            self._route_from_moderator,
            {
                "plaintiff": "plaintiff_agent",
                "defendant": "defendant_agent",
                "consensus": "consensus_check",
                "judgment": "final_judgment",
                "rag": "rag_retrieval"
            }
        )
        
        # Agent flows
        workflow.add_conditional_edges(
            "plaintiff_agent",
            self._route_from_agent,
            {
                "defendant": "defendant_agent",
                "moderator": "moderator_agent",
                "rag": "rag_retrieval"
            }
        )
        
        workflow.add_conditional_edges(
            "defendant_agent", 
            self._route_from_agent,
            {
                "plaintiff": "plaintiff_agent",
                "moderator": "moderator_agent",
                "rag": "rag_retrieval"
            }
        )
        
        # RAG retrieval flows back to requesting agent
        workflow.add_conditional_edges(
            "rag_retrieval",
            self._route_from_rag,
            {
                "plaintiff": "plaintiff_agent",
                "defendant": "defendant_agent",
                "moderator": "moderator_agent"
            }
        )
        
        # Consensus check
        workflow.add_conditional_edges(
            "consensus_check",
            self._route_from_consensus,
            {
                "continue": "moderator_agent",
                "judgment": "final_judgment"
            }
        )
        
        # Final judgment ends workflow
        workflow.add_edge("final_judgment", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    # Node implementations
    async def _plaintiff_node(self, state: LegalDebateState) -> LegalDebateState:
        """Process plaintiff agent"""
        result = await self.plaintiff.process(state)
        return {**state, **result}
    
    async def _defendant_node(self, state: LegalDebateState) -> LegalDebateState:
        """Process defendant agent"""
        result = await self.defendant.process(state)
        return {**state, **result}
    
    async def _moderator_node(self, state: LegalDebateState) -> LegalDebateState:
        """Process moderator agent"""
        result = await self.moderator.process(state)
        return {**state, **result}
    
    async def _rag_node(self, state: LegalDebateState) -> LegalDebateState:
        """Handle RAG retrieval requests"""
        # Get last message to determine search query
        if state["messages"]:
            last_message = state["messages"][-1]
            query = last_message.content
            
            # Perform comprehensive legal search
            retrieved_docs = self.rag_system.retrieve_comprehensive(query)
            
            # Format retrieved documents
            formatted_context = []
            for doc_type, documents in retrieved_docs.items():
                for doc in documents:
                    formatted_context.append({
                        "type": doc_type,
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
            
            return {
                **state,
                "retrieved_documents": state["retrieved_documents"] + formatted_context,
                "search_queries": state["search_queries"] + [query]
            }
        
        return state
    
    async def _consensus_node(self, state: LegalDebateState) -> LegalDebateState:
        """Check consensus and debate progression"""
        
        metrics = state["consensus_metrics"]
        
        # Evaluate consensus criteria
        consensus_threshold = 0.8
        evidence_threshold = 0.7
        rounds_threshold = 8
        
        consensus_reached = (
            metrics.confidence_level > consensus_threshold and
            metrics.evidence_strength > evidence_threshold and
            state["current_round"] >= rounds_threshold
        )
        
        if consensus_reached:
            return {
                **state,
                "debate_phase": "verdict",
                "next_agent": "final_judgment"
            }
        else:
            return {
                **state,
                "next_agent": "moderator_agent"
            }
    
    async def _judgment_node(self, state: LegalDebateState) -> LegalDebateState:
        """Generate final legal judgment"""
        
        # Compile all arguments and evidence
        all_arguments = state["arguments"]
        all_evidence = state["retrieved_documents"]
        
        # Build comprehensive judgment prompt
        judgment_prompt = f"""
FINAL LEGAL JUDGMENT

CASE: {state['case_description']}
LEGAL QUESTION: {state['legal_question']}
JURISDICTION: {state['jurisdiction']}

ARGUMENTS PRESENTED:
{self._format_all_arguments(all_arguments)}

EVIDENCE AND PRECEDENTS:
{self._format_all_evidence(all_evidence)}

CONSENSUS METRICS:
- Agreement Score: {state['consensus_metrics'].agreement_score:.2f}
- Evidence Strength: {state['consensus_metrics'].evidence_strength:.2f}  
- Legal Precedent Alignment: {state['consensus_metrics'].legal_precedent_alignment:.2f}
- Transparency Score: {state['consensus_metrics'].transparency_score:.2f}
- Confidence Level: {state['consensus_metrics'].confidence_level:.2f}

Provide a comprehensive legal judgment that:
1. Summarizes the key legal arguments from both sides
2. Analyzes the strength of evidence presented
3. Applies relevant legal precedents and statutes
4. Reaches a reasoned conclusion based on legal merit
5. Explains the reasoning chain transparently
"""
        
        # Generate final judgment using moderator
        judgment_response = await self.moderator.llm.ainvoke([
            HumanMessage(content=judgment_prompt)
        ])
        
        # Create final verdict
        final_verdict = {
            "verdict": judgment_response.content,
            "decision_basis": "legal_precedent_and_evidence",
            "confidence_score": state["consensus_metrics"].confidence_level,
            "supporting_evidence": [doc["metadata"].get("source", "") for doc in all_evidence[:5]],
            "reasoning_chain": state["reasoning_chain"] + ["Final synthesis of all arguments and evidence"]
        }
        
        return {
            **state,
            "final_verdict": final_verdict,
            "messages": state["messages"] + [HumanMessage(content=judgment_response.content, name="final_judgment")]
        }
    
    # Routing functions
    def _route_from_moderator(self, state: LegalDebateState) -> str:
        """Route from moderator based on debate phase and consensus"""
        
        phase = state["debate_phase"]
        consensus = state["consensus_metrics"]
        
        # Check if ready for judgment
        if consensus.confidence_level > 0.8 or phase == "verdict":
            return "judgment"
        
        # Check if need more evidence
        if consensus.evidence_strength < 0.5:
            return "rag"
        
        # Route based on phase and turn
        if phase in ["opening_arguments", "evidence_presentation"]:
            # Alternate between agents
            last_speaker = self._get_last_speaker(state)
            return "defendant" if last_speaker == "plaintiff" else "plaintiff"
        
        elif phase == "rebuttals":
            return "plaintiff" if state["current_round"] % 2 == 1 else "defendant"
        
        else:
            return "consensus"
    
    def _route_from_agent(self, state: LegalDebateState) -> str:
        """Route from agent back to moderator or opponent"""
        
        # Check if agent requested RAG retrieval
        if state["messages"] and "retrieve" in state["messages"][-1].content.lower():
            return "rag"
        
        # Usually return to moderator for orchestration
        return "moderator"
    
    def _route_from_rag(self, state: LegalDebateState) -> str:
        """Route back to requesting agent after RAG retrieval"""
        
        # Determine which agent requested RAG
        if state["search_queries"]:
            # Simple heuristic - return to moderator for orchestration
            return "moderator"
        
        return "moderator"
    
    def _route_from_consensus(self, state: LegalDebateState) -> str:
        """Route based on consensus assessment"""
        
        if state.get("next_agent") == "final_judgment":
            return "judgment"
        else:
            return "continue"
    
    # Helper functions
    def _get_last_speaker(self, state: LegalDebateState) -> Optional[str]:
        """Get the last agent who spoke"""
        if state["arguments"]:
            return state["arguments"][-1].agent_id
        return None
    
    def _format_all_arguments(self, arguments: List) -> str:
        """Format all arguments for judgment"""
        if not arguments:
            return "No arguments presented."
        
        formatted = []
        for arg in arguments:
            formatted.append(f"""
AGENT: {arg.agent_id}
ROUND: {arg.round_number}
TYPE: {arg.argument_type}
CONTENT: {arg.content}
EVIDENCE: {', '.join(arg.evidence_references[:3])}
---""")
        
        return "\n".join(formatted)
    
    def _format_all_evidence(self, evidence: List[Dict]) -> str:
        """Format all retrieved evidence"""
        if not evidence:
            return "No evidence retrieved."
        
        formatted = []
        for item in evidence[:10]:  # Limit to top 10
            formatted.append(f"""
TYPE: {item.get('type', 'unknown')}
CONTENT: {item.get('content', '')[:200]}...
SOURCE: {item.get('metadata', {}).get('source', 'unknown')}
---""")
        
        return "\n".join(formatted)
    
    # Main execution method
    async def run_legal_debate(
        self,
        case_description: str,
        legal_question: str,
        jurisdiction: str = "federal",
        max_rounds: int = 10
    ) -> Dict[str, Any]:
        """
        Run a complete legal debate to consensus and judgment
        
        Args:
            case_description: Description of the legal case
            legal_question: Core legal question to be resolved
            jurisdiction: Legal jurisdiction (federal, state, etc.)
            max_rounds: Maximum number of debate rounds
            
        Returns:
            Complete debate results including final verdict
        """
        
        # Initialize debate state
        initial_state = LegalDebateState(
            messages=[HumanMessage(content=f"Legal debate initiated: {legal_question}")],
            current_round=1,
            debate_phase="initialization",
            active_agents=["plaintiff_agent", "defendant_agent", "moderator_agent"],
            case_description=case_description,
            legal_question=legal_question,
            applicable_laws=[],
            jurisdiction=jurisdiction,
            agent_positions={},
            evidence_pool=[],
            legal_citations=[],
            arguments=[],
            legal_memory=LegalMemory(),
            retrieved_documents=[],
            search_queries=[],
            consensus_metrics=ConsensusMetrics(),
            next_agent=None,
            moderator_feedback=[],
            final_verdict=None,
            reasoning_chain=["Debate initiated"]
        )
        
        # Configuration for execution
        config = {"configurable": {"thread_id": "legal_debate_001"}}
        
        # Execute debate workflow
        final_state = await self.graph.ainvoke(initial_state, config)
        
        # Return comprehensive results
        return {
            "final_verdict": final_state.get("final_verdict"),
            "total_rounds": final_state.get("current_round", 0),
            "arguments_presented": len(final_state.get("arguments", [])),
            "evidence_retrieved": len(final_state.get("retrieved_documents", [])),
            "consensus_metrics": final_state.get("consensus_metrics"),
            "debate_history": final_state.get("messages", []),
            "legal_citations": final_state.get("legal_citations", []),
            "reasoning_chain": final_state.get("reasoning_chain", [])
        }