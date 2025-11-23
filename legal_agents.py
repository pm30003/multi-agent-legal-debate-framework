from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from legal_types import LegalDebateState, LegalArgument, EvidenceItem
from legal_rag import LegalRAGSystem

class LegalAgent(ABC):
    """Base class for all legal debate agents"""
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseChatModel,
        rag_system: LegalRAGSystem,
        memory: Optional[ConversationBufferMemory] = None
    ):
        self.agent_id = agent_id
        self.llm = llm
        self.rag_system = rag_system
        self.memory = memory or ConversationBufferMemory(return_messages=True)
        self.evidence_citations = []
        
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    @abstractmethod
    async def process(self, state: LegalDebateState) -> Dict[str, Any]:
        """Process the current state and return updates"""
        pass
    
    def retrieve_legal_context(self, query: str) -> Dict[str, List]:
        """Retrieve relevant legal context using RAG"""
        return self.rag_system.retrieve_comprehensive(query)
    
    def format_legal_context(self, context: Dict[str, List]) -> str:
        """Format retrieved legal context for prompt inclusion"""
        formatted = []
        
        for doc_type, documents in context.items():
            if documents:
                formatted.append(f"\n=== {doc_type.upper()} ===")
                for doc in documents:
                    formatted.append(f"- {doc.page_content[:200]}...")
                    if 'source' in doc.metadata:
                        formatted.append(f"  Source: {doc.metadata['source']}")
        
        return "\n".join(formatted)

class PlaintiffAgent(LegalAgent):
    """Agent representing the plaintiff's position"""
    
    def get_system_prompt(self) -> str:
        return """You are an expert plaintiff's attorney in a legal debate. Your role is to:

1. ADVOCATE STRONGLY for your client's position
2. PRESENT COMPELLING EVIDENCE from case law, statutes, and regulations
3. CITE SPECIFIC LEGAL PRECEDENTS with proper citations
4. COUNTER opposing arguments with legal reasoning
5. BUILD toward a transparent consensus based on legal merit

Key responsibilities:
- Use retrieved legal documents to support arguments
- Maintain professional legal discourse
- Cite evidence properly with sources
- Address counterarguments systematically
- Focus on legal principles and precedent

Always structure arguments with:
- Legal theory/principle
- Supporting evidence/precedent
- Application to current case
- Anticipation of counterarguments"""

    async def process(self, state: LegalDebateState) -> Dict[str, Any]:
        """Process current state and generate plaintiff argument"""
        
        # Retrieve relevant legal context
        legal_query = f"{state['legal_question']} {state['case_description']}"
        legal_context = self.retrieve_legal_context(legal_query)
        
        # Format context for prompt
        context_str = self.format_legal_context(legal_context)
        
        # Build prompt with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", f"""
CASE: {state['case_description']}
LEGAL QUESTION: {state['legal_question']}
CURRENT ROUND: {state['current_round']}
PHASE: {state['debate_phase']}

RETRIEVED LEGAL CONTEXT:
{context_str}

RECENT ARGUMENTS:
{self._format_recent_arguments(state['arguments'])}

Based on the legal context and case facts, provide your argument as the plaintiff.
Include specific citations and legal reasoning.
"""),
        ])
        
        # Generate response
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        
        # Create argument record
        argument = LegalArgument(
            agent_id=self.agent_id,
            content=response.content,
            round_number=state['current_round'],
            argument_type=self._determine_argument_type(state),
            evidence_references=[doc.metadata.get('source', '') for docs in legal_context.values() for doc in docs]
        )
        
        return {
            "messages": [HumanMessage(content=response.content, name=self.agent_id)],
            "arguments": state['arguments'] + [argument],
            "retrieved_documents": state['retrieved_documents'] + [
                doc.dict() for docs in legal_context.values() for doc in docs
            ]
        }
    
    def _format_recent_arguments(self, arguments: List[LegalArgument]) -> str:
        """Format recent arguments for context"""
        if not arguments:
            return "No previous arguments."
        
        recent = arguments[-3:]  # Last 3 arguments
        formatted = []
        for arg in recent:
            formatted.append(f"{arg.agent_id}: {arg.content[:200]}...")
        
        return "\n".join(formatted)
    
    def _determine_argument_type(self, state: LegalDebateState) -> str:
        """Determine the type of argument based on debate phase"""
        phase_mapping = {
            "opening_arguments": "opening",
            "evidence_presentation": "opening", 
            "rebuttals": "rebuttal",
            "closing_arguments": "closing"
        }
        return phase_mapping.get(state['debate_phase'], "opening")

class DefendantAgent(LegalAgent):
    """Agent representing the defendant's position"""
    
    def get_system_prompt(self) -> str:
        return """You are an expert defense attorney in a legal debate. Your role is to:

1. DEFEND your client's position vigorously
2. CHALLENGE the plaintiff's evidence and arguments
3. PRESENT ALTERNATIVE LEGAL INTERPRETATIONS
4. CITE DISTINGUISHING PRECEDENTS and statutory defenses
5. EXPOSE weaknesses in opposing arguments

Key responsibilities:
- Question the strength and relevance of opposing evidence
- Present alternative legal theories
- Cite contradictory or distinguishing precedents
- Use procedural and substantive defenses
- Maintain focus on burden of proof

Defense strategy:
- Challenge factual assumptions
- Present alternative legal interpretations
- Cite distinguishing cases
- Highlight procedural deficiencies
- Focus on reasonable doubt or burden not met"""

    async def process(self, state: LegalDebateState) -> Dict[str, Any]:
        """Process current state and generate defense argument"""
        
        # Analyze plaintiff arguments for counter-argument strategy
        plaintiff_args = [arg for arg in state['arguments'] if 'plaintiff' in arg.agent_id]
        
        # Build defensive query including plaintiff claims
        defensive_query = f"defense {state['legal_question']} counter-arguments"
        if plaintiff_args:
            latest_plaintiff = plaintiff_args[-1]
            defensive_query += f" challenging {latest_plaintiff.content[:100]}"
        
        # Retrieve defensive legal context
        legal_context = self.retrieve_legal_context(defensive_query)
        context_str = self.format_legal_context(legal_context)
        
        # Build defensive prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", f"""
CASE: {state['case_description']}
LEGAL QUESTION: {state['legal_question']}
CURRENT ROUND: {state['current_round']}
PHASE: {state['debate_phase']}

RETRIEVED DEFENSIVE CONTEXT:
{context_str}

PLAINTIFF'S RECENT ARGUMENTS:
{self._format_plaintiff_arguments(plaintiff_args)}

Provide your defense argument. Focus on:
1. Challenging plaintiff's evidence
2. Presenting alternative legal theories
3. Citing distinguishing precedents
4. Highlighting burden of proof issues
"""),
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        
        # Determine target of rebuttal
        target_agent = plaintiff_args[-1].agent_id if plaintiff_args else None
        
        argument = LegalArgument(
            agent_id=self.agent_id,
            content=response.content,
            round_number=state['current_round'],
            argument_type=self._determine_argument_type(state),
            target_agent_id=target_agent,
            evidence_references=[doc.metadata.get('source', '') for docs in legal_context.values() for doc in docs]
        )
        
        return {
            "messages": [HumanMessage(content=response.content, name=self.agent_id)],
            "arguments": state['arguments'] + [argument],
            "retrieved_documents": state['retrieved_documents'] + [
                doc.dict() for docs in legal_context.values() for doc in docs
            ]
        }
    
    def _format_plaintiff_arguments(self, plaintiff_args: List[LegalArgument]) -> str:
        """Format plaintiff arguments for defensive analysis"""
        if not plaintiff_args:
            return "No plaintiff arguments yet."
        
        formatted = []
        for arg in plaintiff_args[-2:]:  # Last 2 plaintiff arguments
            formatted.append(f"PLAINTIFF CLAIM: {arg.content}")
            if arg.evidence_references:
                formatted.append(f"EVIDENCE CITED: {', '.join(arg.evidence_references[:3])}")
        
        return "\n\n".join(formatted)

class ModeratorAgent(LegalAgent):
    """Agent acting as judge/moderator to guide debate and assess consensus"""
    
    def get_system_prompt(self) -> str:
        return """You are an expert legal moderator and judge overseeing a structured legal debate. Your role is to:

1. EVALUATE the strength and validity of legal arguments
2. IDENTIFY when sufficient evidence has been presented
3. ASSESS convergence toward legal consensus
4. GUIDE the debate toward transparent resolution
5. PROVIDE BALANCED ANALYSIS of competing positions

Key responsibilities:
- Evaluate legal reasoning quality
- Check citation accuracy and relevance
- Identify logical fallacies or weak arguments
- Determine when debate should progress to next phase
- Synthesize arguments toward consensus

Judicial analysis framework:
- Assess evidence quality and relevance
- Evaluate legal precedent application
- Check for procedural compliance
- Identify areas of agreement/disagreement  
- Guide toward principled resolution"""

    async def process(self, state: LegalDebateState) -> Dict[str, Any]:
        """Moderate the debate and assess progress toward consensus"""
        
        # Analyze all arguments for consensus indicators
        all_arguments = state['arguments']
        recent_args = all_arguments[-4:] if len(all_arguments) >= 4 else all_arguments
        
        # Query for judicial guidance
        judicial_query = f"judicial analysis {state['legal_question']} precedent consensus"
        legal_context = self.retrieve_legal_context(judicial_query)
        context_str = self.format_legal_context(legal_context)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", f"""
CASE: {state['case_description']}
LEGAL QUESTION: {state['legal_question']}
CURRENT PHASE: {state['debate_phase']}
ROUND: {state['current_round']}

LEGAL CONTEXT FOR ANALYSIS:
{context_str}

RECENT ARGUMENTS TO ANALYZE:
{self._format_arguments_for_analysis(recent_args)}

CURRENT CONSENSUS METRICS:
- Agreement Score: {state['consensus_metrics'].agreement_score}
- Evidence Strength: {state['consensus_metrics'].evidence_strength}
- Legal Precedent Alignment: {state['consensus_metrics'].legal_precedent_alignment}

Please provide:
1. Analysis of argument quality and legal reasoning
2. Assessment of evidence presented
3. Evaluation of consensus progress
4. Guidance for next phase or final judgment
5. Updated consensus metrics

Should the debate continue or move toward resolution?
"""),
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        
        # Extract consensus metrics (simplified - would use structured output)
        updated_metrics = self._extract_consensus_metrics(response.content, state['consensus_metrics'])
        
        # Determine next phase
        next_phase = self._determine_next_phase(state, updated_metrics)
        
        return {
            "messages": [HumanMessage(content=response.content, name=self.agent_id)],
            "moderator_feedback": state['moderator_feedback'] + [response.content],
            "consensus_metrics": updated_metrics,
            "debate_phase": next_phase,
            "current_round": state['current_round'] + 1 if next_phase != state['debate_phase'] else state['current_round']
        }
    
    def _format_arguments_for_analysis(self, arguments: List[LegalArgument]) -> str:
        """Format arguments for judicial analysis"""
        if not arguments:
            return "No arguments to analyze."
        
        formatted = []
        for arg in arguments:
            formatted.append(f"""
AGENT: {arg.agent_id}
TYPE: {arg.argument_type}
CONTENT: {arg.content}
EVIDENCE: {', '.join(arg.evidence_references[:2])}
---""")
        
        return "\n".join(formatted)
    
    def _extract_consensus_metrics(self, response_content: str, current_metrics) -> 'ConsensusMetrics':
        """Extract updated consensus metrics from moderator response"""
        # Simplified implementation - would use structured output or parsing
        from legal_types import ConsensusMetrics
        
        # For now, incrementally update based on content analysis
        new_metrics = ConsensusMetrics(
            agreement_score=min(current_metrics.agreement_score + 0.1, 1.0),
            evidence_strength=min(current_metrics.evidence_strength + 0.15, 1.0),
            legal_precedent_alignment=min(current_metrics.legal_precedent_alignment + 0.1, 1.0),
            transparency_score=min(current_metrics.transparency_score + 0.1, 1.0),
            confidence_level=min(current_metrics.confidence_level + 0.05, 1.0)
        )
        
        return new_metrics
    
    def _determine_next_phase(self, state: LegalDebateState, metrics) -> str:
        """Determine if debate should progress to next phase"""
        current_phase = state['debate_phase']
        
        # Simple phase progression logic
        if metrics.consensus_metrics.confidence_level > 0.8:
            return "verdict"
        elif current_phase == "opening_arguments" and state['current_round'] >= 2:
            return "evidence_presentation"
        elif current_phase == "evidence_presentation" and state['current_round'] >= 4:
            return "rebuttals"
        elif current_phase == "rebuttals" and state['current_round'] >= 6:
            return "closing_arguments"
        elif current_phase == "closing_arguments":
            return "deliberation"
        
        return current_phase