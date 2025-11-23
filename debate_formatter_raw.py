#!/usr/bin/env python3
"""
Raw Output Debate Formatter - Preserves All Original Content
Shows complete judicial decisions and arguments without truncation
"""

import re
from typing import Dict, Any, List

def clean_and_format_debate(result: Dict[str, Any]) -> str:
    """
    Format multi-round debate result preserving ALL original content
    """
    if "multi_agent_debate" not in result:
        return "No multi-agent debate data found"
    
    debate = result["multi_agent_debate"]
    
    # Extract debate information
    rounds = debate.get("debate_rounds", 1)
    consensus_summary = debate.get("consensus_summary", {})
    consensus_progression = debate.get("consensus_progression", [])
    all_plaintiff_args = debate.get("all_plaintiff_arguments", [])
    all_defendant_args = debate.get("all_defendant_arguments", [])
    judicial_decision = debate.get("judicial_decision", "")  # Keep original, raw decision
    
    # Build formatted output with header
    formatted_output = f"""🔄 MULTI-ROUND LEGAL DEBATE RESULTS
Total Rounds: {rounds}
Final Consensus Score: {consensus_summary.get('final_consensus_score', 'N/A')}
Consensus Target: {consensus_summary.get('target_consensus', 0.7)}
Status: {'✅ CONSENSUS REACHED' if consensus_summary.get('consensus_reached') else '❌ MAX ROUNDS REACHED'}

{format_consensus_progression(consensus_progression)}

{format_all_rounds_raw(all_plaintiff_args, all_defendant_args, consensus_progression)}

👩‍⚖️ FINAL JUDICIAL DECISION
Judge, High Court
{clean_text_minimal(judicial_decision)}
"""
    
    return formatted_output

def format_all_rounds_raw(plaintiff_args: List[str], defendant_args: List[str], consensus_metrics: List[Dict]) -> str:
    """Format all rounds of arguments preserving original content"""
    
    if not plaintiff_args or not defendant_args:
        return "No round-by-round arguments available"
    
    rounds_output = []
    
    for i, (p_arg, d_arg) in enumerate(zip(plaintiff_args, defendant_args), 1):
        # Get consensus info for this round
        consensus_info = ""
        if i <= len(consensus_metrics):
            metrics = consensus_metrics[i-1]
            score = metrics.get('consensus_score', 0.0)
            convergence = metrics.get('convergence', 0.0)
            similarity = metrics.get('similarity', 0.0)
            
            status_emoji = "✅" if score >= 0.7 else "📈" if score >= 0.5 else "📊"
            consensus_info = f"{status_emoji} Consensus: {score:.3f} | Convergence: {convergence:.3f} | Similarity: {similarity:.3f}"
        
        round_header = f"""
{'='*80}
🏛️ ROUND {i} ARGUMENTS
{consensus_info}
{'='*80}"""
        
        plaintiff_section = f"""
⚖️ PLAINTIFF'S ROUND {i} ARGUMENTS
Advocate for the Employees
{clean_text_minimal(p_arg)}
"""
        
        defendant_section = f"""
🛡️ DEFENDANT'S ROUND {i} ARGUMENTS
Advocate for the Company
{clean_text_minimal(d_arg)}
"""
        
        rounds_output.append(round_header)
        rounds_output.append(plaintiff_section)
        rounds_output.append(defendant_section)
    
    return "\n".join(rounds_output)

def format_consensus_progression(consensus_metrics: List[Dict]) -> str:
    """Format consensus progression across rounds"""
    if not consensus_metrics:
        return ""
    
    progression = ["📊 CONSENSUS PROGRESSION ACROSS ROUNDS"]
    progression.append("-" * 60)
    
    for i, metrics in enumerate(consensus_metrics, 1):
        score = metrics.get('consensus_score', 0.0)
        convergence = metrics.get('convergence', 0.0)
        similarity = metrics.get('similarity', 0.0)
        
        status_emoji = "✅" if score >= 0.7 else "📈" if score >= 0.5 else "📊"
        
        progression.append(
            f"Round {i}: {status_emoji} Consensus: {score:.3f} | "
            f"Convergence: {convergence:.3f} | Similarity: {similarity:.3f}"
        )
    
    return "\n".join(progression) + "\n"

def clean_text_minimal(text: str) -> str:
    """
    Minimal text cleaning - preserves all original content
    Only removes extreme formatting artifacts, keeps everything else
    """
    if not text:
        return "No content available"
    
    # Only remove markdown symbols, keep all text content
    text = re.sub(r'[*#>`_]+', '', text)
    
    # Remove only excessive whitespace (more than 3 spaces)
    text = re.sub(r'   +', '  ', text)
    
    # Clean up only obvious formatting artifacts
    text = re.sub(r'JUDGMENT:|BY THE COURT:', '', text)
    
    # Keep all paragraph breaks and content structure
    text = text.strip()
    
    return text

# Alternative function for completely RAW output (no cleaning at all)
def format_debate_completely_raw(result: Dict[str, Any]) -> str:
    """
    Completely raw output - zero processing of judicial decision
    """
    if "multi_agent_debate" not in result:
        return "No multi-agent debate data found"
    
    debate = result["multi_agent_debate"]
    
    # Extract with zero processing
    rounds = debate.get("debate_rounds", 1)
    consensus_summary = debate.get("consensus_summary", {})
    judicial_decision_raw = debate.get("judicial_decision", "")
    
    return f"""🔄 LEGAL DEBATE RESULTS - COMPLETELY RAW OUTPUT
Total Rounds: {rounds}
Final Consensus: {consensus_summary.get('final_consensus_score', 'N/A')}
Status: {'✅ REACHED' if consensus_summary.get('consensus_reached') else '❌ MAX ROUNDS'}

👩‍⚖️ COMPLETE JUDICIAL DECISION (UNPROCESSED)
Judge, High Court

{judicial_decision_raw}

END OF JUDICIAL DECISION
========================================
"""