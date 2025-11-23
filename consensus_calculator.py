"""
Consensus Calculator for Legal Debates
Calculates consensus scores based on argument similarity and convergence
"""

import re
from typing import List, Dict, Any
from difflib import SequenceMatcher

class ConsensusCalculator:
    """Calculate consensus scores for legal debates"""
    
    def __init__(self, target_consensus: float = 0.7):
        self.target_consensus = target_consensus
        self.convergence_keywords = [
            'agree', 'accept', 'concede', 'acknowledge', 'valid point',
            'correct', 'right', 'appropriate', 'reasonable', 'fair'
        ]
        self.divergence_keywords = [
            'disagree', 'reject', 'deny', 'dispute', 'contest',
            'wrong', 'incorrect', 'unreasonable', 'unfair', 'invalid'
        ]
    
    def calculate_round_consensus(
        self, 
        plaintiff_args: List[str], 
        defendant_args: List[str],
        round_number: int
    ) -> Dict[str, float]:
        """
        Calculate consensus metrics for current debate round
        
        Returns:
            Dict with consensus scores and metrics
        """
        
        if len(plaintiff_args) != len(defendant_args):
            return {'consensus_score': 0.0, 'convergence': 0.0, 'similarity': 0.0}
        
        # Get latest arguments
        latest_plaintiff = plaintiff_args[-1] if plaintiff_args else ""
        latest_defendant = defendant_args[-1] if defendant_args else ""
        
        # Calculate different consensus metrics
        similarity_score = self._calculate_similarity(latest_plaintiff, latest_defendant)
        convergence_score = self._calculate_convergence(latest_plaintiff, latest_defendant)
        position_stability = self._calculate_position_stability(plaintiff_args, defendant_args)
        
        # Weighted consensus score
        consensus_score = (
            similarity_score * 0.4 +
            convergence_score * 0.4 +
            position_stability * 0.2
        )
        
        return {
            'consensus_score': round(consensus_score, 3),
            'similarity': round(similarity_score, 3),
            'convergence': round(convergence_score, 3),
            'position_stability': round(position_stability, 3),
            'round_number': round_number,
            'target_reached': consensus_score >= self.target_consensus
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate textual similarity between arguments"""
        if not text1 or not text2:
            return 0.0
        
        # Clean and normalize text
        clean1 = self._clean_text(text1)
        clean2 = self._clean_text(text2)
        
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, clean1, clean2).ratio()
        return min(similarity, 1.0)
    
    def _calculate_convergence(self, plaintiff_text: str, defendant_text: str) -> float:
        """Calculate convergence based on agreement/disagreement keywords"""
        
        # Count convergence indicators in both texts
        plaintiff_convergence = self._count_keywords(plaintiff_text, self.convergence_keywords)
        plaintiff_divergence = self._count_keywords(plaintiff_text, self.divergence_keywords)
        
        defendant_convergence = self._count_keywords(defendant_text, self.convergence_keywords)
        defendant_divergence = self._count_keywords(defendant_text, self.divergence_keywords)
        
        # Calculate convergence ratio
        total_convergence = plaintiff_convergence + defendant_convergence
        total_divergence = plaintiff_divergence + defendant_divergence
        total_indicators = total_convergence + total_divergence
        
        if total_indicators == 0:
            return 0.5  # Neutral if no clear indicators
        
        convergence_ratio = total_convergence / total_indicators
        return min(convergence_ratio, 1.0)
    
    def _calculate_position_stability(self, plaintiff_args: List[str], defendant_args: List[str]) -> float:
        """Calculate how stable/consistent positions are across rounds"""
        
        if len(plaintiff_args) < 2 or len(defendant_args) < 2:
            return 0.5  # Can't measure stability with less than 2 rounds
        
        # Calculate consistency in plaintiff arguments
        plaintiff_consistency = 0.0
        for i in range(1, len(plaintiff_args)):
            similarity = SequenceMatcher(None, plaintiff_args[i-1], plaintiff_args[i]).ratio()
            plaintiff_consistency += similarity
        plaintiff_consistency /= (len(plaintiff_args) - 1)
        
        # Calculate consistency in defendant arguments
        defendant_consistency = 0.0
        for i in range(1, len(defendant_args)):
            similarity = SequenceMatcher(None, defendant_args[i-1], defendant_args[i]).ratio()
            defendant_consistency += similarity
        defendant_consistency /= (len(defendant_args) - 1)
        
        # Average consistency indicates position stability
        return (plaintiff_consistency + defendant_consistency) / 2
    
    def _clean_text(self, text: str) -> str:
        """Clean text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove legal citations (basic pattern)
        text = re.sub(r'\b\d+\s+\w+\.?\s+\d+\b', '', text)
        
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        """Count occurrence of keywords in text"""
        text_lower = text.lower()
        count = 0
        for keyword in keywords:
            count += text_lower.count(keyword.lower())
        return count
    
    def should_continue_debate(self, consensus_metrics: Dict[str, float]) -> bool:
        """Determine if debate should continue based on consensus metrics"""
        return not consensus_metrics.get('target_reached', False)
    
    def get_consensus_summary(self, all_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Generate summary of consensus progression"""
        
        if not all_metrics:
            return {'status': 'no_data'}
        
        final_metrics = all_metrics[-1]
        
        return {
            'final_consensus_score': final_metrics['consensus_score'],
            'target_consensus': self.target_consensus,
            'consensus_reached': final_metrics['target_reached'],
            'total_rounds': len(all_metrics),
            'consensus_progression': [m['consensus_score'] for m in all_metrics],
            'convergence_trend': self._calculate_trend([m['convergence'] for m in all_metrics])
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        increasing = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        decreasing = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        
        if increasing > decreasing:
            return 'increasing'
        elif decreasing > increasing:
            return 'decreasing'
        else:
            return 'stable'