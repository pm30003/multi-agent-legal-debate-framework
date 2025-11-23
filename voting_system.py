#!/usr/bin/env python3
"""
Voting-based Decision Module for Legal Debate Framework
Allows comparison between Consensus-based and Voting-based decisions
"""

import asyncio
import statistics
from typing import Dict, Any, List
import random

class VotingSystem:
    def __init__(self, llms: Dict[str, Any], model_count: int = 3):
        """
        Initialize the voting system.
        :param llms: A dictionary of loaded Groq LLMs.
        :param model_count: Number of virtual 'judges' to run for voting.
        """
        self.llms = llms
        self.model_count = min(model_count, len(llms)) if llms else 3
        self.models_used = list(llms.values())[:self.model_count] if llms else None

    async def simulate_vote(
        self,
        case_summary: str,
        legal_question: str,
        plaintiff_summary: str,
        defendant_summary: str,
        consensus_score: float,
    ) -> Dict[str, Any]:
        """
        Runs multiple judge agents to vote for Plaintiff or Defendant.
        Each agent acts independently based on debate summaries.
        """
        print("🗳 Running multi-agent voting for judicial decision...")

        if not self.models_used:
            print("⚠️ No Groq models available for voting system. Using simulated judges.")
            return self._simulate_random_votes()

        tasks = []
        for i, model in enumerate(self.models_used, 1):
            prompt = f"""
You are Judge #{i} deciding a civil dispute case.

CASE SUMMARY:
{case_summary}

LEGAL QUESTION:
{legal_question}

PLAINTIFF POSITION:
{plaintiff_summary[:600]}

DEFENDANT POSITION:
{defendant_summary[:600]}

CONSENSUS METRIC: {consensus_score:.3f}

Decide which side has a stronger legal foundation.

Respond strictly with:
PLAINTIFF - if the plaintiff should win
DEFENDANT - if the defendant should win
DRAW - if equally strong
"""
            tasks.append(self._fetch_vote(model, prompt, i))

        results = await asyncio.gather(*tasks)
        return self._analyze_votes(results)

    async def _fetch_vote(self, model, prompt: str, judge_id: int) -> str:
        """Single judge’s vote using the provided model."""
        try:
            response = await model.ainvoke([
                {"role": "system", "content": "You are an impartial High Court judge."},
                {"role": "user", "content": prompt},
            ])
            content = response.content.strip().upper()
            if "PLAINTIFF" in content:
                return "PLAINTIFF"
            elif "DEFENDANT" in content:
                return "DEFENDANT"
            else:
                return "DRAW"
        except Exception as e:
            print(f"⚠️ Judge {judge_id} failed to respond: {e}")
            return random.choice(["PLAINTIFF", "DEFENDANT", "DRAW"])

    def _analyze_votes(self, votes: List[str]) -> Dict[str, Any]:
        """Aggregate all judge votes and determine winner."""
        total_votes = len(votes)
        counts = {v: votes.count(v) for v in ["PLAINTIFF", "DEFENDANT", "DRAW"]}

        winner = max(counts, key=counts.get)
        if counts["PLAINTIFF"] == counts["DEFENDANT"]:  # Tie-breaker
            winner = "DRAW"

        confidence = max(counts.values()) / total_votes

        print(f"🗳 Vote Results: {counts}")
        print(f"🏛 Final Vote-based Decision: {winner} (confidence {confidence:.2f})")

        return {
            "winner": winner,
            "vote_distribution": counts,
            "confidence": round(confidence, 3),
            "total_votes": total_votes,
        }

    def _simulate_random_votes(self):
        """Fallback pseudo-random voting if Groq unavailable."""
        outcomes = random.choices(["PLAINTIFF", "DEFENDANT", "DRAW"], weights=[0.45, 0.45, 0.1], k=3)
        return self._analyze_votes(outcomes)
