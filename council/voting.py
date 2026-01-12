"""
Voting system for LLM Council.
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from collections import Counter, defaultdict

from loguru import logger

from .prompts import format_vote_prompt


class VotingMethod(Enum):
    """Available voting methods."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    RANKED = "ranked"
    BORDA = "borda"
    APPROVAL = "approval"
    CONDORCET = "condorcet"


@dataclass
class Vote:
    """A single vote from a council member."""
    voter: str
    voter_role: str
    selected: int  # 1-indexed candidate number
    confidence: float
    ranking: List[int] = field(default_factory=list)
    reasoning: str = ""
    weight: float = 1.0
    pros_cons: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure ranking includes selected if not present
        if self.ranking and self.selected not in self.ranking:
            self.ranking.insert(0, self.selected)


@dataclass
class VotingResult:
    """Result of a voting session."""
    winner: int  # 1-indexed
    winner_content: str
    method: VotingMethod
    votes: List[Vote]
    vote_counts: Dict[int, float]
    confidence: float
    consensus_level: float
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_votes(self) -> int:
        return len(self.votes)
    
    @property
    def winner_vote_share(self) -> float:
        total = sum(self.vote_counts.values())
        if total == 0:
            return 0.0
        return self.vote_counts.get(self.winner, 0) / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "winner_content": self.winner_content[:200] + "..." if len(self.winner_content) > 200 else self.winner_content,
            "method": self.method.value,
            "total_votes": self.total_votes,
            "vote_counts": self.vote_counts,
            "confidence": round(self.confidence, 3),
            "consensus_level": round(self.consensus_level, 3),
            "winner_vote_share": round(self.winner_vote_share, 3),
            "detailed_results": self.detailed_results
        }


class VotingSystem:
    """
    Implements various voting methods for the council.
    """
    
    def __init__(
        self,
        default_method: VotingMethod = VotingMethod.WEIGHTED,
        min_agreement_threshold: float = 0.6
    ):
        self.default_method = default_method
        self.min_agreement_threshold = min_agreement_threshold
        self._votes: List[Vote] = []
    
    def reset(self):
        """Reset votes for a new voting session."""
        self._votes = []
    
    def add_vote(self, vote: Vote):
        """Add a vote to the current session."""
        self._votes.append(vote)
    
    def parse_vote_response(
        self,
        response: str,
        voter: str,
        voter_role: str,
        weight: float = 1.0
    ) -> Optional[Vote]:
        """Parse a vote from an LLM response."""
        try:
            # Try to extract JSON from response
            json_str = response
            
            # Handle markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            
            # Try to find JSON object
            if "{" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]
            
            data = json.loads(json_str)
            
            vote = Vote(
                voter=voter,
                voter_role=voter_role,
                selected=int(data.get("selected", 1)),
                confidence=float(data.get("confidence", 0.5)),
                ranking=data.get("ranking", []),
                reasoning=data.get("reasoning", ""),
                weight=weight,
                pros_cons=data.get("pros_cons", {})
            )
            
            return vote
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse vote from {voter}: {e}")
            
            # Try simple extraction
            try:
                # Look for "selected": N pattern
                import re
                match = re.search(r'"selected"\s*:\s*(\d+)', response)
                if match:
                    return Vote(
                        voter=voter,
                        voter_role=voter_role,
                        selected=int(match.group(1)),
                        confidence=0.5,
                        weight=weight
                    )
            except:
                pass
        
        return None
    
    def tally(
        self,
        candidates: List[str],
        method: Optional[VotingMethod] = None,
        votes: Optional[List[Vote]] = None
    ) -> VotingResult:
        """
        Tally votes using the specified method.
        """
        method = method or self.default_method
        votes = votes or self._votes
        
        if not votes:
            return VotingResult(
                winner=1,
                winner_content=candidates[0] if candidates else "",
                method=method,
                votes=[],
                vote_counts={},
                confidence=0.0,
                consensus_level=0.0
            )
        
        # Dispatch to appropriate method
        if method == VotingMethod.MAJORITY:
            return self._majority_vote(candidates, votes)
        elif method == VotingMethod.WEIGHTED:
            return self._weighted_vote(candidates, votes)
        elif method == VotingMethod.UNANIMOUS:
            return self._unanimous_vote(candidates, votes)
        elif method == VotingMethod.RANKED:
            return self._ranked_vote(candidates, votes)
        elif method == VotingMethod.BORDA:
            return self._borda_count(candidates, votes)
        elif method == VotingMethod.APPROVAL:
            return self._approval_vote(candidates, votes)
        elif method == VotingMethod.CONDORCET:
            return self._condorcet_vote(candidates, votes)
        else:
            return self._weighted_vote(candidates, votes)
    
    def _majority_vote(
        self,
        candidates: List[str],
        votes: List[Vote]
    ) -> VotingResult:
        """Simple majority voting (one vote per member)."""
        vote_counts = Counter(v.selected for v in votes)
        
        if not vote_counts:
            winner = 1
        else:
            winner = vote_counts.most_common(1)[0][0]
        
        # Calculate consensus level
        total_votes = len(votes)
        winner_votes = vote_counts.get(winner, 0)
        consensus_level = winner_votes / total_votes if total_votes > 0 else 0
        
        # Average confidence for winner
        winner_confidences = [v.confidence for v in votes if v.selected == winner]
        avg_confidence = sum(winner_confidences) / len(winner_confidences) if winner_confidences else 0
        
        return VotingResult(
            winner=winner,
            winner_content=candidates[winner - 1] if 0 < winner <= len(candidates) else "",
            method=VotingMethod.MAJORITY,
            votes=votes,
            vote_counts=dict(vote_counts),
            confidence=avg_confidence,
            consensus_level=consensus_level,
            detailed_results={
                "vote_distribution": dict(vote_counts),
                "total_voters": total_votes
            }
        )
    
    def _weighted_vote(
        self,
        candidates: List[str],
        votes: List[Vote]
    ) -> VotingResult:
        """Weighted voting based on member weights and confidence."""
        weighted_counts: Dict[int, float] = defaultdict(float)
        
        for vote in votes:
            # Weight = member_weight * confidence
            effective_weight = vote.weight * vote.confidence
            weighted_counts[vote.selected] += effective_weight
        
        if not weighted_counts:
            winner = 1
        else:
            winner = max(weighted_counts, key=weighted_counts.get)
        
        # Calculate consensus level
        total_weight = sum(weighted_counts.values())
        winner_weight = weighted_counts.get(winner, 0)
        consensus_level = winner_weight / total_weight if total_weight > 0 else 0
        
        # Normalize confidence
        confidence = min(consensus_level * 1.2, 1.0)  # Slight boost for strong consensus
        
        return VotingResult(
            winner=winner,
            winner_content=candidates[winner - 1] if 0 < winner <= len(candidates) else "",
            method=VotingMethod.WEIGHTED,
            votes=votes,
            vote_counts=dict(weighted_counts),
            confidence=confidence,
            consensus_level=consensus_level,
            detailed_results={
                "weighted_scores": {k: round(v, 3) for k, v in weighted_counts.items()},
                "total_weight": round(total_weight, 3)
            }
        )
    
    def _unanimous_vote(
        self,
        candidates: List[str],
        votes: List[Vote]
    ) -> VotingResult:
        """Requires unanimous agreement."""
        if not votes:
            return VotingResult(
                winner=1,
                winner_content=candidates[0] if candidates else "",
                method=VotingMethod.UNANIMOUS,
                votes=[],
                vote_counts={},
                confidence=0.0,
                consensus_level=0.0
            )
        
        vote_counts = Counter(v.selected for v in votes)
        
        # Check for unanimity
        if len(vote_counts) == 1:
            winner = list(vote_counts.keys())[0]
            consensus_level = 1.0
            confidence = sum(v.confidence for v in votes) / len(votes)
        else:
            # No unanimity - fall back to weighted
            result = self._weighted_vote(candidates, votes)
            result.detailed_results["unanimous"] = False
            result.method = VotingMethod.UNANIMOUS
            return result
        
        return VotingResult(
            winner=winner,
            winner_content=candidates[winner - 1] if 0 < winner <= len(candidates) else "",
            method=VotingMethod.UNANIMOUS,
            votes=votes,
            vote_counts=dict(vote_counts),
            confidence=confidence,
            consensus_level=consensus_level,
            detailed_results={"unanimous": True}
        )
    
    def _ranked_vote(
        self,
        candidates: List[str],
        votes: List[Vote]
    ) -> VotingResult:
        """Instant-runoff voting using rankings."""
        if not votes:
            return self._majority_vote(candidates, votes)
        
        # Get rankings from votes (fall back to [selected] if no ranking)
        rankings = []
        weights = []
        
        for v in votes:
            if v.ranking:
                rankings.append(v.ranking.copy())
            else:
                rankings.append([v.selected])
            weights.append(v.weight)
        
        num_candidates = len(candidates)
        eliminated = set()
        round_results = []
        
        while True:
            # Count first-place votes
            first_place_counts: Dict[int, float] = defaultdict(float)
            
            for ranking, weight in zip(rankings, weights):
                # Find first non-eliminated candidate
                for candidate in ranking:
                    if candidate not in eliminated and 1 <= candidate <= num_candidates:
                        first_place_counts[candidate] += weight
                        break
            
            if not first_place_counts:
                # All eliminated
                winner = 1
                break
            
            round_results.append(dict(first_place_counts))
            
            # Check for majority
            total = sum(first_place_counts.values())
            for candidate, count in first_place_counts.items():
                if count > total / 2:
                    winner = candidate
                    break
            else:
                # Eliminate lowest
                lowest = min(first_place_counts, key=first_place_counts.get)
                eliminated.add(lowest)
                continue
            
            break
        
        vote_counts = round_results[-1] if round_results else {}
        total = sum(vote_counts.values())
        winner_count = vote_counts.get(winner, 0)
        
        return VotingResult(
            winner=winner,
            winner_content=candidates[winner - 1] if 0 < winner <= len(candidates) else "",
            method=VotingMethod.RANKED,
            votes=votes,
            vote_counts=vote_counts,
            confidence=winner_count / total if total > 0 else 0,
            consensus_level=winner_count / total if total > 0 else 0,
            detailed_results={
                "rounds": round_results,
                "eliminated_order": list(eliminated)
            }
        )
    
    def _borda_count(
        self,
        candidates: List[str],
        votes: List[Vote]
    ) -> VotingResult:
        """Borda count method - points based on ranking position."""
        if not votes:
            return self._majority_vote(candidates, votes)
        
        num_candidates = len(candidates)
        scores: Dict[int, float] = defaultdict(float)
        
        for vote in votes:
            ranking = vote.ranking if vote.ranking else [vote.selected]
            for position, candidate in enumerate(ranking):
                if 1 <= candidate <= num_candidates:
                    # Points = num_candidates - position
                    points = (num_candidates - position) * vote.weight
                    scores[candidate] += points
        
        if not scores:
            winner = 1
        else:
            winner = max(scores, key=scores.get)
        
        total_score = sum(scores.values())
        winner_score = scores.get(winner, 0)
        
        return VotingResult(
            winner=winner,
            winner_content=candidates[winner - 1] if 0 < winner <= len(candidates) else "",
            method=VotingMethod.BORDA,
            votes=votes,
            vote_counts=dict(scores),
            confidence=winner_score / total_score if total_score > 0 else 0,
            consensus_level=winner_score / (num_candidates * len(votes)) if votes else 0,
            detailed_results={
                "borda_scores": {k: round(v, 2) for k, v in scores.items()},
                "max_possible_score": num_candidates * len(votes)
            }
        )
    
    def _approval_vote(
        self,
        candidates: List[str],
        votes: List[Vote]
    ) -> VotingResult:
        """Approval voting - approve all acceptable candidates."""
        # Use ranking as approval list (all ranked = approved)
        approval_counts: Dict[int, float] = defaultdict(float)
        
        for vote in votes:
            approved = vote.ranking if vote.ranking else [vote.selected]
            for candidate in approved:
                approval_counts[candidate] += vote.weight
        
        if not approval_counts:
            winner = 1
        else:
            winner = max(approval_counts, key=approval_counts.get)
        
        total_approvals = sum(approval_counts.values())
        
        return VotingResult(
            winner=winner,
            winner_content=candidates[winner - 1] if 0 < winner <= len(candidates) else "",
            method=VotingMethod.APPROVAL,
            votes=votes,
            vote_counts=dict(approval_counts),
            confidence=approval_counts.get(winner, 0) / len(votes) if votes else 0,
            consensus_level=approval_counts.get(winner, 0) / len(votes) if votes else 0,
            detailed_results={
                "approval_counts": dict(approval_counts),
                "approval_rates": {
                    k: round(v / len(votes), 3)
                    for k, v in approval_counts.items()
                } if votes else {}
            }
        )
    
    def _condorcet_vote(
        self,
        candidates: List[str],
        votes: List[Vote]
    ) -> VotingResult:
        """Condorcet method - pairwise comparison winner."""
        if not votes:
            return self._majority_vote(candidates, votes)
        
        num_candidates = len(candidates)
        
        # Build pairwise comparison matrix
        pairwise: Dict[Tuple[int, int], float] = defaultdict(float)
        
        for vote in votes:
            ranking = vote.ranking if vote.ranking else [vote.selected]
            # For each pair, record who is ranked higher
            for i, higher in enumerate(ranking):
                for lower in ranking[i + 1:]:
                    pairwise[(higher, lower)] += vote.weight
        
        # Find Condorcet winner (beats all others in pairwise)
        wins: Dict[int, int] = defaultdict(int)
        
        for i in range(1, num_candidates + 1):
            for j in range(1, num_candidates + 1):
                if i != j:
                    if pairwise[(i, j)] > pairwise[(j, i)]:
                        wins[i] += 1
        
        # Condorcet winner beats all others
        condorcet_winner = None
        for candidate, win_count in wins.items():
            if win_count == num_candidates - 1:
                condorcet_winner = candidate
                break
        
        if condorcet_winner:
            winner = condorcet_winner
            confidence = 1.0
        else:
            # No Condorcet winner - use Borda as fallback
            borda_result = self._borda_count(candidates, votes)
            winner = borda_result.winner
            confidence = 0.7 * borda_result.confidence
        
        return VotingResult(
            winner=winner,
            winner_content=candidates[winner - 1] if 0 < winner <= len(candidates) else "",
            method=VotingMethod.CONDORCET,
            votes=votes,
            vote_counts=dict(wins),
            confidence=confidence,
            consensus_level=wins.get(winner, 0) / (num_candidates - 1) if num_candidates > 1 else 1,
            detailed_results={
                "pairwise_wins": dict(wins),
                "condorcet_winner_exists": condorcet_winner is not None,
                "pairwise_matrix": {
                    f"{i}v{j}": pairwise[(i, j)]
                    for i in range(1, num_candidates + 1)
                    for j in range(1, num_candidates + 1)
                    if i < j
                }
            }
        )
    
    async def collect_votes(
        self,
        question: str,
        candidates: List[str],
        voters: List[Any],  # List of CouncilMember
        method: Optional[VotingMethod] = None
    ) -> VotingResult:
        """
        Collect votes from council members and tally results.
        """
        self.reset()
        
        # Format candidates for prompt
        candidates_text = "\n\n".join([
            f"**Candidate {i + 1}:**\n{content}"
            for i, content in enumerate(candidates)
        ])
        
        vote_prompt = format_vote_prompt(question, candidates_text)
        
        # Collect votes in parallel
        async def get_vote(voter) -> Optional[Vote]:
            try:
                result = await voter.generate_async(
                    vote_prompt,
                    format="json"
                )
                
                if result.success:
                    return self.parse_vote_response(
                        result.content,
                        voter=voter.name,
                        voter_role=voter.role,
                        weight=voter.weight
                    )
            except Exception as e:
                logger.error(f"Error getting vote from {voter.name}: {e}")
            
            return None
        
        tasks = [get_vote(v) for v in voters]
        vote_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Add valid votes
        for result in vote_results:
            if isinstance(result, Vote):
                self.add_vote(result)
        
        # Tally and return
        return self.tally(candidates, method=method or self.default_method)
