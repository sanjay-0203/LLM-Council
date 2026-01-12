"""
Response evaluation system for LLM Council.
"""
import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from statistics import mean, stdev
from loguru import logger
from .prompts import format_evaluation_prompt
@dataclass
class EvaluationResult:
    """Result from a single evaluator."""
    evaluator: str
    evaluator_role: str
    ratings: Dict[str, float]
    overall_score: float
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    reasoning: str = ""
    weight: float = 1.0
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "role": self.evaluator_role,
            "ratings": self.ratings,
            "overall_score": round(self.overall_score, 2),
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "reasoning": self.reasoning[:200] + "..." if len(self.reasoning) > 200 else self.reasoning
        }
@dataclass
class AggregatedEvaluation:
    """Aggregated evaluation from multiple evaluators."""
    individual_evaluations: List[EvaluationResult]
    aggregated_ratings: Dict[str, float]
    overall_score: float
    score_std: float
    confidence: float
    common_strengths: List[str]
    common_weaknesses: List[str]
    all_suggestions: List[str]
    consensus_level: float
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "score_std": round(self.score_std, 2),
            "confidence": round(self.confidence, 2),
            "consensus_level": round(self.consensus_level, 2),
            "aggregated_ratings": {k: round(v, 2) for k, v in self.aggregated_ratings.items()},
            "common_strengths": self.common_strengths,
            "common_weaknesses": self.common_weaknesses,
            "suggestions": self.all_suggestions,
            "num_evaluators": len(self.individual_evaluations)
        }
@dataclass
class EvaluationCriterion:
    """A single evaluation criterion."""
    name: str
    description: str
    weight: float = 1.0
    min_score: int = 1
    max_score: int = 10
class ResponseEvaluator:
    """
    Evaluates responses using council members.
    """
    DEFAULT_CRITERIA = [
        EvaluationCriterion("accuracy", "Factual correctness and truthfulness", 1.0),
        EvaluationCriterion("relevance", "How well the response addresses the question", 0.95),
        EvaluationCriterion("clarity", "Clear, understandable explanation", 0.85),
        EvaluationCriterion("completeness", "Comprehensive coverage of the topic", 0.8),
        EvaluationCriterion("reasoning", "Logical reasoning and argumentation quality", 0.9),
    ]
    def __init__(
        self,
        criteria: Optional[List[EvaluationCriterion]] = None
    ):
        self.criteria = criteria or self.DEFAULT_CRITERIA
        self._criteria_weights = {c.name: c.weight for c in self.criteria}
    def format_criteria_text(self) -> str:
        """Format criteria for prompt."""
        lines = []
        for c in self.criteria:
            lines.append(f"- **{c.name}** ({c.min_score}-{c.max_score}): {c.description}")
        return "\n".join(lines)
    def parse_evaluation_response(
        self,
        response: str,
        evaluator: str,
        evaluator_role: str,
        weight: float = 1.0
    ) -> Optional[EvaluationResult]:
        """Parse evaluation from LLM response."""
        try:
            json_str = response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            if "{" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]
            data = json.loads(json_str)
            ratings = data.get("ratings", {})
            normalized_ratings = {}
            for key, value in ratings.items():
                key_lower = key.lower().replace(" ", "_")
                try:
                    normalized_ratings[key_lower] = float(value)
                except (ValueError, TypeError):
                    normalized_ratings[key_lower] = 5.0  
            overall_score = data.get("overall_score")
            if overall_score is None:
                if normalized_ratings:
                    weighted_sum = sum(
                        normalized_ratings.get(c.name, 5) * c.weight
                        for c in self.criteria
                    )
                    total_weight = sum(c.weight for c in self.criteria)
                    overall_score = weighted_sum / total_weight
                else:
                    overall_score = 5.0
            return EvaluationResult(
                evaluator=evaluator,
                evaluator_role=evaluator_role,
                ratings=normalized_ratings,
                overall_score=float(overall_score),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", []),
                reasoning=data.get("reasoning", ""),
                weight=weight
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse evaluation from {evaluator}: {e}")
            return None
    def aggregate_evaluations(
        self,
        evaluations: List[EvaluationResult]
    ) -> AggregatedEvaluation:
        """Aggregate multiple evaluations."""
        if not evaluations:
            return AggregatedEvaluation(
                individual_evaluations=[],
                aggregated_ratings={},
                overall_score=0.0,
                score_std=0.0,
                confidence=0.0,
                common_strengths=[],
                common_weaknesses=[],
                all_suggestions=[],
                consensus_level=0.0
            )
        aggregated_ratings: Dict[str, List[tuple]] = {}
        for eval_result in evaluations:
            for criterion, score in eval_result.ratings.items():
                if criterion not in aggregated_ratings:
                    aggregated_ratings[criterion] = []
                aggregated_ratings[criterion].append((score, eval_result.weight))
        final_ratings = {}
        for criterion, scores in aggregated_ratings.items():
            weighted_sum = sum(score * weight for score, weight in scores)
            total_weight = sum(weight for _, weight in scores)
            final_ratings[criterion] = weighted_sum / total_weight if total_weight > 0 else 0
        overall_scores = [e.overall_score for e in evaluations]
        overall_score = mean(overall_scores)
        score_std = stdev(overall_scores) if len(overall_scores) > 1 else 0.0
        max_possible_std = 4.5  
        consensus_level = max(0, 1 - (score_std / max_possible_std))
        base_confidence = consensus_level
        evaluator_bonus = min(len(evaluations) / 5, 1.0) * 0.2
        confidence = min(base_confidence + evaluator_bonus, 1.0)
        strength_counts: Dict[str, int] = {}
        weakness_counts: Dict[str, int] = {}
        all_suggestions = []
        for eval_result in evaluations:
            for s in eval_result.strengths:
                s_lower = s.lower()
                strength_counts[s_lower] = strength_counts.get(s_lower, 0) + 1
            for w in eval_result.weaknesses:
                w_lower = w.lower()
                weakness_counts[w_lower] = weakness_counts.get(w_lower, 0) + 1
            all_suggestions.extend(eval_result.suggestions)
        min_mentions = min(2, len(evaluations))
        common_strengths = [s for s, c in strength_counts.items() if c >= min_mentions]
        common_weaknesses = [w for w, c in weakness_counts.items() if c >= min_mentions]
        seen_suggestions = set()
        unique_suggestions = []
        for s in all_suggestions:
            s_lower = s.lower()
            if s_lower not in seen_suggestions:
                seen_suggestions.add(s_lower)
                unique_suggestions.append(s)
        return AggregatedEvaluation(
            individual_evaluations=evaluations,
            aggregated_ratings=final_ratings,
            overall_score=overall_score,
            score_std=score_std,
            confidence=confidence,
            common_strengths=common_strengths[:5],
            common_weaknesses=common_weaknesses[:5],
            all_suggestions=unique_suggestions[:10],
            consensus_level=consensus_level
        )
    async def evaluate(
        self,
        question: str,
        response: str,
        evaluators: List[Any],  
    ) -> AggregatedEvaluation:
        """
        Evaluate a response using multiple council members.
        """
        criteria_text = self.format_criteria_text()
        prompt = format_evaluation_prompt(question, response, criteria_text)
        async def get_evaluation(evaluator) -> Optional[EvaluationResult]:
            try:
                result = await evaluator.generate_async(
                    prompt,
                    format="json"
                )
                if result.success:
                    return self.parse_evaluation_response(
                        result.content,
                        evaluator=evaluator.name,
                        evaluator_role=evaluator.role,
                        weight=evaluator.weight
                    )
            except Exception as e:
                logger.error(f"Error getting evaluation from {evaluator.name}: {e}")
            return None
        tasks = [get_evaluation(e) for e in evaluators]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_evaluations = [r for r in results if isinstance(r, EvaluationResult)]
        return self.aggregate_evaluations(valid_evaluations)
    async def evaluate_multiple(
        self,
        question: str,
        responses: List[str],
        evaluators: List[Any]
    ) -> List[AggregatedEvaluation]:
        """Evaluate multiple responses."""
        tasks = [self.evaluate(question, resp, evaluators) for resp in responses]
        return await asyncio.gather(*tasks)
    async def rank_responses(
        self,
        question: str,
        responses: List[str],
        evaluators: List[Any]
    ) -> List[tuple]:
        """Evaluate and rank responses."""
        evaluations = await self.evaluate_multiple(question, responses, evaluators)
        ranked = [
            (i, responses[i], eval_result, eval_result.overall_score)
            for i, eval_result in enumerate(evaluations)
        ]
        ranked.sort(key=lambda x: x[3], reverse=True)
        return ranked
