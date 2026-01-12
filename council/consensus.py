"""
Consensus building for LLM Council.
"""
import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from loguru import logger
from .prompts import format_consensus_prompt, format_improve_prompt, format_debate_prompt
@dataclass
class ConsensusResult:
    """Result of consensus building."""
    consensus: str
    confidence: float
    key_points: List[str]
    areas_of_agreement: List[str]
    areas_of_disagreement: List[str]
    integrated_insights: List[str]
    final_recommendation: str
    synthesis_model: str
    original_responses: List[Dict[str, str]] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus": self.consensus[:500] + "..." if len(self.consensus) > 500 else self.consensus,
            "confidence": round(self.confidence, 3),
            "key_points": self.key_points,
            "areas_of_agreement": self.areas_of_agreement,
            "areas_of_disagreement": self.areas_of_disagreement,
            "integrated_insights": self.integrated_insights,
            "final_recommendation": self.final_recommendation,
            "synthesis_model": self.synthesis_model,
            "num_original_responses": len(self.original_responses)
        }
class ConsensusBuilder:
    """
    Builds consensus from multiple council responses.
    """
    def __init__(
        self,
        min_responses: int = 2,
        require_agreement_threshold: float = 0.6
    ):
        self.min_responses = min_responses
        self.require_agreement_threshold = require_agreement_threshold
    def format_responses_for_prompt(
        self,
        responses: List[Dict[str, Any]]
    ) -> str:
        """Format responses for the consensus prompt."""
        formatted = []
        for i, resp in enumerate(responses, 1):
            model = resp.get("model", f"Model {i}")
            role = resp.get("role", "unknown")
            content = resp.get("content", "")
            formatted.append(
                f"**Response from {model} ({role}):**\n{content}"
            )
        return "\n\n---\n\n".join(formatted)
    def parse_consensus_response(
        self,
        response: str,
        synthesis_model: str
    ) -> Optional[ConsensusResult]:
        """Parse consensus from LLM response."""
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
            return ConsensusResult(
                consensus=data.get("consensus", ""),
                confidence=float(data.get("confidence", 0.5)),
                key_points=data.get("key_points", []),
                areas_of_agreement=data.get("areas_of_agreement", []),
                areas_of_disagreement=data.get("areas_of_disagreement", []),
                integrated_insights=data.get("integrated_insights", []),
                final_recommendation=data.get("final_recommendation", ""),
                synthesis_model=synthesis_model
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse consensus response: {e}")
            if response.strip():
                return ConsensusResult(
                    consensus=response.strip(),
                    confidence=0.5,
                    key_points=[],
                    areas_of_agreement=[],
                    areas_of_disagreement=[],
                    integrated_insights=[],
                    final_recommendation="",
                    synthesis_model=synthesis_model
                )
        return None
    async def build_consensus(
        self,
        question: str,
        responses: List[Dict[str, Any]],
        synthesizer: Any,  
    ) -> ConsensusResult:
        """
        Build consensus from multiple responses.
        """
        if len(responses) < self.min_responses:
            logger.warning(f"Only {len(responses)} responses, need at least {self.min_responses}")
            if responses:
                return ConsensusResult(
                    consensus=responses[0].get("content", ""),
                    confidence=0.3,
                    key_points=[],
                    areas_of_agreement=[],
                    areas_of_disagreement=[],
                    integrated_insights=[],
                    final_recommendation="Insufficient responses for full consensus",
                    synthesis_model=synthesizer.name,
                    original_responses=responses
                )
        responses_text = self.format_responses_for_prompt(responses)
        prompt = format_consensus_prompt(question, responses_text)
        result = await synthesizer.generate_async(
            prompt,
            format="json",
            temperature=0.5  
        )
        if result.success:
            consensus = self.parse_consensus_response(result.content, synthesizer.name)
            if consensus:
                consensus.original_responses = responses
                return consensus
        logger.warning("Consensus generation failed, using fallback")
        return ConsensusResult(
            consensus=responses[0].get("content", "") if responses else "",
            confidence=0.2,
            key_points=[],
            areas_of_agreement=[],
            areas_of_disagreement=["Consensus generation failed"],
            integrated_insights=[],
            final_recommendation="",
            synthesis_model=synthesizer.name,
            original_responses=responses
        )
    async def iterative_consensus(
        self,
        question: str,
        responses: List[Dict[str, Any]],
        synthesizers: List[Any],  
        max_iterations: int = 2
    ) -> ConsensusResult:
        """
        Build consensus iteratively with multiple synthesis rounds.
        """
        if not synthesizers:
            raise ValueError("At least one synthesizer required")
        current_consensus = None
        for i, synthesizer in enumerate(synthesizers[:max_iterations]):
            if i == 0:
                current_consensus = await self.build_consensus(
                    question,
                    responses,
                    synthesizer
                )
            else:
                refined_responses = responses + [{
                    "model": "previous_consensus",
                    "role": "synthesizer",
                    "content": current_consensus.consensus
                }]
                current_consensus = await self.build_consensus(
                    question,
                    refined_responses,
                    synthesizer
                )
        return current_consensus
    async def improve_response(
        self,
        question: str,
        original_response: str,
        feedback: str,
        improver: Any  
    ) -> str:
        """Improve a response based on feedback."""
        prompt = format_improve_prompt(question, original_response, feedback)
        result = await improver.generate_async(prompt)
        if result.success:
            return result.content
        return original_response  
@dataclass
class DebateRound:
    """A single round in a debate."""
    round_number: int
    arguments: List[Dict[str, Any]]
    summary: str = ""
class DebateManager:
    """
    Manages structured debates between council members.
    """
    def __init__(
        self,
        max_rounds: int = 5,
        enable_devil_advocate: bool = True
    ):
        self.max_rounds = max_rounds
        self.enable_devil_advocate = enable_devil_advocate
        self.debate_history: List[DebateRound] = []
    async def conduct_debate(
        self,
        topic: str,
        participants: List[Any],  
        moderator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Conduct a structured debate on a topic.
        """
        self.debate_history = []
        for round_num in range(1, self.max_rounds + 1):
            round_arguments = []
            previous_args = ""
            if self.debate_history:
                for prev_round in self.debate_history:
                    previous_args += f"\n**Round {prev_round.round_number}:**\n"
                    for arg in prev_round.arguments:
                        previous_args += f"- {arg['model']} ({arg['role']}): {arg['content'][:200]}...\n"
            async def get_argument(participant) -> Dict[str, Any]:
                prompt = format_debate_prompt(topic, previous_args, participant.role)
                result = await participant.generate_async(prompt)
                return {
                    "model": participant.name,
                    "role": participant.role,
                    "content": result.content if result.success else "",
                    "success": result.success
                }
            tasks = [get_argument(p) for p in participants]
            arguments = await asyncio.gather(*tasks)
            round_arguments = [a for a in arguments if a["success"]]
            summary = ""
            if moderator and round_arguments:
                summary_prompt = f"Summarize the key points from round {round_num} of this debate:\n"
                for arg in round_arguments:
                    summary_prompt += f"\n{arg['model']}: {arg['content']}\n"
                summary_result = await moderator.generate_async(summary_prompt, max_tokens=300)
                if summary_result.success:
                    summary = summary_result.content
            self.debate_history.append(DebateRound(
                round_number=round_num,
                arguments=round_arguments,
                summary=summary
            ))
            if self._check_consensus(round_arguments):
                logger.info(f"Consensus reached at round {round_num}")
                break
        return {
            "topic": topic,
            "rounds": len(self.debate_history),
            "history": [
                {
                    "round": r.round_number,
                    "arguments": r.arguments,
                    "summary": r.summary
                }
                for r in self.debate_history
            ],
            "participants": [p.name for p in participants]
        }
    def _check_consensus(self, arguments: List[Dict[str, Any]]) -> bool:
        """Check if arguments show consensus (simplified)."""
        if len(arguments) < 2:
            return True
        return False
