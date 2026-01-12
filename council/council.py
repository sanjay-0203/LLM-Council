"""
LLM Council - The main orchestrator
"""
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from loguru import logger
from .models import ModelManager, CouncilMember
from .voting import VotingSystem, VotingResult, VotingMethod
from .evaluator import ResponseEvaluator, AggregatedEvaluation
from .consensus import ConsensusBuilder, DebateManager
from .calibration import ConfidenceCalibrator
from .persistence import CouncilDatabase
from .streaming import StreamingHandler
from .utils import create_session_id
class CouncilResponse:
    """Response from the council."""
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    @property
    def answer(self) -> str:
        """Get the final answer."""
        return self.data.get("final_answer", "")
    @property
    def session_id(self) -> str:
        return self.data.get("session_id", "")
    @property
    def responses(self) -> List[Dict[str, Any]]:
        return self.data.get("responses", [])
    @property
    def confidence(self) -> float:
        consensus = self.data.get("consensus", {})
        if isinstance(consensus, dict):
            return consensus.get("confidence", 0.5)
        return 0.5
    def to_dict(self) -> Dict[str, Any]:
        return self.data
class LLMCouncil:
    """The main council orchestrator."""
    def __init__(
        self,
        config: Dict[str, Any],
        model_manager: Optional[ModelManager] = None,
        database: Optional[CouncilDatabase] = None
    ):
        self.config = config
        self.model_manager = model_manager or ModelManager()
        self.database = database or CouncilDatabase()
        voting_method_str = config.get("council", {}).get("voting_method", "weighted")
        try:
            voting_method = VotingMethod(voting_method_str)
        except ValueError:
            voting_method = VotingMethod.WEIGHTED
        self.voting = VotingSystem(
            default_method=voting_method,
            min_agreement_threshold=config.get("council", {}).get("min_agreement_threshold", 0.6)
        )
        self.evaluator = ResponseEvaluator()
        self.consensus_builder = ConsensusBuilder()
        self.calibrator = ConfidenceCalibrator()
        self.stream_handler = StreamingHandler()
        self.debate_manager = DebateManager(
            max_rounds=config.get("council", {}).get("max_debate_rounds", 5),
            enable_devil_advocate=config.get("council", {}).get("enable_devil_advocate", True)
        )
        self.session_id = str(uuid.uuid4())
    async def initialize(self):
        """Initialize all components."""
        await self.model_manager.initialize()
        await self.database.initialize()
        for model_cfg in self.config.get("models", []):
            if model_cfg.get("enabled", False):
                await self.model_manager.add_member(**model_cfg)
        active_count = len(self.model_manager.get_active_members())
        logger.info(f"LLM Council initialized with {active_count} active members")
    async def ask(
        self,
        question: str,
        members: Optional[List[str]] = None,
        use_voting: bool = True,
        use_evaluation: bool = True,
        use_consensus: bool = True,
        stream: bool = False
    ) -> CouncilResponse:
        """Main entry point: ask the council a question."""
        session_id = create_session_id(question)
        self.session_id = session_id
        if members:
            selected_members = [
                m for m in self.model_manager.members.values()
                if m.name in members
            ]
        else:
            selected_members = self.model_manager.get_active_members()
        if not selected_members:
            raise ValueError("No active council members")
        logger.info(f"Council session {session_id}: {len(selected_members)} members responding")
        responses = await self.model_manager.parallel_generate(question)
        successful_responses = [r for r in responses if r["success"]]
        if not successful_responses:
            return CouncilResponse({
                "session_id": session_id,
                "question": question,
                "final_answer": "No successful responses from council members",
                "responses": responses,
                "num_responses": 0,
                "council_size": len(selected_members),
                "error": "All council members failed to respond"
            })
        evaluation = None
        if use_evaluation and len(successful_responses) > 1:
            combined_response = "\n\n".join([r["content"] for r in successful_responses])
            evaluation = await self.evaluator.evaluate(
                question,
                combined_response,
                selected_members
            )
        winner_response = None
        voting_result = None
        if use_voting and len(successful_responses) > 1:
            candidates = [r["content"] for r in successful_responses]
            voting_result = await self.voting.collect_votes(
                question,
                candidates,
                selected_members
            )
            if voting_result.winner <= len(candidates):
                winner_response = candidates[voting_result.winner - 1]
        final_answer = winner_response
        consensus_result = None
        if use_consensus and len(successful_responses) > 1:
            synthesizers = self.model_manager.get_members_by_role("synthesizer")
            if not synthesizers:
                synthesizers = selected_members[:1]
            if synthesizers:
                consensus_result = await self.consensus_builder.build_consensus(
                    question,
                    successful_responses,
                    synthesizers[0]
                )
                final_answer = consensus_result.consensus
        if not final_answer and successful_responses:
            final_answer = successful_responses[0]["content"]
        await self.database.save_session(
            session_id=session_id,
            question=question,
            responses=successful_responses,
            votes=voting_result.to_dict() if voting_result else None,
            consensus=consensus_result.to_dict() if consensus_result else None,
            evaluation=evaluation.to_dict() if evaluation else None
        )
        result_data = {
            "session_id": session_id,
            "question": question,
            "final_answer": final_answer,
            "responses": successful_responses,
            "voting": voting_result.to_dict() if voting_result else None,
            "evaluation": evaluation.to_dict() if evaluation else None,
            "consensus": consensus_result.to_dict() if consensus_result else None,
            "num_responses": len(successful_responses),
            "council_size": len(selected_members)
        }
        return CouncilResponse(result_data)
    async def debate(
        self,
        topic: str,
        participants: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Conduct a structured debate."""
        members = [
            self.model_manager.get_member(name)
            for name in (participants or [])
            if self.model_manager.get_member(name)
        ] or self.model_manager.get_active_members()[:6]
        result = await self.debate_manager.conduct_debate(topic, members)
        return result
    async def evaluate_response(
        self,
        question: str,
        response: str,
        evaluators: Optional[List[str]] = None
    ) -> AggregatedEvaluation:
        """Evaluate a specific response."""
        if evaluators:
            eval_members = [
                m for m in self.model_manager.members.values()
                if m.name in evaluators
            ]
        else:
            eval_members = self.model_manager.get_active_members()[:5]
        return await self.evaluator.evaluate(question, response, eval_members)
    async def shutdown(self):
        """Shutdown the council."""
        await self.model_manager.shutdown()
        logger.info("LLM Council shut down")
