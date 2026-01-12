
"""
Multi-Model Orchestration - Benchmark Script
Tests orchestration performance with various questions and voting methods.
"""
import asyncio
import yaml
import time
import json
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any
from council import LLMCouncil
from council.voting import VotingMethod
from loguru import logger
BENCHMARK_QUESTIONS = [
    {
        "question": "What are the key principles of good software architecture?",
        "domain": "technical",
        "expected_elements": ["modularity", "scalability", "maintainability"]
    },
    {
        "question": "Explain the water cycle in simple terms.",
        "domain": "science",
        "expected_elements": ["evaporation", "condensation", "precipitation"]
    },
    {
        "question": "What makes a good leader?",
        "domain": "general",
        "expected_elements": ["communication", "vision", "empathy"]
    },
    {
        "question": "How does encryption protect data?",
        "domain": "technical",
        "expected_elements": ["algorithm", "key", "security"]
    },
    {
        "question": "What are the benefits of renewable energy?",
        "domain": "environment",
        "expected_elements": ["sustainable", "clean", "renewable"]
    }
]
class CouncilBenchmark:
    """Benchmark the LLM Council."""
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.results: List[Dict[str, Any]] = []
    async def run(self):
        """Run the full benchmark suite."""
        print("\n" + "="*60)
        print("🎭 MULTI-MODEL ORCHESTRATION BENCHMARK")
        print("="*60 + "\n")
        council = LLMCouncil(self.config)
        await council.initialize()
        active_members = len(council.model_manager.get_active_members())
        print(f"Active Members: {active_members}")
        print(f"Questions: {len(BENCHMARK_QUESTIONS)}\n")
        voting_methods = [
            VotingMethod.MAJORITY,
            VotingMethod.WEIGHTED,
            VotingMethod.RANKED,
            VotingMethod.BORDA
        ]
        for method in voting_methods:
            print(f"\n{'─'*60}")
            print(f"Testing: {method.value.upper()}")
            print(f"{'─'*60}\n")
            method_results = await self._test_voting_method(council, method)
            self.results.append({
                "voting_method": method.value,
                "results": method_results
            })
        self._generate_report()
        await council.shutdown()
    async def _test_voting_method(
        self,
        council: LLMCouncil,
        method: VotingMethod
    ) -> Dict[str, Any]:
        """Test a specific voting method."""
        original_method = council.voting.default_method
        council.voting.default_method = method
        response_times = []
        consensus_levels = []
        token_counts = []
        for i, benchmark in enumerate(BENCHMARK_QUESTIONS, 1):
            question = benchmark["question"]
            print(f"  [{i}/{len(BENCHMARK_QUESTIONS)}] {question[:50]}...")
            start = time.time()
            result = await council.ask(
                question,
                use_voting=True,
                use_evaluation=False,
                use_consensus=False
            )
            elapsed = time.time() - start
            response_times.append(elapsed)
            if result.data.get('voting'):
                consensus_levels.append(
                    result.data['voting'].get('consensus_level', 0)
                )
            total_tokens = sum(
                r.get('tokens', 0)
                for r in result.data.get('responses', [])
            )
            token_counts.append(total_tokens)
            print(f"      ✓ {elapsed:.2f}s | Consensus: {consensus_levels[-1]:.1%}")
        council.voting.default_method = original_method
        return {
            "avg_response_time": mean(response_times),
            "std_response_time": stdev(response_times) if len(response_times) > 1 else 0,
            "avg_consensus": mean(consensus_levels) if consensus_levels else 0,
            "avg_tokens": mean(token_counts) if token_counts else 0,
            "total_time": sum(response_times)
        }
    def _generate_report(self):
        """Generate benchmark report."""
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS")
        print("="*60 + "\n")
        for result in self.results:
            method = result["voting_method"]
            metrics = result["results"]
            print(f"📋 {method.upper()}")
            print(f"   Avg Response Time: {metrics['avg_response_time']:.2f}s "
                  f"(±{metrics['std_response_time']:.2f}s)")
            print(f"   Avg Consensus:     {metrics['avg_consensus']:.1%}")
            print(f"   Avg Tokens:        {metrics['avg_tokens']:.0f}")
            print(f"   Total Time:        {metrics['total_time']:.2f}s")
            print()
        best_consensus = max(self.results, key=lambda x: x['results']['avg_consensus'])
        print(f"🏆 Best Consensus: {best_consensus['voting_method']}")
        fastest = min(self.results, key=lambda x: x['results']['avg_response_time'])
        print(f"⚡ Fastest: {fastest['voting_method']}")
        output_file = Path("benchmark_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "results": self.results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
async def main():
    """Main entry point."""
    benchmark = CouncilBenchmark()
    await benchmark.run()
if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: None)  
    asyncio.run(main())
