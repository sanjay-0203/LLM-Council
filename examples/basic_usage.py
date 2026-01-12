
"""
Multi-Model Orchestration - Basic Usage Example
Simple example showing how to use the orchestration programmatically.
"""
import asyncio
import yaml
async def main():
    from council import LLMCouncil
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    council = LLMCouncil(config)
    await council.initialize()
    print("🎭 Multi-Model Orchestration initialized!")
    print(f"Active members: {len(council.model_manager.get_active_members())}\n")
    question = "What are the three laws of robotics?"
    print(f"Question: {question}\n")
    print("Consulting the council...\n")
    result = await council.ask(question)
    print("="*60)
    print("COUNCIL ANSWER:")
    print("="*60)
    print(result.answer)
    print()
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Based on {result.data['num_responses']} model responses")
    print("="*60)
    await council.shutdown()
if __name__ == "__main__":
    asyncio.run(main())
