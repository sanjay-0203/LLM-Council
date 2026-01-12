
"""
Multi-Model Orchestration - Debate Example
Shows how to conduct a structured debate between models.
"""
import asyncio
import yaml
async def main():
    from council import LLMCouncil
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    council = LLMCouncil(config)
    await council.initialize()
    topic = "Should AI development be regulated by governments?"
    print("🎭 Multi-Model Orchestration - Debate Mode")
    print("="*60)
    print(f"Topic: {topic}")
    print("="*60 + "\n")
    active_members = council.model_manager.get_active_members()
    participant_names = [m.name for m in active_members[:4]]
    print(f"Participants ({len(participant_names)}):")
    for name in participant_names:
        print(f"  • {name}")
    print()
    result = await council.debate(
        topic=topic,
        participants=participant_names
    )
    print("\n" + "="*60)
    print("DEBATE RESULTS")
    print("="*60 + "\n")
    for round_data in result['history']:
        print(f"\n{'─'*60}")
        print(f"Round {round_data['round']}")
        print(f"{'─'*60}\n")
        for arg in round_data['arguments']:
            print(f"🗣️  {arg['model']} ({arg['role']}):")
            print(f"{arg['content'][:300]}...\n")
        if round_data.get('summary'):
            print(f"📝 Round Summary:")
            print(f"{round_data['summary']}\n")
    print("="*60)
    print(f"Debate completed in {result['rounds']} rounds")
    print("="*60)
    await council.shutdown()
if __name__ == "__main__":
    asyncio.run(main())
