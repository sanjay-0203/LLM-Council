#!/usr/bin/env python3
"""
LLM Council - CLI Interface
"""

import asyncio
import yaml
import sys
from pathlib import Path

from council import LLMCouncil
from loguru import logger


async def main():
    """Main CLI entry point."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found. Please create it from the template.")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize council
    logger.info("Initializing LLM Council...")
    council = LLMCouncil(config)
    
    try:
        await council.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize council: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        return
    
    print("\n" + "="*60)
    print("🏛️  LLM COUNCIL - Multi-Model AI Consensus System")
    print("="*60)
    print(f"\nActive Members: {len(council.model_manager.get_active_members())}")
    for member in council.model_manager.get_active_members():
        print(f"  • {member.name} ({member.role})")
    
    print("\nCommands:")
    print("  - Type your question to ask the council")
    print("  - 'stats' - Show council statistics")
    print("  - 'quit' or 'exit' - Exit the program")
    print("\n" + "="*60 + "\n")
    
    while True:
        try:
            question = input("\n🤔 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == "stats":
                stats = council.model_manager.get_all_stats()
                print("\n📊 Council Statistics:")
                print(f"  Total Members: {stats['total_members']}")
                print(f"  Active Members: {stats['active_members']}")
                print(f"  Backends: {', '.join(stats['backends'])}")
                print("\n  Member Stats:")
                for member_stats in stats['members']:
                    if member_stats['total_calls'] > 0:
                        print(f"    {member_stats['name']}:")
                        print(f"      Calls: {member_stats['total_calls']}")
                        print(f"      Success Rate: {member_stats['success_rate']:.1%}")
                        print(f"      Avg Response: {member_stats['avg_response_time']:.2f}s")
                continue
            
            # Ask the council
            print("\n💭 Consulting the council...")
            
            result = await council.ask(
                question,
                use_voting=True,
                use_evaluation=True,
                use_consensus=True
            )
            
            print(f"\n{'='*60}")
            print("📜 COUNCIL ANSWER")
            print(f"{'='*60}\n")
            print(result.answer)
            
            # Show metadata
            print(f"\n{'─'*60}")
            print(f"📊 Confidence: {result.confidence:.1%}")
            print(f"📝 Responses: {result.data['num_responses']}/{result.data['council_size']} members")
            
            if result.data.get('voting'):
                voting = result.data['voting']
                print(f"🗳️  Consensus: {voting['consensus_level']:.1%}")
            
            print(f"{'─'*60}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}\n")
    
    # Shutdown
    await council.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
