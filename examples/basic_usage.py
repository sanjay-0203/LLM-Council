#!/usr/bin/env python3
"""
LLM Council - Basic Usage Example

Simple example showing how to use the council programmatically.
"""

import asyncio
import yaml


async def main():
    # Import after making sure council package is available
    from council import LLMCouncil
    
    # Load configuration
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize the council
    council = LLMCouncil(config)
    await council.initialize()
    
    print("🏛️  LLM Council initialized!")
    print(f"Active members: {len(council.model_manager.get_active_members())}\n")
    
    # Ask a simple question
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
    
    # Cleanup
    await council.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
