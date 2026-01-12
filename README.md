# 🏛️ LLM Council

**Open-Source Multi-Model AI Consensus System**

LLM Council is a production-ready Python framework that orchestrates multiple local LLMs to work together, providing more reliable and well-rounded AI responses through voting, evaluation, and consensus building.

## ✨ Features

- **🗳️Multiple Voting Methods**: Majority, weighted, ranked-choice, Borda count, approval, and Condorcet
- **📊 Response Evaluation**: Multi-criteria scoring with aggregated feedback
- **🤝 Consensus Building**: Synthesize insights from multiple models
- 💬 **Debate Mode**: Structured multi-round discussions
- **🎯 Confidence Calibration**: Adaptive confidence scoring based on historical accuracy
- **💾 Persistence**: SQLite-based session storage
- **⚡ Streaming Support**: Real-time response streaming
- **🔌 Multiple Backends**: Ollama (primary), vLLM, llama.cpp

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** ([installation guide](https://ollama.ai))

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull some models
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull phi3:medium
ollama pull gemma2:9b
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-council.git
cd llm-council

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `config.yaml` to enable your installed models:
```yaml
models:
  - name: "llama3.1:8b"
    role: "general_reasoner"
    enabled: true  # Set to true
    
  - name: "mistral:7b"
    role: "analytical_thinker"
    enabled: true  # Set to true
```

### Running

```bash
# Start the CLI
python main.py

# Or with Streamlit UI (coming soon)
# streamlit run streamlit_app.py
```

## 📖 Usage Examples

### Basic Question

```python
import asyncio
import yaml
from council import LLMCouncil

async def main():
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize council
    council = LLMCouncil(config)
    await council.initialize()
    
    # Ask a question
    result = await council.ask(
        "What are the key factors to consider when designing a database schema?"
    )
    
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Participants: {result.data['num_responses']} models")
    
    await council.shutdown()

asyncio.run(main())
```

### Debate Mode

```python
# Conduct a structured debate
debate_result = await council.debate(
    topic="Should we prioritize performance or readability in code?",
    participants=["llama3.1:8b", "mistral:7b", "gemma2:9b"]
)

for round in debate_result['history']:
    print(f"\nRound {round['round']}:")
    for arg in round['arguments']:
        print(f"  {arg['model']}: {arg['content'][:200]}...")
```

### Custom Voting

```python
from council.voting import VotingMethod

result = await council.ask(
    "Explain quantum computing",
    use_voting=True,
    use_evaluation=True,
    use_consensus=True
)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         LLM Council                 │
│  ┌──────────────────────────────┐  │
│  │   Model Manager              │  │
│  │  • Ollama Backend            │  │
│  │  • vLLM Backend (optional)   │  │
│  │  • LlamaCpp Backend          │  │
│  └──────────────────────────────┘  │
│                                      │
│  ┌──────────────────────────────┐  │
│  │   Voting System              │  │
│  │  • 7 voting methods          │  │
│  │  • Confidence weighting      │  │
│  └──────────────────────────────┘  │
│                                      │
│  ┌──────────────────────────────┐  │
│  │   Evaluator                  │  │
│  │  • Multi-criteria scoring    │  │
│  │  • Aggregated feedback       │  │
│  └──────────────────────────────┘  │
│                                      │
│  ┌──────────────────────────────┐  │
│  │   Consensus Builder          │  │
│  │  • Response synthesis        │  │
│  │  • Debate management         │  │
│  └──────────────────────────────┘  │
│                                      │
│  ┌──────────────────────────────┐  │
│  │   Persistence Layer          │  │
│  │  • SQLite database           │  │
│  │  • Session history           │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## 🎯 Model Roles

The council supports specialized roles for different models:

- **General Reasoner**: Balanced, well-rounded responses
- **Analytical Thinker**: Logic-focused, evidence-based analysis
- **Creative Thinker**: Novel perspectives and innovative ideas
- **Concise Responder**: Brief, focused answers
- **Technical Expert**: Detailed technical accuracy
- **Devil's Advocate**: Challenges assumptions
- **Synthesizer**: Combines diverse perspectives
- **Fact Checker**: Verifies claims
- **Ethical Reviewer**: Considers ethical implications

## 🛠️ Configuration

See `config.yaml` for full configuration options:

- Council behavior (voting methods, thresholds)
- Model definitions (roles, weights, parameters)
- Evaluation criteria
- Backend settings (Ollama, vLLM, llama.cpp)
- Persistence and caching
- Logging

## 📊 Voting Methods

1. **Majority**: Simple majority wins
2. **Weighted**: Member weight × confidence
3. **Unanimous**: Requires full agreement
4. **Ranked**: Instant-runoff voting
5. **Borda Count**: Points by ranking
6. **Approval**: Multiple approvals
7. **Condorcet**: Pairwise comparisons

## 🔧 Development

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black council/
isort council/

# Type checking
mypy council/
```

## 📝 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 🙏 Acknowledgments

- Built with [Ollama](https://ollama.ai)
- Inspired by ensemble methods in ML
- Thanks to the open-source LLM community

## 📚 Citation

If you use LLM Council in your research, please cite:

```bibtex
@software{llm_council,
  title = {LLM Council: Multi-Model AI Consensus System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/llm-council}
}
```

---

**Made with ❤️ for the open-source AI community**
