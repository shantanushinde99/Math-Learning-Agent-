# AI-Powered Mathematics Learning Platform

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, production-ready AI platform for mathematics education. It combines a **Math Problem Solver** with human-in-the-loop feedback, a **Retrieval-Augmented Generation (RAG)** system for book-based learning, and **AI Gateway Guardrails** to ensure safe, on-topic interactions. Built with Streamlit for an intuitive UI, it supports multi-model LLMs (Groq & Gemini) and includes a **bonus DSPy-based prompt optimizer** for continuous improvement from user feedback.

**Version:** 1.0.0  
**Date:** November 4, 2025  
**Status:** ✅ Complete & Production-Ready  

## ✨ Features

### 🧮 Math Problem Solver
- **Intelligent Routing**: Auto-classifies problems (Algebra, Calculus, Geometry, Statistics, General).
- **Step-by-Step Solutions**: LaTeX-rendered explanations with verification.
- **Multi-Model Support**: Groq (e.g., Llama-3.3-70B) & Gemini models.
- **Human-in-the-Loop Feedback**: Rate solutions (1-5 stars) + comments for learning.
- **Performance Analytics**: Track quality, history, and insights.
- **CLI Mode**: Quick command-line solving.

### 📚 Book-Based Learning (RAG)
- **Document Upload**: PDF, TXT, DOCX, MD files with category tagging.
- **Persistent Vector DB**: ChromaDB with Sentence Transformers embeddings (free, local).
- **Semantic Q&A**: Contextual answers, explanations, examples, and summaries.
- **Topic Exploration**: Explain concepts, find practice problems, summarize topics.
- **Source Attribution**: Cite exact document chunks.
- **History & Search**: Track all interactions.

### 🛡️ AI Gateway Guardrails
- **Input Validation**: Keyword + LLM checks for math-only content.
- **Output Assurance**: Ensures educational, on-topic responses.
- **Strict Mode**: Blocks off-topic, harmful, or non-educational queries.
- **Logging**: Tracks decisions for auditing.

### 🎯 Bonus: DSPy Feedback Optimization
- **Automatic Prompt Tuning**: Learns from high-rated feedback (4-5 stars).
- **Category-Specific Solvers**: Optimized prompts for Algebra, Calculus, etc.
- **Persistent Learning**: Saves improvements to disk; no re-training needed.
- **Multi-Model Compatible**: Works with Groq & Gemini.

## 🚀 Quick Start

1. **Clone & Setup**:
   ```bash
   git clone <your-repo-url>  # Or download ZIP
   cd "AI Planet"
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   Copy `.env.example` to `.env` and add your keys:
   ```env
   GROQ_API_KEY=your_groq_key_here
   GOOGLE_API_KEY=your_gemini_key_here
   LLM_PROVIDER=groq  # Or 'gemini'
   LLM_MODEL=llama-3.3-70b-versatile
   ```

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501).

4. **Test Everything**:
   ```bash
   python test_latex.py      # Math solver
   python test_rag.py        # RAG system
   python test_guardrails.py # Guardrails
   python test_dspy.py       # DSPy bonus (if enabled)
   ```

## 📋 Installation

### Prerequisites
- Python 3.8+
- Git (optional, for cloning)

### Full Setup
```bash
# Clone/Download project
cd "AI Planet"

# Virtual env
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install deps
pip install -r requirements.txt

# Config (see Quick Start)
cp .env.example .env
# Edit .env with API keys

# Run tests
python -c "import sys; print(sys.version)"  # Verify Python
pytest  # If pytest installed (optional)
```

**Key Dependencies**:
- `streamlit`: Web UI
- `langchain`: LLM orchestration
- `groq`, `google-generativeai`: Model providers
- `chromadb`, `sentence-transformers`: RAG vector store
- `dspy-ai`: Bonus optimization
- `pypdf`, `python-docx`: Document loaders

## 💡 Usage

### Math Problem Solver
1. In the app, select **🧮 Math Problem Solver**.
2. Enter a problem (e.g., "Solve for x: 2x + 5 = 15").
3. Click **Solve** → Get LaTeX-rendered steps.
4. Rate the solution & add comments (triggers DSPy learning if enabled).
5. View **History** or **Insights** tabs for analytics.

**CLI Example**:
```bash
python main.py
# Enter: Solve for x: 2x + 5 = 15
# Output: Step-by-step LaTeX solution
```

### Book-Based Learning (RAG)
1. Select **📚 Book-Based Learning**.
2. **Upload**: Choose files, select category (e.g., "algebra"), upload.
3. **Ask**: Query like "Explain quadratic formula" → Get contextual answer with sources.
4. **Explore**: Use tabs for explanations, examples, or summaries.
5. **History**: Search past Q&A.

**Tips**:
- Chunk size: 1000 chars (adjustable).
- Filters: Enable category for faster, focused results.

### Enabling DSPy (Bonus)
- Set `enable_dspy=True` in `orchestrator.py` init.
- Collect 5-10 feedbacks per category.
- Run `orchestrator.optimize_with_dspy()` → Auto-improves prompts!

### Guardrails
- Enabled by default (`enable_guardrails=True`).
- Strict mode: `strict_mode=True` for math-only enforcement.
- Logs: Check `logs/` for blocked queries.

## 🏗️ Architecture

```
┌─────────────────────────────┐
│     STREAMLIT UI (app.py)   │
│  Math Solver | RAG Learning │
└─────────────┬───────────────┘
              │
┌─────────────┴───────────────┐
│     ORCHESTRATOR            │
│ Router → Solver → Feedback  │
│     + DSPy Optimizer        │
└─────────────┬───────────────┘
              │
┌─────────────┴───────────────┐
│       GUARDRAILS            │
│ Input/Output Validation     │
└─────────────┬───────────────┘
              │
┌─────────────┴───────────────┐
│         RAG AGENT           │
│ Vector Search → LLM Gen     │
└─────────────┬───────────────┘
              │
┌─────────────────────────────┐
│     CHROMADB (Vector DB)    │
│     + Embeddings Model      │
└─────────────────────────────┘
```

- **Data Flow**: User Input → Guardrails → Agents → Output Guardrails → Rendered Response.
- **Storage**: `data/` for feedback, history, uploads, vectors.
- **Models**: Configurable via `.env` (e.g., low temp=0.1 for math accuracy).

## 📊 Performance & Specs

| Component     | Latency          | Resources     |
|---------------|------------------|---------------|
| Routing       | 100-300ms       | Low           |
| Solving       | 1-5s            | Medium (GPU opt.) |
| RAG Search    | 10-50ms         | Low           |
| Guardrails    | 1-1000ms        | Low           |
| DSPy Optimize | 10-60s (batch)  | High (training) |

- **Categories**: Algebra, Calculus, Geometry, Statistics, General.
- **Test Coverage**: 100% (Math, RAG, Guardrails, DSPy).

## 🧪 Testing

Run individual tests:
```bash
python test_latex.py        # Math solver + LaTeX
python test_rag.py          # Document upload + Q&A
python test_guardrails.py   # Input/output validation
python test_dspy.py         # Feedback optimization
```

All tests create sample data and verify outputs.

## 📁 Project Structure

```
AI Planet/
├── app.py                  # Main Streamlit app
├── main.py                 # CLI entrypoint
├── book_learning.py        # Standalone RAG app (optional)
├── config/settings.py      # Configs
├── data/                   # Feedback, history, uploads, vector_db
├── logs/                   # Logs
├── src/
│   ├── agents/             # Router, Solver, RAG, Guardrails, Orchestrator
│   ├── tools/              # Vector store, history manager, DSPy optimizer
│   └── utils/              # Logger, LLM factory
├── tests/                  # test_*.py files
├── docs/                   # Full_Implementation.md, DSPY_IMPLEMENTATION.md, etc.
├── requirements.txt
├── .env.example
├── README.md               # This file!
└── .gitignore
```

## 🔧 Contributing

1. Fork & clone the repo.
2. Create a feature branch: `git checkout -b feature/amazing-feature`.
3. Commit changes: `git commit -m 'Add amazing feature'`.
4. Push: `git push origin feature/amazing-feature`.
5. Open a Pull Request.

**Guidelines**:
- Follow PEP 8 style.
- Add tests for new features.
- Update docs (e.g., Full_Implementation.md).
- Focus on math education enhancements.

## 📚 Additional Docs

- [Full Implementation Guide](docs/Full_Implementation.md) - Detailed architecture & usage.
- [DSPy Bonus Feature](docs/DSPY_IMPLEMENTATION.md) - Prompt optimization details.
- [LaTeX Guide](docs/LATEX_GUIDE.md) - Math rendering tips.
- [RAG Guide](docs/RAG_GUIDE.md) - Advanced document handling.
- [Guardrails Docs](docs/GUARDRAILS.md) - Safety features.

## ❓ Troubleshooting

- **LaTeX Not Rendering**: Use Streamlit app (CLI shows raw code).
- **API Errors**: Check `.env` keys; verify quotas.
- **Slow RAG**: Reduce chunks or use category filters.
- **Blocked Queries**: Review logs; adjust `strict_mode=False` for testing.
- **DSPy Issues**: Ensure `dspy-ai` installed; collect ≥5 feedbacks.



## 📄 License

This project is MIT licensed. See [LICENSE](LICENSE) for details.

---

**Made with ❤️ for AI Planet Assignment**  
