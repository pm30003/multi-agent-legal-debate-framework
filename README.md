# STRUCTURED MULTI AGENT DEBATE FRAMEWORK

## ABSTRACT

The legal system faces significant challenges in processing complex cases efficiently, with traditional methods requiring extensive human resources and time. This project presents an AI-powered Legal Judgment Framework that leverages Large Language Models (LLMs) and multi-agent systems to simulate structured legal debates, analyze cases, and generate reasoned judicial decisions. The system employs multiple AI agents representing different legal perspectives (plaintiff, defendant, and judge) that engage in iterative debate rounds, retrieve relevant legal precedents and judgements using Retrieval Augmented Generation (RAG) and converge toward consensus through algorithmic analysis..

The framework integrates advanced technologies including Groq's fast LLM inference, vector databases for legal document retrieval, LSTM-based memory systems for learning from past debates, and consensus calculation algorithms. It provides a RESTful API interface for case submission and receives structured legal analysis with supporting citations, consensus metrics, and final judgments. The system demonstrates potential applications in legal research assistance, case analysis automation, and educational tools for law students, while maintaining transparency through explainable AI reasoning chains.

Keywords: Legal AI, Multi-Agent Systems, Retrieval Augmented Generation, Legal Judgment, Consensus Algorithm, LSTM Memory, Natural Language Processing, Judicial Decision Making

---

# Quick Start Guide - Legal Judgment Framework

## Step 1: Setup Groq API Key

Create a `.env` file in the `multi-agent-legal-judgment-framework` folder:

```cmd
echo GROQ_API_KEY=your_api_key_here > .env
```

(Get free API key from https://console.groq.com/)

## Step 2: Install Dependencies

```cmd
cd multi-agent-legal-judgment-framework
pip install -r requirements.txt
pip install langchain-groq
pip install langchain langchain-core langchain-community  
pip install fastapi uvicorn python-dotenv pydantic
pip install sentence-transformers faiss-cpu langchain-huggingface
```

## Step 3: Setup Legal Documents (Optional)

```cmd
python setup_legal_documents.py
```

---

## Running the Project (2 Terminals)

### Terminal 1: Start the Server

```cmd
cd multi-agent-legal-judgment-framework
python main.py
```

Wait for: "Starting Groq-Powered Legal Framework on http://0.0.0.0:8000"

**Keep this terminal running!**

---

### Terminal 2: Run Legal Debate

Open a **NEW** Command Prompt window:

**Option 1: Show output in terminal**
```cmd
cd multi-agent-legal-judgment-framework
python debate_cli.py input.json
```

**Option 2: Save to text file**
```cmd
cd multi-agent-legal-judgment-framework
python debate_cli_ultra_raw.py input.json
```
This creates a separate `.txt` file with all debate rounds.

**Note:** Edit `input.json` to change the case details before running.

---

## Stop the Server

In Terminal 1, press: `Ctrl + C`

---

## Troubleshooting

**Error: "GROQ_API_KEY not found"**
- Check `.env` file exists in `multi-agent-legal-judgment-framework` folder
- Make sure it contains: `GROQ_API_KEY=gsk_xxxxx...`

**Error: "Module not found"**
- Run: `pip install -r requirements.txt`

**Error: "Port 8000 already in use"**
- Close other programs using port 8000
- Or edit `main.py` line 637: change `port=8000` to `port=8001`

**Error: "Legal documents not found"**
- Run: `python setup_legal_documents.py`

---

## Summary

**Terminal 1:** 
```cmd
cd multi-agent-legal-judgment-framework
python main.py
```

**Terminal 2:** 
```cmd
cd multi-agent-legal-judgment-framework
python debate_cli.py input.json
```
OR
```cmd
python debate_cli_ultra_raw.py input.json
```

