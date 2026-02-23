# Project — Multi-Agent Supply Chain System

This directory contains the complete implementation and evaluation artifacts.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:

```
UDACITY_OPENAI_API_KEY=your_key_here
```

## Running

```bash
python project_starter.py
```

Processes 20 customer requests from `quote_requests_sample.csv` through the multi-agent pipeline and writes results to `test_results.csv`.

## Files

| File | Description |
|------|-------------|
| `project_starter.py` | Complete multi-agent implementation |
| `reflection_report.md` | Architecture decisions, evaluation analysis, improvement proposals |
| `workflow_diagram.png` | Visual architecture and data flow diagram |
| `workflow_diagram.dot` | Graphviz source for the diagram |
| `test_results.csv` | Evaluation output for all 20 sample requests |
| `quote_requests_sample.csv` | Input: 20 simulated customer requests |
| `quote_requests.csv` | Full customer request dataset |
| `quotes.csv` | Historical quote reference data |
| `requirements.txt` | Python dependencies |
