# Multi-Agent Supply Chain System

A multi-agent system that automates inventory management, quote generation, and order fulfillment for a paper supply business. Built with the **smolagents** framework (Hugging Face) and GPT-4o-mini, the system coordinates five specialized AI agents through a deterministic orchestrator to process customer requests end-to-end.

## Architecture

The system follows an **Orchestrator-Worker** pattern where a central Python class delegates tasks to specialized `ToolCallingAgent` instances:

| Agent | Role | Key Tools |
|-------|------|-----------|
| **Orchestrator** | Central coordinator (deterministic Python class, not LLM-based) | Item parsing, catalog matching, workflow management |
| **Inventory Agent** | Catalog matching and stock verification | `match_item_tool`, `check_stock_tool`, `check_inventory_tool` |
| **Quote Agent** | Pricing with bulk discounts and historical context | `calculate_quote_tool`, `search_quotes_tool` |
| **Sales Agent** | Transaction recording and cash management | `finalize_sale_tool`, `reorder_stock_tool`, `check_delivery_tool` |
| **Business Advisor** | Financial analysis and proactive recommendations | `financial_report_tool`, `check_balance_tool` |
| **Customer Agent** | Simulates realistic customer feedback on quotes | Observation-only (external to internal system) |

### Request Processing Pipeline

```
PARSE → INVENTORY → QUOTE → SALES → REORDER → RESPOND → CUSTOMER FEEDBACK
```

Each customer request flows through a 10-stage pipeline: parse natural language into item-quantity pairs, match to product catalog via multi-phase string matching, verify stock, calculate pricing with bulk discounts (5%-15%), record transactions, evaluate reorders, compose customer response, and simulate customer feedback.

## Key Design Decisions

- **Deterministic orchestrator**: Critical business logic (item matching, pricing, transaction recording) runs as plain Python — not through LLM inference — ensuring consistent, auditable behavior across all requests.
- **Multi-phase item matching**: Synonym lookup → exact match → substring containment → word-overlap scoring with paper-size guards. Handles diverse customer phrasings without requiring embedding models.
- **Graceful degradation**: Agent failures fall back to direct function calls, ensuring business-critical operations always complete.
- **Proactive business intelligence**: The Business Advisor provides mid-session analysis every 5 requests, surfacing demand trends and inventory concerns before they become critical.

## Results

| Metric | Value |
|--------|-------|
| Requests processed | 20 |
| Successfully fulfilled | 17 (85%) |
| Cash balance change | $45,121 → $45,536 (+$415) |
| Inventory value change | $4,940 → $4,384 |

Three requests were unfulfilled due to out-of-stock items or products not in the catalog — demonstrating the system's graceful handling of edge cases rather than silent failures.

## Tech Stack

- **Agent Framework**: [smolagents](https://github.com/huggingface/smolagents) (Hugging Face)
- **Language Model**: GPT-4o-mini (OpenAI)
- **Database**: SQLite via SQLAlchemy
- **Data Processing**: Pandas, NumPy

## Setup

```bash
pip install -r project/requirements.txt
```

Create a `.env` file with your OpenAI API key:

```
UDACITY_OPENAI_API_KEY=your_key_here
```

## Usage

```bash
cd project
python project_starter.py
```

The system processes all 20 sample customer requests with animated terminal output showing real-time workflow progression, then generates `test_results.csv` with full interaction logs.

## Project Structure

```
project/
├── project_starter.py          # Complete multi-agent implementation
├── reflection_report.md        # Architecture decisions and evaluation analysis
├── workflow_diagram.png        # Visual architecture diagram
├── workflow_diagram.dot        # Graphviz source
├── test_results.csv            # Evaluation output for 20 requests
├── quote_requests_sample.csv   # Input test dataset
├── quote_requests.csv          # Full customer request data
├── quotes.csv                  # Historical quote reference data
└── requirements.txt            # Python dependencies
```
