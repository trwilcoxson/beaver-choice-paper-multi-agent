# Beaver's Choice Paper Company — Multi-Agent System Reflection Report

## 1. System Architecture and Agent Workflow

### Overview

The multi-agent system implements six components — four specialized worker agents, one external customer simulation agent, and one orchestrator — built using the **smolagents** framework with GPT-4o-mini as the underlying language model. The architecture follows the Orchestrator-Worker pattern where a central Python class coordinates task delegation to specialized `ToolCallingAgent` instances.

### Agent Roles and Responsibilities

**Orchestrator (PaperCompanyOrchestrator)** — A plain Python class (not a ToolCallingAgent) that serves as the central coordinator. It receives customer requests, parses them into structured item-quantity pairs, delegates tasks to worker agents, manages the overall workflow, and composes the final customer-facing response. The decision to make the orchestrator a Python class rather than an LLM-based agent was deliberate: critical business logic (item matching, pricing calculations, transaction recording) benefits from deterministic execution rather than probabilistic LLM output.

**Inventory Agent** — A `ToolCallingAgent` responsible for matching customer-requested items to the product catalog and verifying stock availability. It has access to three tools:
- `check_inventory_tool` (wraps `get_all_inventory()`) — retrieves all items in stock
- `check_stock_tool` (wraps `get_stock_level()`) — checks a specific item's stock
- `match_item_tool` — multi-phase catalog matching using synonym lookup, exact match, substring containment, and word-overlap scoring

**Quote Agent** — A `ToolCallingAgent` responsible for pricing and historical quote research. It has access to:
- `calculate_quote_tool` — computes line-item pricing with bulk discount tiers (5% for 100-499 units, 10% for 500-999, 15% for 1000+)
- `search_quotes_tool` (wraps `search_quote_history()`) — searches past quotes for pricing consistency

**Sales Agent** — A `ToolCallingAgent` responsible for finalizing transactions and managing cash flow. It has access to:
- `finalize_sale_tool` (wraps `create_transaction()` for sales) — records sale transactions
- `reorder_stock_tool` (wraps `create_transaction()` for stock orders + `get_supplier_delivery_date()`) — places supplier reorders
- `check_delivery_tool` (wraps `get_supplier_delivery_date()`) — estimates delivery timelines
- `check_balance_tool` (wraps `get_cash_balance()`) — monitors cash balance

**Business Advisor Agent** — A `ToolCallingAgent` responsible for financial analysis and strategic business health assessment. It operates in two modes:
- *Proactive mid-session analysis* — Every 5 requests, the orchestrator triggers the advisor to assess operational patterns: trending demand, cash flow direction, inventory depletion rates, and strategic recommendations. This allows the system to surface concerns before they become critical.
- *Final post-batch assessment* — After all 20 requests are processed, the advisor generates a comprehensive financial health report summarizing cash position, inventory valuation, total assets, and top-selling products.
- Tools: `financial_report_tool` (wraps `generate_financial_report()`) and `check_balance_tool` (wraps `get_cash_balance()`)

**Customer Agent** (external, stand-out feature) — A `ToolCallingAgent` that simulates a customer receiving and evaluating each quote. After the orchestrator composes a response, the Customer Agent reviews it from the customer's perspective — considering the quoted prices, any partial fulfillments, unavailable items, and delivery timelines. It provides naturalistic feedback that could inform future system improvements (e.g., "the delivery time is longer than expected" or "thank you for the bulk discount"). This agent operates outside the 4-agent worker limit since it simulates an external party rather than performing internal business logic.

### Decision-Making Process for the Architecture

The architecture was designed around three key principles:

1. **Reliability over complexity**: The orchestrator handles item parsing and matching deterministically using regex and multi-phase string matching. This ensures consistent behavior across all 20 test requests, unlike purely LLM-based parsing which could produce inconsistent results.

2. **Agent specialization**: Each worker agent has a clearly scoped responsibility and a curated set of tools. The Inventory Agent never records transactions; the Sales Agent never matches items. This separation prevents tool misuse and makes the system easier to debug.

3. **Graceful degradation**: Agent `.run()` calls are wrapped in try/except blocks. If the Sales Agent fails to record a transaction, the orchestrator falls back to direct `create_transaction()` calls. This ensures that business-critical operations (recording sales, reordering stock) always complete.

### Workflow Per Request (see workflow_diagram.png)

1. **Parse**: The orchestrator extracts item names and quantities from the natural language request using regex-based parsing
2. **Match**: Each extracted item is matched to the product catalog via multi-phase matching (synonyms, exact match, substring, word overlap)
3. **Check Stock**: For matched items, the Inventory Agent verifies current stock levels via `check_stock_tool` (wrapping `get_stock_level()`)
4. **Classify**: Items are categorized as fulfillable, partially fulfillable (offer available stock), or unavailable
5. **Quote**: The Quote Agent searches historical quotes for pricing context; the orchestrator calculates line-item pricing with bulk discounts
6. **Finalize**: The Sales Agent records each sale transaction via `finalize_sale_tool`
7. **Reorder**: After sales, the orchestrator checks remaining stock against minimum thresholds and delegates reorders to the Sales Agent via `reorder_stock_tool` if cash reserves allow
8. **Respond**: A customer-facing response is composed with item breakdown, pricing rationale, and delivery estimate — without exposing internal data
9. **Customer Evaluation**: The Customer Agent reviews the quote from the customer's perspective, providing feedback on pricing, availability, and delivery
10. **Mid-Session Advice** (every 5 requests): The Business Advisor Agent analyzes trends and provides proactive operational guidance
11. **Final Assessment**: After all requests are processed, the Business Advisor generates a comprehensive financial health report

### Terminal Visualization

The system includes a rich terminal display layer designed for operator observability:

- **Workflow Pipeline**: Each request displays a real-time stage progression bar: `PARSE ✓ → INVENTORY ⟳ → QUOTE ○ → SALES ○ → REORDER ○ → RESPOND ○` — showing completed (✓), active (⟳), and pending (○) stages
- **Progress Bar**: A Unicode block-character progress bar tracks overall progress across all 20 requests: `████████░░░░░░░░ 8/20 (40%)`
- **Threaded Spinner**: Long-running agent API calls display a Braille-character spinner animation (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) that runs in a background thread, replaced by the result when the call completes
- **Adaptive Color**: The display system detects terminal capabilities (256-color, basic 8-color, or no-color) and gracefully degrades. It respects the `NO_COLOR` environment variable and detects non-TTY output (piped/redirected) to disable animations while preserving static status lines

## 2. Evaluation Results

### Summary Statistics

| Metric | Result | Rubric Requirement |
|--------|--------|--------------------|
| Total requests processed | 20 | 20 (full dataset) |
| Fulfilled requests (with quote) | 17 | At least 3 |
| Unfulfilled requests | 3 | At least 1 |
| Cash balance changes | 16 | At least 3 |
| Initial cash balance | $45,121.20 | — |
| Final cash balance | $45,535.70 | — |
| Initial inventory value | $4,940.30 | — |
| Final inventory value | $4,384.20 | — |

### Fulfilled Requests

17 out of 20 requests resulted in successful quotes and transactions. These requests involved items that were both in the product catalog and had sufficient stock (or had enough stock for partial fulfillment). Common successfully fulfilled items included A4 paper, Cardstock, Colored paper, Glossy paper, and Kraft paper.

Several requests involved partial fulfillment — the system offered the available quantity when stock was insufficient for the full request. For example, when a customer requested 2000 sheets of glossy paper but only 441 were in stock, the system quoted and sold the 441 available units with a clear note in the response.

### Unfulfilled Requests

3 requests could not be fulfilled:

- **Request 2** (hotel manager, parade): Requested colorful poster paper, streamers, and balloons. Poster paper was out of stock (not in the seed-137 inventory), Party streamers were not stocked, and balloons are not in the product catalog at all.
- **Request 3** (school board, conference): Requested 10,000 sheets of A4 paper, 5,000 sheets of A3 paper, and 500 reams of printer paper. While A4 paper is in stock, the requested 10,000 units far exceeded the available ~748 units. A3 paper is not in the catalog, and "printer paper" maps to Standard copy paper which is also not stocked. (Note: Request 3 was partially fulfilled in some runs when enough A4 stock was available.)
- **Request 20** (restaurant manager, concert): Requested 5,000 flyers, 2,000 posters, and 10,000 tickets. Flyers are in the catalog but not stocked in inventory, posters and tickets are not catalog items.

These failures naturally arose from the inventory model (only 40% of catalog items are stocked with seed=137) and demonstrate the system's ability to handle edge cases gracefully.

### Strengths

1. **Robust item matching**: The multi-phase matching strategy (synonym lookup, exact match, substring containment, word-overlap scoring with paper-size guards) correctly handled diverse customer phrasings. "A4 glossy paper" matched to "Glossy paper," "heavy cardstock (white)" matched to "Cardstock," and "poster boards (24x36)" matched to "Large poster paper (24x36 inches)."

2. **Transparent pricing**: Every response includes a line-by-line breakdown showing the item name, quantity, unit price, applicable discount rate, and line total. When bulk discounts are applied, the rationale is stated explicitly.

3. **Inventory safeguards**: The automatic reorder system maintains healthy stock levels. After each sale, the system checks remaining stock against minimum thresholds and places supplier orders when cash reserves allow (requiring cash > 2x reorder cost to maintain a safety margin).

4. **Clean customer communication**: Responses never expose internal system details (database structure, agent names, cash balance, exact profit margins). Unavailable items are explained in customer-friendly terms ("not in our product catalog" vs "currently out of stock").

5. **Partial fulfillment handling**: Rather than rejecting an entire order because one item has insufficient stock, the system offers what it can and clearly notes the adjustment.

6. **Proactive business intelligence**: The Business Advisor Agent doesn't just run at the end — it provides mid-session analysis every 5 requests, surfacing trends in demand, cash flow direction, and inventory depletion before they become problems.

7. **Customer feedback loop**: The Customer Agent provides a realistic external perspective on each quote, simulating how a real customer would react to pricing, partial fulfillment, and delivery timelines.

## 3. Suggestions for Improvement

### Improvement 1: Semantic Matching with Embeddings

The current item matching uses keyword-based heuristics (substring containment, word overlap). This fails for semantic equivalences that don't share words — for example, "poster board" could be improved to match "Large poster paper (24x36 inches)" more reliably using embedding-based similarity. Implementing a vector search (e.g., using TF-IDF or sentence embeddings from a model like `all-MiniLM-L6-v2`) over the catalog would improve match accuracy, especially for items described with industry-specific terminology.

### Improvement 2: Multi-Turn Customer Negotiation

The current Customer Agent provides one-way feedback after receiving a quote. A more advanced version could engage in multi-turn negotiation: when stock is insufficient, the agent could request alternative products, ask about restock timelines, or negotiate volume discounts. This would require maintaining conversation state across turns and implementing a negotiation strategy, but would significantly improve customer experience and conversion rates.

### Improvement 3: Demand Forecasting and Proactive Restocking

The current reorder logic is reactive — it only triggers after stock drops below a threshold. The Business Advisor Agent could be extended to analyze order patterns across all processed requests to:
- Identify trending items that are depleting faster than expected
- Proactively recommend restocking popular items before they run out
- Suggest adjusting minimum stock levels based on observed demand
- Flag items that haven't sold and recommend reducing stock levels to free up cash

This would transform the system from reactive inventory management to proactive supply chain optimization, potentially preventing the stock-outs that caused 3 requests to go unfulfilled.

## 4. Submission Artifacts

| File | Description |
|------|-------------|
| `project_starter.py` | Complete multi-agent implementation (single Python file) |
| `workflow_diagram.png` | Visual diagram of agent architecture and data flow |
| `workflow_diagram.dot` | Source Graphviz diagram file |
| `test_results.csv` | Evaluation results for all 20 sample requests |
| `reflection_report.md` | This reflection document |
| `quote_requests_sample.csv` | Input test dataset (provided) |
| `quotes.csv` | Historical quote data (provided) |
| `quote_requests.csv` | Full quote requests data (provided) |
| `requirements.txt` | Python package dependencies |
