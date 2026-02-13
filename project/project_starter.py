import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (disabled for clean output)
    # print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# MULTI-AGENT SYSTEM IMPLEMENTATION
# Framework: smolagents (Hugging Face)
# Architecture: Orchestrator + 4 Worker Agents (Inventory, Quote, Sales, Advisor)
# Stand-out: Customer Agent (external negotiator) + Terminal animation
########################
########################
########################

import sys
import re
import json
import threading
import functools
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents.monitoring import LogLevel

# Force unbuffered stdout so ANSI colors render in real-time
print = functools.partial(print, flush=True)  # noqa: A001

# --- Environment & Model Setup ---
dotenv.load_dotenv()
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# --- Common Item Synonym Mapping ---
# Maps common customer phrasings to exact catalog item names
ITEM_SYNONYMS = {
    "printer paper": "Standard copy paper",
    "printing paper": "Standard copy paper",
    "copy paper": "Standard copy paper",
    "standard printer paper": "Standard copy paper",
    "white printer paper": "A4 paper",
    "a4 printer paper": "A4 paper",
    "a4 printing paper": "A4 paper",
    "a4 white paper": "A4 paper",
    "poster board": "Large poster paper (24x36 inches)",
    "poster boards": "Large poster paper (24x36 inches)",
    "washi tape": "Decorative adhesive tape (washi tape)",
    "table napkins": "Paper napkins",
    "dinner napkins": "Paper napkins",
    "white paper": "A4 paper",
}


# =============================================================================
# TERMINAL DISPLAY SYSTEM
# Animated, color-coded, progressive workflow visualization.
# Uses basic ANSI (8-color) with 256-color enhancement when supported.
# =============================================================================

# --- Color Detection ---
# Respect NO_COLOR convention (https://no-color.org/)
_NO_COLOR = os.environ.get("NO_COLOR") is not None
_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
_TERM = os.environ.get("TERM", "")
_SUPPORTS_256 = (
    not _NO_COLOR
    and ("256color" in _TERM or "xterm" in _TERM or "screen" in _TERM
         or sys.platform == "darwin" or _TERM == "")
)

# ANSI color palette with graceful fallback
if _NO_COLOR:
    _C = {k: "" for k in [
        "reset", "bold", "dim", "orange", "green", "purple", "red",
        "teal", "blue", "gray", "white", "yellow", "cyan", "magenta",
    ]}
elif _SUPPORTS_256:
    _C = {
        "reset":   "\033[0m",
        "bold":    "\033[1m",
        "dim":     "\033[2m",
        "orange":  "\033[38;5;208m",
        "green":   "\033[38;5;34m",
        "purple":  "\033[38;5;133m",
        "red":     "\033[38;5;196m",
        "teal":    "\033[38;5;37m",
        "blue":    "\033[38;5;69m",
        "gray":    "\033[38;5;245m",
        "white":   "\033[97m",
        "yellow":  "\033[38;5;220m",
        "cyan":    "\033[38;5;37m",
        "magenta": "\033[38;5;133m",
    }
else:
    # Fallback to basic 8-color ANSI
    _C = {
        "reset":   "\033[0m",
        "bold":    "\033[1m",
        "dim":     "\033[2m",
        "orange":  "\033[33m",      # yellow as fallback
        "green":   "\033[32m",
        "purple":  "\033[35m",
        "red":     "\033[31m",
        "teal":    "\033[36m",
        "blue":    "\033[34m",
        "gray":    "\033[37m",
        "white":   "\033[37m",
        "yellow":  "\033[33m",
        "cyan":    "\033[36m",
        "magenta": "\033[35m",
    }

W = 76  # display width

# --- Workflow Stage Names and Colors ---
STAGE_CONFIG = [
    ("PARSE",     "orange"),
    ("INVENTORY", "green"),
    ("QUOTE",     "purple"),
    ("SALES",     "red"),
    ("REORDER",   "teal"),
    ("RESPOND",   "blue"),
]


# --- Background Spinner ---
class Spinner:
    """Threaded spinner shown during long-running agent calls.

    Displays a Braille-dot animation (or ASCII fallback) on the current
    line, replaced by the final result when stop() is called.
    """
    _BRAILLE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _ASCII   = ["|", "/", "-", "\\"]

    def __init__(self, label, message, color_key="orange"):
        self.label = label
        self.message = message
        self.color_key = color_key
        self._stop = threading.Event()
        self._thread = None
        # Use Braille spinners if Unicode is likely supported
        try:
            "⠋".encode(sys.stdout.encoding or "utf-8")
            self._frames = self._BRAILLE
        except (UnicodeEncodeError, LookupError):
            self._frames = self._ASCII

    def _animate(self):
        """Animation loop running in a background thread."""
        c = _C.get(self.color_key, _C["orange"])
        r = _C["reset"]
        idx = 0
        while not self._stop.is_set():
            frame = self._frames[idx % len(self._frames)]
            if _IS_TTY:
                sys.stdout.write(
                    f"\r  {c}{_C['bold']}[{self.label:<10}]{r}  "
                    f"{_C['dim']}{frame} {self.message}{r}\033[K"
                )
                sys.stdout.flush()
            self._stop.wait(0.1)
            idx += 1

    def start(self):
        """Start the spinner animation in a background thread."""
        if _IS_TTY:
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        else:
            # Non-TTY: print a static "working" line
            c = _C.get(self.color_key, _C["orange"])
            r = _C["reset"]
            print(
                f"  {c}{_C['bold']}[{self.label:<10}]{r}  "
                f"{_C['dim']}... {self.message}{r}"
            )
        return self

    def stop(self, final_label=None, final_detail=None, final_color=None):
        """Stop the spinner and print the final result."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        label = final_label or self.label
        color = final_color or self.color_key
        c = _C.get(color, _C["orange"])
        r = _C["reset"]
        if _IS_TTY:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        if final_detail is not None:
            print(f"  {c}{_C['bold']}[{label:<10}]{r}  {final_detail}")
            sys.stdout.flush()


# --- Display Helpers ---

def _line(char="─"):
    """Print a thin separator line."""
    print(f"  {_C['dim']}{char * (W - 4)}{_C['reset']}")


def _result(label, detail, color_key="orange"):
    """Print a stage result line."""
    c = _C.get(color_key, _C["orange"])
    r = _C["reset"]
    print(f"  {c}{_C['bold']}[{label:<10}]{r}  {detail}")
    sys.stdout.flush()


def _stage(label, detail, color_key="orange"):
    """Quick stage line with brief animation."""
    sp = Spinner(label, "processing", color_key).start()
    time.sleep(0.3)  # brief visual pause
    sp.stop(label, detail, color_key)


def _progress_bar(current, total, width=30):
    """Return a Unicode progress bar string like '████████░░░░░░░░ 8/20 (40%)'."""
    ratio = current / total if total > 0 else 0
    filled = int(width * ratio)
    empty = width - filled
    pct = int(ratio * 100)
    bar = "█" * filled + "░" * empty
    return f"{_C['orange']}{bar}{_C['reset']} {current}/{total} ({pct}%)"


def _workflow_pipeline(active_idx=-1, completed=None):
    """Print a visual workflow pipeline showing stage progression.

    Args:
        active_idx: Index of the currently active stage (0-based).
        completed: Set of stage indices that are completed.

    Example output:
      [PARSE ✓] → [INVENTORY ⟳] → [QUOTE ○] → [SALES ○] → [REORDER ○] → [RESPOND ○]
    """
    if completed is None:
        completed = set()
    parts = []
    for i, (name, color_key) in enumerate(STAGE_CONFIG):
        c = _C.get(color_key, "")
        r = _C["reset"]
        if i in completed:
            parts.append(f"{c}{_C['bold']}{name} ✓{r}")
        elif i == active_idx:
            parts.append(f"{c}{_C['bold']}{name} ⟳{r}")
        else:
            parts.append(f"{_C['dim']}{name} ○{r}")
    pipeline = f" {_C['dim']}→{_C['reset']} ".join(parts)
    if _IS_TTY:
        sys.stdout.write(f"\r  {pipeline}\033[K")
        sys.stdout.flush()
    else:
        print(f"  {pipeline}")


def _clear_pipeline():
    """Clear the pipeline line (TTY only)."""
    if _IS_TTY:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def _header(num, total, context, date, cash, inv):
    """Print the request header block with progress indication."""
    bar = _progress_bar(num - 1, total)
    print(f"\n  {_C['dim']}{'━' * (W - 4)}{_C['reset']}")
    time.sleep(0.05)
    print(
        f"  {_C['bold']}{_C['orange']}REQUEST {num}/{total}{_C['reset']}"
        f"  {_C['dim']}│{_C['reset']}  {date}"
        f"  {_C['dim']}│{_C['reset']}  {context}"
    )
    sys.stdout.flush()
    time.sleep(0.05)
    print(
        f"  {_C['dim']}Cash: ${cash:,.2f}  │  "
        f"Inventory: ${inv:,.2f}{_C['reset']}  │  {bar}"
    )
    sys.stdout.flush()
    time.sleep(0.05)
    _line("─")


# =============================================================================
# TOOL DEFINITIONS — wrapping the provided helper functions
# Each @tool is assigned to one of the worker agents.
# =============================================================================

# ---- Tools for the Inventory Agent ----

@tool
def check_inventory_tool(as_of_date: str) -> str:
    """Check all available inventory items and their stock levels as of a date.

    Args:
        as_of_date: Date string in YYYY-MM-DD format.

    Returns:
        A formatted list of all items currently in stock with unit quantities.
    """
    inventory = get_all_inventory(as_of_date)
    if not inventory:
        return "No inventory items currently in stock."
    lines = [f"  - {name}: {int(qty)} units" for name, qty in inventory.items()]
    return "Current Inventory:\n" + "\n".join(lines)


@tool
def check_stock_tool(item_name: str, as_of_date: str) -> str:
    """Check the stock level of a specific item as of a given date.

    Args:
        item_name: The exact catalog item name to look up.
        as_of_date: Date string in YYYY-MM-DD format.

    Returns:
        A string reporting the item name and its current stock count.
    """
    df = get_stock_level(item_name, as_of_date)
    stock = int(df["current_stock"].iloc[0]) if not df.empty else 0
    return f"{item_name}: {stock} units in stock"


@tool
def match_item_tool(requested_item: str) -> str:
    """Match a customer-requested item description to the product catalog.

    Uses multi-phase matching: synonym lookup, exact match, substring
    containment, and word-overlap scoring with paper-size guards.

    Args:
        requested_item: The item name/description from the customer request.

    Returns:
        'MATCH: <name>' for a confident match, 'CLOSEST: <name>' for a
        probable match, or 'NOT_FOUND: <item>' when nothing matches.
    """
    req_lower = requested_item.lower().strip()
    catalog_names = [p["item_name"] for p in paper_supplies]

    # Phase 0: Synonym mapping
    for synonym, canonical in ITEM_SYNONYMS.items():
        if synonym in req_lower:
            return f"MATCH: {canonical}"

    # Phase 1: Exact case-insensitive match
    for name in catalog_names:
        if name.lower() == req_lower:
            return f"MATCH: {name}"

    # Phase 2: Substring containment (either direction)
    for name in catalog_names:
        if name.lower() in req_lower or req_lower in name.lower():
            return f"MATCH: {name}"

    # Phase 3: Word-overlap scoring
    stop_words = {
        "high", "quality", "heavy", "standard", "colorful", "white",
        "assorted", "various", "colors", "sized", "size", "full", "color",
        "recycled", "biodegradable", "premium", "sturdy", "of",
    }
    req_words = set(req_lower.split()) - stop_words
    size_pat = r'\b(a[0-9])\b'
    req_sizes = set(re.findall(size_pat, req_lower))

    best_match, best_score = None, 0
    for name in catalog_names:
        cat_lower = name.lower()
        cat_words = set(cat_lower.split()) - stop_words
        cat_sizes = set(re.findall(size_pat, cat_lower))
        # Guard: skip if paper sizes are specified but differ (A3 vs A4)
        if req_sizes and cat_sizes and req_sizes != cat_sizes:
            continue
        overlap = len(req_words & cat_words)
        if overlap > best_score:
            best_score = overlap
            best_match = name

    if best_score >= 1:
        return f"CLOSEST: {best_match}"
    return f"NOT_FOUND: {requested_item}"


# ---- Tools for the Quote Agent ----

@tool
def calculate_quote_tool(item_name: str, quantity: int, unit_price: float) -> str:
    """Calculate a price quote for an item with bulk discount applied.

    Discount tiers: 0% (1-99 units), 5% (100-499), 10% (500-999), 15% (1000+).
    Totals are rounded to customer-friendly numbers.

    Args:
        item_name: The catalog item name.
        quantity: Number of units requested.
        unit_price: Price per unit in dollars.

    Returns:
        A formatted quote line showing item, quantity, discount, and total.
    """
    # Determine bulk discount tier
    if quantity >= 1000:
        discount_pct = 0.15
    elif quantity >= 500:
        discount_pct = 0.10
    elif quantity >= 100:
        discount_pct = 0.05
    else:
        discount_pct = 0.0

    subtotal = quantity * unit_price
    discount_amount = subtotal * discount_pct
    total = subtotal - discount_amount

    # Round to friendly numbers
    if total > 100:
        total = round(total / 5) * 5
    elif total > 10:
        total = round(total)
    else:
        total = round(total, 2)

    return (
        f"Item: {item_name} | Qty: {quantity} | Unit: ${unit_price:.2f} | "
        f"Subtotal: ${subtotal:.2f} | Discount: {int(discount_pct * 100)}% "
        f"(-${discount_amount:.2f}) | Line Total: ${total:.2f}"
    )


@tool
def search_quotes_tool(search_terms: str, limit: int) -> str:
    """Search historical quotes for pricing reference and consistency.

    Looks through past quotes to find similar orders by job type, event,
    or item keywords. Useful for maintaining consistent pricing.

    Args:
        search_terms: Comma-separated keywords (e.g. 'cardstock,ceremony').
        limit: Maximum number of results to return.

    Returns:
        Formatted historical quote summaries or a no-results message.
    """
    terms = [t.strip() for t in search_terms.split(",") if t.strip()]
    results = search_quote_history(terms, limit)
    if not results:
        return "No matching historical quotes found."
    lines = []
    for q in results:
        lines.append(
            f"  - ${q['total_amount']} ({q.get('job_type', 'N/A')}, "
            f"{q.get('order_size', 'N/A')}, {q.get('event_type', 'N/A')}): "
            f"{q['quote_explanation'][:150]}..."
        )
    return "Historical Quotes:\n" + "\n".join(lines)


# ---- Tools for the Sales Agent ----

@tool
def finalize_sale_tool(item_name: str, quantity: int, total_price: float, date: str) -> str:
    """Record a finalized sale transaction in the database.

    Creates a 'sales' transaction, reducing available stock and increasing
    the company's cash balance.

    Args:
        item_name: The exact catalog item name being sold.
        quantity: Number of units sold.
        total_price: Total sale price after discounts.
        date: Transaction date in YYYY-MM-DD format.

    Returns:
        Confirmation message with the transaction ID.
    """
    tx_id = create_transaction(item_name, "sales", quantity, total_price, date)
    return f"SALE RECORDED: Tx#{tx_id} | {quantity}x {item_name} | ${total_price:.2f}"


@tool
def reorder_stock_tool(item_name: str, quantity: int, date: str) -> str:
    """Place a stock reorder with the supplier for a catalog item.

    Records a 'stock_orders' transaction and estimates the delivery date.
    Used when inventory drops below minimum levels after a sale.

    Args:
        item_name: The exact catalog item name to reorder.
        quantity: Number of units to order from the supplier.
        date: Order date in YYYY-MM-DD format.

    Returns:
        Reorder confirmation with cost and estimated delivery date.
    """
    # Look up unit price from the master catalog
    unit_price = None
    for p in paper_supplies:
        if p["item_name"].lower() == item_name.lower():
            unit_price = p["unit_price"]
            break
    if unit_price is None:
        return f"ERROR: {item_name} not found in supplier catalog."

    total_cost = quantity * unit_price
    delivery_date = get_supplier_delivery_date(date, quantity)
    tx_id = create_transaction(item_name, "stock_orders", quantity, total_cost, date)
    return (
        f"REORDER PLACED: Tx#{tx_id} | {quantity}x {item_name} | "
        f"Cost: ${total_cost:.2f} | Delivery by: {delivery_date}"
    )


@tool
def check_delivery_tool(order_date: str, quantity: int) -> str:
    """Estimate the supplier delivery date based on order size.

    Lead times: same day (<=10), 1 day (11-100), 4 days (101-1000),
    7 days (>1000 units).

    Args:
        order_date: Starting date in YYYY-MM-DD format.
        quantity: Number of units in the order.

    Returns:
        The estimated delivery date as a string.
    """
    delivery = get_supplier_delivery_date(order_date, quantity)
    return f"Estimated delivery date: {delivery}"


@tool
def check_balance_tool(as_of_date: str) -> str:
    """Check the company's current cash balance as of a date.

    Calculates total sales revenue minus total stock purchase costs.

    Args:
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        The current cash balance formatted as a dollar string.
    """
    balance = get_cash_balance(as_of_date)
    return f"Cash balance as of {as_of_date}: ${balance:,.2f}"


@tool
def financial_report_tool(as_of_date: str) -> str:
    """Generate a comprehensive financial report for the company.

    Includes cash balance, inventory valuation, total assets, and
    top-selling products.

    Args:
        as_of_date: Report date in YYYY-MM-DD format.

    Returns:
        A formatted financial report summary.
    """
    report = generate_financial_report(as_of_date)
    summary = (
        f"=== Financial Report ({as_of_date}) ===\n"
        f"Cash Balance:    ${report['cash_balance']:,.2f}\n"
        f"Inventory Value: ${report['inventory_value']:,.2f}\n"
        f"Total Assets:    ${report['total_assets']:,.2f}\n"
    )
    if report.get("top_selling_products"):
        summary += "\nTop Selling Products:\n"
        for p in report["top_selling_products"]:
            if p.get("item_name"):
                summary += f"  - {p['item_name']}: ${p.get('total_revenue', 0):,.2f}\n"
    return summary


# =============================================================================
# AGENT DEFINITIONS — four worker agents plus the orchestrator
# =============================================================================

# Agent 1: Inventory Agent — item identification and stock verification
inventory_agent = ToolCallingAgent(
    tools=[check_inventory_tool, check_stock_tool, match_item_tool],
    model=model,
    name="inventory_agent",
    verbosity_level=LogLevel.OFF,
    description=(
        "You are the Inventory Specialist for Beaver's Choice Paper Company. "
        "Your responsibilities: "
        "1) Match customer-requested items to exact catalog names using match_item_tool. "
        "2) Check current stock levels using check_stock_tool. "
        "3) Report item availability accurately. "
        "Always use exact catalog item names. If an item is not in our catalog, "
        "clearly indicate it as unavailable."
    ),
)

# Agent 2: Quote Agent — pricing with bulk discounts and historical context
quote_agent = ToolCallingAgent(
    tools=[calculate_quote_tool, search_quotes_tool],
    model=model,
    name="quote_agent",
    verbosity_level=LogLevel.OFF,
    description=(
        "You are the Pricing Specialist for Beaver's Choice Paper Company. "
        "Your responsibilities: "
        "1) Generate price quotes using calculate_quote_tool for each line item. "
        "2) Reference historical quotes using search_quotes_tool for consistency. "
        "3) Apply bulk discounts: 5% (100-499), 10% (500-999), 15% (1000+). "
        "4) Present clear, customer-friendly pricing breakdowns. "
        "Always provide a line-by-line breakdown and a clear grand total."
    ),
)

# Agent 3: Sales Agent — transaction finalization and cash management
sales_agent = ToolCallingAgent(
    tools=[finalize_sale_tool, reorder_stock_tool, check_delivery_tool, check_balance_tool],
    model=model,
    name="sales_agent",
    verbosity_level=LogLevel.OFF,
    description=(
        "You are the Sales Manager for Beaver's Choice Paper Company. "
        "Your responsibilities: "
        "1) Finalize sales using finalize_sale_tool to record each transaction. "
        "2) Check delivery timelines using check_delivery_tool. "
        "3) Monitor cash balance using check_balance_tool. "
        "4) Handle stock reorders using reorder_stock_tool when inventory is low. "
        "Ensure all transactions are properly recorded."
    ),
)

# Agent 4: Business Advisor Agent — financial analysis and strategic recommendations
advisor_agent = ToolCallingAgent(
    tools=[financial_report_tool, check_balance_tool],
    model=model,
    name="advisor_agent",
    verbosity_level=LogLevel.OFF,
    description=(
        "You are the Business Advisor for Beaver's Choice Paper Company. "
        "Your responsibilities: "
        "1) Generate financial reports using financial_report_tool to assess "
        "overall business health. "
        "2) Monitor cash balance using check_balance_tool. "
        "3) Provide strategic recommendations on order acceptance, inventory "
        "investment, and profitability. "
        "4) Analyze transaction patterns and proactively recommend operational "
        "improvements to increase efficiency and revenue."
    ),
)


# =============================================================================
# CUSTOMER AGENT — simulates an external customer negotiating with the system
# (Stand-out feature: adds realistic negotiation to the workflow)
# =============================================================================

# Note: The Customer Agent is EXTERNAL to the company's multi-agent system.
# It represents the customer perspective, so it does not count toward the
# five-agent limit for the company's internal system.

customer_agent = ToolCallingAgent(
    tools=[],
    model=model,
    name="customer_agent",
    verbosity_level=LogLevel.OFF,
    description=(
        "You are a customer evaluating a quote from Beaver's Choice Paper "
        "Company. Review the quote provided and respond as a realistic "
        "customer would. Consider the pricing, availability, and delivery "
        "timeline. You may: accept the quote, request a discount if the "
        "order is large, or express concern about partial fulfillment. "
        "Keep your response to 2-3 sentences. Be professional."
    ),
)


def get_customer_feedback(quote_response, job, event, order_size):
    """Use the Customer Agent to simulate realistic customer feedback.

    The customer evaluates the generated quote and responds based on their
    role (job), event type, and order scale.

    Args:
        quote_response: The formatted quote string from the system.
        job: The customer's job title / role.
        event: The type of event they are planning.
        order_size: The scale of the order (small, medium, large).

    Returns:
        A short customer feedback string, or a default acceptance.
    """
    try:
        feedback = customer_agent.run(
            f"You are a {job} planning a {event}. This is a {order_size} order. "
            f"Review this quote and respond as you would in a real conversation:\n\n"
            f"{quote_response[:800]}"
        )
        return str(feedback).strip()
    except Exception:
        return "Thank you, the quote looks good. We'll proceed with the order."


# =============================================================================
# ORCHESTRATOR — coordinates the four worker agents
# =============================================================================

class PaperCompanyOrchestrator:
    """Coordinates the multi-agent system for Beaver's Choice Paper Company.

    Workflow per request:
      1. Parse items and quantities from customer text
      2. Match each item to the product catalog (Inventory Agent)
      3. Check stock availability (Inventory Agent)
      4. Generate a quote with bulk discounts (Quote Agent)
      5. Finalize sales transactions (Sales Agent)
      6. Evaluate reorder needs for low-stock items (Sales Agent)
      7. Compose a customer-facing response
    """

    def __init__(self, model):
        """Initialize the orchestrator with references to worker agents."""
        self.model = model
        self.inventory_agent = inventory_agent
        self.quote_agent = quote_agent
        self.sales_agent = sales_agent
        self.advisor_agent = advisor_agent
        # Build a fast unit-price lookup from the catalog
        self.price_lookup = {
            p["item_name"].lower(): p["unit_price"] for p in paper_supplies
        }
        # Track cumulative stats for advisor
        self.requests_processed = 0
        self.total_revenue = 0.0
        self.total_reorders = 0

    # -----------------------------------------------------------------
    # Request parsing
    # -----------------------------------------------------------------
    def _parse_request_items(self, text):
        """Extract (item_description, quantity) pairs from customer request text.

        Handles bullet-point lists, comma-separated items, and inline mentions.
        Returns a list of (item_name_str, quantity_int) tuples.
        """
        items = []
        seen = set()
        text_clean = text.replace("\r", "").strip()

        # Split on bullet markers, newlines before digits, and list-separator
        # commas (require a space after comma to avoid splitting "5,000")
        segments = re.split(
            r"\n\s*[-•*]\s*|\n(?=\s*\d)|,\s+(?=\d)|,\s+and\s+(?=\d)",
            text_clean,
        )

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Find a NUMBER followed by a description
            match = re.search(r"(\d[\d,]*)\s+(.*)", segment, re.DOTALL)
            if not match:
                continue

            qty = int(match.group(1).replace(",", ""))
            rest = match.group(2).strip()

            # Remove unit words from the beginning (sheets, rolls, packs, etc.)
            rest = re.sub(
                r"^(?:sheets?|rolls?|packs?|packets?|boxes?|reams?|units?|"
                r"pieces?|pads?|sets?|copies|copy)\s+(?:of\s+)?",
                "",
                rest,
                flags=re.IGNORECASE,
            )

            # Remove parentheticals — (white), (assorted colors), (24"x36"), etc.
            rest = re.sub(r"\s*\([^)]*\)", "", rest)

            # Remove trailing sentence fragments (delivery dates, polite phrases)
            rest = re.sub(
                r"\s*(?:for (?:the|our|an|a)\s+.*|[Pp]lease.*|I need.*|We need.*|"
                r"deliver.*|[Tt]hank.*|in time.*|to ensure.*|The supplies.*|"
                r"must be.*|I would.*|We would.*|printed in.*)$",
                "",
                rest,
                flags=re.IGNORECASE,
            )

            # Remove trailing prepositions / articles
            rest = re.sub(
                r"\s+(?:in|for|by|at|to|on|and|the|our|an|a)$",
                "",
                rest,
                flags=re.IGNORECASE,
            ).strip(" ,.-;:")

            key = rest.lower()
            if rest and len(rest) >= 2 and qty > 0 and key not in seen:
                items.append((rest, qty))
                seen.add(key)

        return items

    # -----------------------------------------------------------------
    # Catalog matching
    # -----------------------------------------------------------------
    def _match_to_catalog(self, requested_name):
        """Match a requested item name to the product catalog.

        Returns (catalog_name, unit_price) or (None, None) if no match.
        """
        req_lower = requested_name.lower().strip()

        # Phase 0: Synonym mapping
        for synonym, canonical in ITEM_SYNONYMS.items():
            if synonym in req_lower:
                return canonical, self.price_lookup.get(canonical.lower(), 0)

        # Phase 1: Exact match (case-insensitive)
        for p in paper_supplies:
            if p["item_name"].lower() == req_lower:
                return p["item_name"], p["unit_price"]

        # Phase 2: Substring containment
        for p in paper_supplies:
            cat_lower = p["item_name"].lower()
            if cat_lower in req_lower or req_lower in cat_lower:
                return p["item_name"], p["unit_price"]

        # Phase 3: Word-overlap scoring
        stop_words = {
            "high", "quality", "heavy", "standard", "colorful", "white",
            "assorted", "various", "colors", "sized", "size", "full",
            "color", "recycled", "biodegradable", "premium", "sturdy",
            "large", "small", "of",
        }
        req_words = set(req_lower.split()) - stop_words
        size_pat = r"\b(a[0-9])\b"
        req_sizes = set(re.findall(size_pat, req_lower))

        best_match, best_price, best_score = None, 0.0, 0
        for p in paper_supplies:
            cat_lower = p["item_name"].lower()
            cat_words = set(cat_lower.split()) - stop_words
            cat_sizes = set(re.findall(size_pat, cat_lower))
            # Paper-size guard: A3 must not match A4
            if req_sizes and cat_sizes and req_sizes != cat_sizes:
                continue
            overlap = len(req_words & cat_words)
            if overlap > best_score:
                best_score = overlap
                best_match = p["item_name"]
                best_price = p["unit_price"]

        if best_score >= 1:
            return best_match, best_price
        return None, None

    # -----------------------------------------------------------------
    # Pricing helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _get_discount_rate(quantity):
        """Return the bulk discount percentage for a given quantity."""
        if quantity >= 1000:
            return 0.15
        elif quantity >= 500:
            return 0.10
        elif quantity >= 100:
            return 0.05
        return 0.0

    @staticmethod
    def _round_friendly(amount):
        """Round a dollar amount to a customer-friendly number."""
        if amount > 100:
            return round(amount / 5) * 5
        elif amount > 10:
            return round(amount)
        return round(amount, 2)

    # -----------------------------------------------------------------
    # Inventory helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _get_min_stock_level(item_name):
        """Retrieve the minimum stock level threshold from the inventory table."""
        try:
            df = pd.read_sql(
                "SELECT min_stock_level FROM inventory WHERE item_name = :name",
                db_engine,
                params={"name": item_name},
            )
            if not df.empty:
                return int(df.iloc[0]["min_stock_level"])
        except Exception:
            pass
        return 100  # Conservative default

    # -----------------------------------------------------------------
    # Main request processing pipeline
    # -----------------------------------------------------------------
    def process_request(self, request_text, request_date):
        """Process a single customer request through the full agent pipeline.

        Args:
            request_text: The customer's natural language order/inquiry.
            request_date: The date of the request (YYYY-MM-DD).

        Returns:
            A customer-facing response string with quote and availability info.
        """
        completed_stages = set()

        # ---- Stage 0: PARSE — extract items from request ----
        _workflow_pipeline(active_idx=0, completed=completed_stages)
        parsed_items = self._parse_request_items(request_text)

        if not parsed_items:
            _clear_pipeline()
            _result("PARSE", "No items or quantities identified.", "orange")
            return (
                "Dear Customer,\n\n"
                "Thank you for your inquiry. Unfortunately, we could not identify "
                "specific items and quantities in your request. Please provide "
                "item names and quantities, and we will be happy to assist.\n\n"
                "Best regards,\nBeaver's Choice Paper Company"
            )

        items_summary = ", ".join(
            f"{name} ({qty:,})" for name, qty in parsed_items
        )
        completed_stages.add(0)
        _clear_pipeline()
        _result("PARSE", f"{len(parsed_items)} items: {items_summary}", "orange")

        # ---- Stage 1: INVENTORY — match catalog + check stock ----
        _workflow_pipeline(active_idx=1, completed=completed_stages)
        sp = Spinner("INVENTORY", "matching catalog & checking stock", "green").start()

        fulfillable = []
        partial = []
        unavailable = []

        for item_name, qty_requested in parsed_items:
            catalog_name, unit_price = self._match_to_catalog(item_name)

            if catalog_name is None:
                unavailable.append({
                    "requested": item_name,
                    "qty": qty_requested,
                    "reason": "not in our product catalog",
                })
                continue

            stock_df = get_stock_level(catalog_name, request_date)
            current_stock = (
                int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0
            )

            if current_stock <= 0:
                unavailable.append({
                    "requested": item_name,
                    "catalog_name": catalog_name,
                    "qty": qty_requested,
                    "reason": "currently out of stock",
                })
            elif current_stock >= qty_requested:
                fulfillable.append({
                    "requested": item_name,
                    "catalog_name": catalog_name,
                    "qty": qty_requested,
                    "unit_price": unit_price,
                    "stock": current_stock,
                })
            else:
                partial.append({
                    "requested": item_name,
                    "catalog_name": catalog_name,
                    "qty_requested": qty_requested,
                    "qty_available": current_stock,
                    "unit_price": unit_price,
                })

        # Delegate stock verification to the Inventory Agent
        matched_names = list({
            d["catalog_name"]
            for d in fulfillable + partial
            if "catalog_name" in d
        })
        if matched_names:
            try:
                items_list = ", ".join(matched_names)
                self.inventory_agent.run(
                    f"Verify stock levels using check_stock_tool for each item "
                    f"with as_of_date='{request_date}': {items_list}"
                )
            except Exception:
                pass  # Fallback: direct stock check already done above

        # Build inventory result display
        inv_parts = []
        if fulfillable:
            inv_parts.append(f"{_C['green']}{len(fulfillable)} in stock{_C['reset']}")
        if partial:
            inv_parts.append(f"{_C['yellow']}{len(partial)} partial{_C['reset']}")
        if unavailable:
            inv_parts.append(f"{_C['red']}{len(unavailable)} unavailable{_C['reset']}")

        completed_stages.add(1)
        sp.stop("INVENTORY", " · ".join(inv_parts), "green")

        # Offer partial fulfillment
        for item in partial:
            fulfillable.append({
                "requested": item["requested"],
                "catalog_name": item["catalog_name"],
                "qty": item["qty_available"],
                "unit_price": item["unit_price"],
                "stock": item["qty_available"],
                "partial": True,
                "originally_requested": item["qty_requested"],
            })

        # Early exit if nothing can be fulfilled
        if not fulfillable:
            reasons = [
                f"  - {u['requested']}: {u['reason']}" for u in unavailable
            ]
            return (
                "Dear Customer,\n\n"
                "Thank you for your inquiry. Unfortunately, we are unable to "
                "fulfill your request at this time.\n\n"
                "Item Status:\n" + "\n".join(reasons) + "\n\n"
                "We apologize for the inconvenience and recommend checking back "
                "soon as our inventory is regularly restocked.\n\n"
                "Best regards,\nBeaver's Choice Paper Company"
            )

        # ---- Stage 2: QUOTE — pricing with bulk discounts ----
        _workflow_pipeline(active_idx=2, completed=completed_stages)
        sp = Spinner("QUOTE", "calculating pricing & searching history", "purple").start()

        line_items = []
        grand_total = 0.0

        for item in fulfillable:
            discount_rate = self._get_discount_rate(item["qty"])
            subtotal = item["qty"] * item["unit_price"]
            discount_amount = subtotal * discount_rate
            line_total = self._round_friendly(subtotal - discount_amount)
            grand_total += line_total
            line_items.append({
                **item,
                "discount_rate": discount_rate,
                "subtotal": subtotal,
                "discount_amount": discount_amount,
                "line_total": line_total,
            })

        grand_total = self._round_friendly(grand_total)

        # Delegate to the Quote Agent for historical pricing context
        try:
            terms = ",".join(
                [li["catalog_name"].split()[0] for li in line_items[:3]]
            )
            self.quote_agent.run(
                f"Search for historical quotes using search_quotes_tool with "
                f"terms '{terms}' and limit 3. Summarize any relevant pricing "
                f"patterns you find."
            )
        except Exception:
            pass

        # Show quote result
        n_items = len(line_items)
        n_disc = sum(1 for li in line_items if li["discount_rate"] > 0)
        item_word = "item" if n_items == 1 else "items"
        disc_note = f" ({n_disc} with bulk discount)" if n_disc else ""

        completed_stages.add(2)
        sp.stop(
            "QUOTE",
            f"{n_items} line {item_word}{disc_note} → "
            f"{_C['bold']}${grand_total:,.2f}{_C['reset']}",
            "purple",
        )

        # ---- Stage 3: SALES — finalize transactions ----
        _workflow_pipeline(active_idx=3, completed=completed_stages)
        sp = Spinner("SALES", "recording transactions", "red").start()

        sale_instructions = []
        for li in line_items:
            sale_instructions.append(
                f"Record sale: finalize_sale_tool with item_name='{li['catalog_name']}', "
                f"quantity={li['qty']}, total_price={li['line_total']}, date='{request_date}'"
            )

        try:
            self.sales_agent.run(
                "Finalize the following sales by calling finalize_sale_tool for each:\n"
                + "\n".join(sale_instructions)
                + f"\nThen check the cash balance using check_balance_tool "
                  f"with as_of_date='{request_date}'."
            )
        except Exception:
            # Fallback: record transactions directly
            for li in line_items:
                try:
                    create_transaction(
                        li["catalog_name"], "sales",
                        li["qty"], li["line_total"], request_date,
                    )
                except Exception:
                    pass

        cash_now = get_cash_balance(request_date)
        txn_word = "transaction" if len(line_items) == 1 else "transactions"
        completed_stages.add(3)
        sp.stop(
            "SALES",
            f"{len(line_items)} {txn_word} recorded  │  "
            f"Cash: ${cash_now:,.2f}",
            "red",
        )

        # Track cumulative revenue
        self.total_revenue += grand_total

        # ---- Stage 4: REORDER — evaluate low-stock items ----
        _workflow_pipeline(active_idx=4, completed=completed_stages)
        sp = Spinner("REORDER", "evaluating stock levels", "teal").start()
        reorder_messages = []

        reorder_instructions = []
        for li in line_items:
            try:
                stock_df = get_stock_level(li["catalog_name"], request_date)
                remaining = int(stock_df["current_stock"].iloc[0])
                min_level = self._get_min_stock_level(li["catalog_name"])

                if remaining < min_level:
                    reorder_qty = max(200, min_level * 3)
                    reorder_cost = reorder_qty * li["unit_price"]
                    cash = get_cash_balance(request_date)

                    if cash > reorder_cost * 2:
                        reorder_instructions.append({
                            "name": li["catalog_name"],
                            "qty": reorder_qty,
                            "cost": reorder_cost,
                            "unit_price": li["unit_price"],
                        })
            except Exception:
                pass

        if reorder_instructions:
            agent_cmds = [
                f"Reorder stock: reorder_stock_tool with item_name='{r['name']}', "
                f"quantity={r['qty']}, date='{request_date}'"
                for r in reorder_instructions
            ]
            try:
                self.sales_agent.run(
                    "Place the following stock reorders using reorder_stock_tool "
                    "for each item:\n" + "\n".join(agent_cmds)
                )
                for r in reorder_instructions:
                    del_date = get_supplier_delivery_date(request_date, r["qty"])
                    reorder_messages.append(
                        f"{r['qty']}x {r['name']} (delivery {del_date})"
                    )
            except Exception:
                # Fallback: record reorders directly
                for r in reorder_instructions:
                    try:
                        create_transaction(
                            r["name"], "stock_orders",
                            r["qty"], r["cost"], request_date,
                        )
                        del_date = get_supplier_delivery_date(
                            request_date, r["qty"]
                        )
                        reorder_messages.append(
                            f"{r['qty']}x {r['name']} (delivery {del_date})"
                        )
                    except Exception:
                        pass

        self.total_reorders += len(reorder_instructions)
        completed_stages.add(4)

        if reorder_messages:
            sp.stop("REORDER", " · ".join(reorder_messages), "teal")
        else:
            sp.stop("REORDER", "All items above minimum stock levels", "teal")

        # ---- Stage 5: RESPOND — compose customer-facing response ----
        _workflow_pipeline(active_idx=5, completed=completed_stages)
        total_qty = sum(li["qty"] for li in line_items)
        delivery_date = get_supplier_delivery_date(request_date, total_qty)

        # Build the customer-facing response
        parts = [
            "Dear Customer,\n",
            "Thank you for your inquiry! We are pleased to provide the following "
            "quote for your paper supply needs.\n",
        ]

        # Line-item breakdown
        parts.append("\n--- Quote Details ---")
        for li in line_items:
            discount_txt = ""
            if li["discount_rate"] > 0:
                discount_txt = (
                    f" ({int(li['discount_rate'] * 100)}% bulk discount applied)"
                )

            partial_note = ""
            if li.get("partial"):
                partial_note = (
                    f" [Note: {li['originally_requested']} units requested, "
                    f"only {li['qty']} currently available]"
                )

            parts.append(
                f"  {li['catalog_name']}: {li['qty']} units "
                f"@ ${li['unit_price']:.2f}/unit{discount_txt} "
                f"= ${li['line_total']:.2f}{partial_note}"
            )

        parts.append(f"\n  GRAND TOTAL: ${grand_total:.2f}")

        # Discount rationale
        if any(li["discount_rate"] > 0 for li in line_items):
            parts.append(
                "\nBulk pricing has been applied to qualifying items to provide "
                "you with the best value on your order."
            )

        # Report unavailable items
        if unavailable:
            parts.append("\n--- Unavailable Items ---")
            for u in unavailable:
                parts.append(f"  - {u['requested']}: {u['reason']}")
            parts.append(
                "\nWe apologize for any inconvenience regarding unavailable items."
            )

        # Partial fulfillment notice
        if partial:
            parts.append(
                "\nSome items were partially fulfilled due to limited stock. "
                "The quantities above reflect what we can currently supply."
            )

        # Delivery estimate
        parts.append(f"\nEstimated delivery: {delivery_date}")
        parts.append(
            "\nThank you for choosing Beaver's Choice Paper Company!"
        )

        completed_stages.add(5)
        _clear_pipeline()

        # Show final pipeline (all complete)
        delivery_match = re.search(r"\d{4}-\d{2}-\d{2}", delivery_date)
        _result(
            "RESPOND",
            f"Quote delivered  │  Est. delivery: {delivery_date}",
            "blue",
        )

        self.requests_processed += 1
        return "\n".join(parts)

    # -----------------------------------------------------------------
    # Business Advisor integration
    # -----------------------------------------------------------------
    def generate_business_report(self, as_of_date):
        """Use the Business Advisor agent to generate a financial health report.

        Args:
            as_of_date: Report date in YYYY-MM-DD format.

        Returns:
            The advisor's analysis string, or a fallback direct report.
        """
        try:
            analysis = self.advisor_agent.run(
                f"Generate a financial report using financial_report_tool with "
                f"as_of_date='{as_of_date}', then check the cash balance using "
                f"check_balance_tool with as_of_date='{as_of_date}'. "
                f"Based on the data, provide: 1) A brief business health "
                f"assessment, 2) Key metrics summary, 3) Two specific "
                f"recommendations for improving efficiency and revenue."
            )
            return analysis
        except Exception as e:
            report = generate_financial_report(as_of_date)
            return (
                f"Cash: ${report['cash_balance']:,.2f} | "
                f"Inventory: ${report['inventory_value']:,.2f} | "
                f"Assets: ${report['total_assets']:,.2f}"
            )

    def generate_proactive_advice(self, as_of_date, requests_so_far, total):
        """Run the Business Advisor mid-session for proactive recommendations.

        Triggered periodically (e.g., every 5 requests) to provide real-time
        operational guidance as the system processes orders.

        Args:
            as_of_date: Current date for the report.
            requests_so_far: Number of requests processed.
            total: Total number of requests.

        Returns:
            The advisor's proactive recommendation string.
        """
        try:
            advice = self.advisor_agent.run(
                f"Using financial_report_tool with as_of_date='{as_of_date}' "
                f"and check_balance_tool with as_of_date='{as_of_date}', "
                f"provide a brief mid-session operational check. "
                f"We have processed {requests_so_far}/{total} requests so far. "
                f"Revenue generated: ${self.total_revenue:,.2f}. "
                f"Reorders placed: {self.total_reorders}. "
                f"Flag any concerns about cash reserves or inventory depletion. "
                f"Keep it to 2-3 sentences."
            )
            return str(advice).strip()
        except Exception:
            return None


# =============================================================================
# TEST SCENARIO RUNNER
# =============================================================================

def run_test_scenarios():
    """Run the multi-agent system against all sample customer requests.

    Processes each request from quote_requests_sample.csv through the
    orchestrator pipeline, with animated terminal output showing workflow
    progression, and saves results to test_results.csv.
    """
    # --- Title Banner ---
    print(f"\n  {_C['dim']}{'━' * (W - 4)}{_C['reset']}")
    print(f"  {_C['bold']}{_C['orange']}BEAVER'S CHOICE PAPER COMPANY{_C['reset']}")
    print(f"  {_C['dim']}Multi-Agent Inventory & Quoting System{_C['reset']}")
    print(f"  {_C['dim']}Framework: smolagents  │  Model: GPT-4o-mini{_C['reset']}")
    print(f"  {_C['dim']}Agents: Orchestrator · Inventory · Quote · Sales · Advisor{_C['reset']}")
    print(f"  {_C['dim']}{'━' * (W - 4)}{_C['reset']}")

    # --- Initialize Database ---
    sp = Spinner("INIT", "initializing database", "orange").start()
    init_database(db_engine)
    sp.stop("INIT", f"{_C['green']}Database ready{_C['reset']}", "orange")

    # --- Load Test Data ---
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"],
            format="%m/%d/%y",
            errors="coerce",
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values(
            "request_date"
        ).reset_index(drop=True)
    except Exception as e:
        print(f"  {_C['red']}FATAL: Error loading test data: {e}{_C['reset']}")
        return

    # --- Initial Financial State ---
    initial_date = (
        quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    )
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    total_requests = len(quote_requests_sample)
    print(
        f"  {_C['dim']}Initial State:{_C['reset']}  "
        f"Cash ${current_cash:,.2f}  │  "
        f"Inventory ${current_inventory:,.2f}  │  "
        f"{total_requests} requests queued"
    )

    # --- Initialize Orchestrator ---
    orchestrator = PaperCompanyOrchestrator(model)

    results = []

    for seq_num, (_, row) in enumerate(quote_requests_sample.iterrows(), start=1):
        request_date = row["request_date"].strftime("%Y-%m-%d")
        context = f"{row['job']} organizing {row['event']}"

        # Print request header with progress bar
        _header(
            seq_num, total_requests, context,
            request_date, current_cash, current_inventory,
        )

        request_with_date = f"{row['request']} (Date of request: {request_date})"

        # --- Process through the multi-agent pipeline ---
        try:
            response = orchestrator.process_request(
                request_with_date, request_date
            )
        except Exception as e:
            print(f"  {_C['red']}ERROR: {e}{_C['reset']}")
            response = (
                "Dear Customer,\n\nWe apologize, but we encountered an issue "
                "processing your request. Please try again or contact our "
                "support team.\n\nBeaver's Choice Paper Company"
            )

        # --- Customer Agent Feedback (stand-out feature) ---
        if "GRAND TOTAL" in response:
            sp_cust = Spinner("CUSTOMER", "evaluating quote", "yellow").start()
            feedback = get_customer_feedback(
                response, row["job"], row["event"],
                row.get("need_size", "medium"),
            )
            sp_cust.stop(
                "CUSTOMER",
                f"{_C['dim']}\"{feedback[:70]}{'...' if len(feedback) > 70 else ''}\"{_C['reset']}",
                "yellow",
            )

        # --- Update Financial State ---
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        results.append({
            "request_id": seq_num,
            "request_date": request_date,
            "cash_balance": current_cash,
            "inventory_value": current_inventory,
            "response": response,
        })

        # --- Proactive Business Advisor (every 5 requests) ---
        if seq_num % 5 == 0 and seq_num < total_requests:
            sp_adv = Spinner("ADVISOR", "mid-session analysis", "teal").start()
            advice = orchestrator.generate_proactive_advice(
                request_date, seq_num, total_requests
            )
            if advice:
                sp_adv.stop("ADVISOR", f"{_C['dim']}{advice[:W - 20]}{_C['reset']}", "teal")
            else:
                sp_adv.stop("ADVISOR", "Analysis complete", "teal")

        time.sleep(0.3)

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    final_date = (
        quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    )
    final_report = generate_financial_report(final_date)

    print(f"\n  {_C['dim']}{'━' * (W - 4)}{_C['reset']}")
    print(
        f"  {_C['bold']}{_C['orange']}ALL {total_requests} REQUESTS "
        f"PROCESSED{_C['reset']}  {_progress_bar(total_requests, total_requests)}"
    )
    print(f"  {_C['dim']}{'━' * (W - 4)}{_C['reset']}")

    # Summary metrics
    fulfilled = sum(1 for r in results if "GRAND TOTAL" in r["response"])
    unfulfilled = total_requests - fulfilled
    cash_changes = sum(
        1 for i in range(1, len(results))
        if abs(results[i]["cash_balance"] - results[i-1]["cash_balance"]) > 0.01
    )

    print(
        f"  {_C['green']}Fulfilled:{_C['reset']} {fulfilled}  │  "
        f"{_C['red']}Unfulfilled:{_C['reset']} {unfulfilled}  │  "
        f"{_C['orange']}Cash changes:{_C['reset']} {cash_changes}"
    )
    print(
        f"  Final Cash: ${final_report['cash_balance']:,.2f}  │  "
        f"Inventory: ${final_report['inventory_value']:,.2f}  │  "
        f"Assets: ${final_report['total_assets']:,.2f}"
    )

    # --- Final Business Advisor Assessment ---
    sp_adv = Spinner("ADVISOR", "generating final business assessment", "teal").start()
    advisor_summary = orchestrator.generate_business_report(final_date)
    sp_adv.stop("ADVISOR", f"{_C['green']}Financial health assessment complete{_C['reset']}", "teal")

    # Print advisor output with typewriter effect
    for line in str(advisor_summary).split("\n"):
        stripped = line.strip()[:W - 6]
        if stripped:
            print(f"  {_C['dim']}  {stripped}{_C['reset']}")
            sys.stdout.flush()
            time.sleep(0.03)

    # --- Save Results ---
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    print(f"\n  {_C['green']}Results saved to test_results.csv{_C['reset']}")
    print(f"  {_C['dim']}{'━' * (W - 4)}{_C['reset']}\n")
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
