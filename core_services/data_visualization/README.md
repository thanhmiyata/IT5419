# Data Visualization Module

Standalone visualization tools for development and analysis purposes. Generate interactive charts from database data without requiring the full UI.

## Features

- **Candlestick Charts**: OHLC data with volume overlay
- **Line Charts**: Simple closing price trends
- **Volume Charts**: Color-coded volume bars
- **Technical Indicators**: MA, RSI, MACD, Bollinger Bands
- **Comparison Charts**: Multi-stock normalized comparison
- **Export**: Save to HTML for sharing

## Installation

```bash
pip install plotly sqlalchemy psycopg2-binary
```

## Environment Setup

Set the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/stockaids"
```

Or it will use the default: `postgresql://user:password@localhost:5432/stockaids`

## Usage

### 1. Candlestick Chart

```bash
# Command line
python -m core_services.data_visualization.candlestick_chart VNM 2024-01-01 2024-01-31 output.html

# Python code
from core_services.data_visualization import plot_candlestick

fig = plot_candlestick(
    symbol="VNM",
    start_date="2024-01-01",
    end_date="2024-01-31",
    output_file="vnm_candlestick.html",
    show=True  # Open in browser
)
```

### 2. Line Chart

```bash
# Command line
python -m core_services.data_visualization.line_chart FPT 2024-01-01 2024-01-31 output.html

# Python code
from core_services.data_visualization import plot_line_chart

fig = plot_line_chart(
    symbol="FPT",
    start_date="2024-01-01",
    end_date="2024-01-31",
    output_file="fpt_line.html"
)
```

### 3. Volume Chart

```bash
# Command line
python -m core_services.data_visualization.volume_chart VIC 2024-01-01 2024-01-31 output.html

# Python code
from core_services.data_visualization import plot_volume_chart

fig = plot_volume_chart(
    symbol="VIC",
    start_date="2024-01-01",
    end_date="2024-01-31",
    output_file="vic_volume.html"
)
```

### 4. Technical Indicators

```bash
# Command line
python -m core_services.data_visualization.technical_chart VNM 2024-01-01 2024-01-31 output.html

# Python code
from core_services.data_visualization import plot_technical_indicators

fig = plot_technical_indicators(
    symbol="VNM",
    start_date="2024-01-01",
    end_date="2024-01-31",
    indicators=["MA20", "MA50", "RSI", "MACD"],  # Optional
    output_file="vnm_technical.html"
)
```

**Available Indicators:**
- `MA20`, `MA50`, `MA200` - Moving Averages
- `RSI` - Relative Strength Index (14-period)
- `MACD` - Moving Average Convergence Divergence

### 5. Comparison Chart

```bash
# Command line (comma-separated symbols)
python -m core_services.data_visualization.comparison_chart VNM,FPT,VIC 2024-01-01 2024-01-31 output.html

# Python code
from core_services.data_visualization import plot_comparison

fig = plot_comparison(
    symbols=["VNM", "FPT", "VIC"],
    start_date="2024-01-01",
    end_date="2024-01-31",
    normalize=True,  # Normalize to 100 at start
    output_file="comparison.html"
)
```

## Output

All charts are saved as interactive HTML files that can be:
- Opened in any web browser
- Shared with team members
- Embedded in reports or documentation

Charts include:
- Hover tooltips with detailed information
- Zoom and pan capabilities
- Legend toggling
- Export to PNG (via browser)

## Examples

### Quick Analysis

```python
from core_services.data_visualization import (
    plot_candlestick,
    plot_technical_indicators,
    plot_comparison
)

# Single stock analysis
plot_candlestick("VNM", "2024-01-01", "2024-01-31", "vnm_price.html")
plot_technical_indicators("VNM", "2024-01-01", "2024-01-31", "vnm_ta.html")

# Multi-stock comparison
plot_comparison(
    ["VNM", "FPT", "VIC"],
    "2024-01-01", "2024-01-31",
    "comparison.html"
)
```

### Custom Analysis

```python
from core_services.data_visualization import plot_technical_indicators

# Only show moving averages (no RSI/MACD)
fig = plot_technical_indicators(
    symbol="VNM",
    start_date="2024-01-01",
    end_date="2024-01-31",
    indicators=["MA20", "MA50", "MA200"],
    output_file="vnm_ma_only.html",
    show=False  # Don't open browser
)
```

## Database Requirements

These modules require:
1. PostgreSQL database with `stocks` and `stock_prices` tables
2. Stock price data with OHLC fields (open, high, low, close, volume)
3. Access to `app.schemas.models` and `app.services.technical_indicators`

## Error Handling

Common errors:
- `Stock {symbol} not found`: Symbol doesn't exist in database
- `No price data found`: No data for the specified date range
- `Connection refused`: Database is not running

## Notes

- All dates should be in ISO format: `YYYY-MM-DD`
- Stock symbols are case-insensitive (VNM = vnm)
- Default behavior opens chart in browser unless `show=False`
- Charts require data to exist in the database for the specified date range
