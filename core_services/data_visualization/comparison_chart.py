"""
Comparison Chart Visualization
===============================

Create normalized comparison charts for multiple stocks.
"""

import os
from datetime import datetime
from typing import List, Optional

import plotly.graph_objects as go
from sqlalchemy import and_, create_engine
from sqlalchemy.orm import sessionmaker


def plot_comparison(
    symbols: List[str],
    start_date: str,
    end_date: str,
    normalize: bool = True,
    output_file: Optional[str] = None,
    show: bool = True,
    database_url: Optional[str] = None
):
    """
    Plot comparison chart for multiple stocks.

    Args:
        symbols: List of stock symbols (e.g., ["VNM", "FPT", "VIC"])
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        normalize: Whether to normalize prices to start at 100
        output_file: Optional path to save HTML file
        show: Whether to display the chart in browser
        database_url: Database connection string

    Returns:
        plotly.graph_objects.Figure
    """
    # Database connection
    if database_url is None:
        db_url = "postgresql://user:password@localhost:5432/stockaids"
        database_url = os.getenv("DATABASE_URL", db_url)

    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        import os.path as osp
        from sys import path as sys_path
        backend_path = osp.join(
            osp.dirname(osp.dirname(osp.dirname(__file__))),
            "stockaids-backend"
        )
        if backend_path not in sys_path:
            sys_path.insert(0, backend_path)

        from app.schemas.models import Stock, StockPrice

        # Parse dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Create figure
        fig = go.Figure()

        # Define colors for different stocks
        colors = ["#1976d2", "#ff9800", "#4caf50", "#9c27b0", "#f44336", "#00bcd4"]

        # Plot each stock
        for idx, symbol in enumerate(symbols):
            # Get stock
            stock = session.query(Stock).filter(Stock.symbol == symbol.upper()).first()
            if not stock:
                print(f"Warning: Stock {symbol} not found, skipping")
                continue

            # Query stock prices
            prices = session.query(StockPrice).filter(
                and_(StockPrice.stock_id == stock.id, StockPrice.date >= start_dt, StockPrice.date <= end_dt)
            ).order_by(StockPrice.date.asc()).all()

            if not prices:
                print(f"Warning: No data for {symbol}, skipping")
                continue

            # Prepare data
            dates = [p.date for p in prices]
            closes = [float(p.close) if p.close else None for p in prices]

            # Normalize if requested
            if normalize and closes and closes[0] is not None:
                first_price = closes[0]
                closes = [(price / first_price * 100) if price is not None else None for price in closes]

            # Add trace
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=dates, y=closes, mode="lines", name=symbol.upper(), line=dict(color=color, width=2)
            ))

        # Update layout
        y_label = "Normalized Price (%)" if normalize else "Price"
        title_symbols = ", ".join([s.upper() for s in symbols])
        title_text = f"Stock Comparison: {title_symbols} ({start_date} to {end_date})"

        fig.update_layout(
            title=title_text, xaxis_title="Date", yaxis_title=y_label, hovermode="x unified",
            template="plotly_white", height=600, showlegend=True
        )

        # Add baseline if normalized
        if normalize:
            fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

        # Save to file if specified
        if output_file:
            fig.write_html(output_file)
            print(f"Chart saved to {output_file}")

        # Show in browser if requested
        if show:
            fig.show()

        return fig

    finally:
        session.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 4:
        # Parse symbols (comma-separated)
        symbols = sys.argv[1].split(",")
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        output = sys.argv[4] if len(sys.argv) > 4 else None
        plot_comparison(symbols, start_date, end_date, output_file=output)
    else:
        # Default example
        plot_comparison(
            ["VNM", "FPT", "VIC"],
            "2024-01-01", "2024-01-31",
            output_file="comparison.html"
        )
