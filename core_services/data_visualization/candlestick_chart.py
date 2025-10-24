"""
Candlestick Chart Visualization
================================

Create interactive candlestick charts using Plotly.
"""

import os
from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import and_, create_engine
from sqlalchemy.orm import sessionmaker


def plot_candlestick(
    symbol: str,
    start_date: str,
    end_date: str,
    output_file: Optional[str] = None,
    show: bool = True,
    database_url: Optional[str] = None
):
    """
    Plot candlestick chart for a stock symbol.

    Args:
        symbol: Stock symbol (e.g., "VNM")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_file: Optional path to save HTML file
        show: Whether to display the chart in browser
        database_url: Database connection string (uses env var if not provided)

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> plot_candlestick("VNM", "2024-01-01", "2024-01-31")
        >>> plot_candlestick("VNM", "2024-01-01", "2024-01-31", output_file="vnm_chart.html")
    """
    # Database connection
    if database_url is None:
        database_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/stockaids")

    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Import models (assuming they"re available)
        import os.path as osp
        from sys import path as sys_path
        backend_path = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "stockaids-backend")
        if backend_path not in sys_path:
            sys_path.insert(0, backend_path)

        from app.schemas.models import Stock, StockPrice

        # Get stock
        stock = session.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if not stock:
            raise ValueError(f"Stock {symbol} not found in database")

        # Parse dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Query stock prices
        prices = session.query(StockPrice).filter(
            and_(StockPrice.stock_id == stock.id, StockPrice.date >= start_dt, StockPrice.date <= end_dt)
        ).order_by(StockPrice.date.asc()).all()

        if not prices:
            raise ValueError(f"No price data found for {symbol} between {start_date} and {end_date}")

        # Prepare data
        dates = [p.date for p in prices]
        opens = [float(p.open) if p.open else None for p in prices]
        highs = [float(p.high) if p.high else None for p in prices]
        lows = [float(p.low) if p.low else None for p in prices]
        closes = [float(p.close) if p.close else None for p in prices]
        volumes = [int(p.volume) if p.volume else 0 for p in prices]

        # Create candlestick chart with volume subplot
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            subplot_titles=[f"{symbol.upper()} Stock Price", "Volume"],
            row_heights=[0.7, 0.3]
        )

        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes, name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
        ), row=1, col=1)

        # Add volume bars on separate subplot
        fig.add_trace(go.Bar(
            x=dates, y=volumes, name="Volume",
            marker=dict(color=["#26a69a" if closes[i] >= opens[i] else "#ef5350" for i in range(len(closes))])
        ), row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f"{symbol.upper()} Stock Price ({start_date} to {end_date})",
            xaxis2_title="Date", xaxis_rangeslider_visible=False, hovermode="x unified",
            template="plotly_white", height=600, showlegend=True
        )

        # Update y-axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Ensure x-axis is properly configured
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

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
    # Example usage
    import sys

    if len(sys.argv) >= 4:
        symbol = sys.argv[1]
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        output = sys.argv[4] if len(sys.argv) > 4 else None
        plot_candlestick(symbol, start_date, end_date, output_file=output)
    else:
        # Default example
        plot_candlestick("VNM", "2024-01-01", "2024-01-31", output_file="vnm_candlestick.html")
