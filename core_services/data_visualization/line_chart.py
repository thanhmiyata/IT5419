"""
Line Chart Visualization
=========================

Create interactive line charts for closing prices.
"""

import os
from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
from sqlalchemy import and_, create_engine
from sqlalchemy.orm import sessionmaker


def plot_line_chart(
    symbol: str,
    start_date: str,
    end_date: str,
    output_file: Optional[str] = None,
    show: bool = True,
    database_url: Optional[str] = None
):
    """
    Plot line chart for stock closing prices.

    Args:
        symbol: Stock symbol (e.g., "VNM")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_file: Optional path to save HTML file
        show: Whether to display the chart in browser
        database_url: Database connection string

    Returns:
        plotly.graph_objects.Figure
    """
    # Database connection
    if database_url is None:
        database_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/stockaids")

    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
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
        closes = [float(p.close) if p.close else None for p in prices]

        # Create line chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates, y=closes, mode="lines+markers", name="Close Price",
            line=dict(color="#1976d2", width=2), marker=dict(size=4)
        ))

        # Update layout
        fig.update_layout(
            title=f"{symbol.upper()} Closing Price ({start_date} to {end_date})",
            xaxis_title="Date", yaxis_title="Price", hovermode="x unified",
            template="plotly_white", height=500
        )

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
        symbol = sys.argv[1]
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        output = sys.argv[4] if len(sys.argv) > 4 else None
        plot_line_chart(symbol, start_date, end_date, output_file=output)
    else:
        plot_line_chart("VNM", "2024-01-01", "2024-01-31", output_file="vnm_line.html")
