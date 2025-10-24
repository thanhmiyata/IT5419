"""
Volume Chart Visualization
===========================

Create interactive volume bar charts.
"""

import os
from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
from sqlalchemy import and_, create_engine
from sqlalchemy.orm import sessionmaker


def plot_volume_chart(
    symbol: str,
    start_date: str,
    end_date: str,
    output_file: Optional[str] = None,
    show: bool = True,
    database_url: Optional[str] = None
):
    """
    Plot volume bar chart with color-coded bars.

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
        volumes = [int(p.volume) if p.volume else 0 for p in prices]
        opens = [float(p.open) if p.open else None for p in prices]
        closes = [float(p.close) if p.close else None for p in prices]

        # Determine bar colors (green if close >= open, else red)
        colors = ["#26a69a" if closes[i] >= opens[i] else "#ef5350" for i in range(len(prices))]

        # Create volume chart
        fig = go.Figure()

        fig.add_trace(go.Bar(x=dates, y=volumes, name="Volume", marker=dict(color=colors)))

        # Calculate total volume
        total_volume = sum(volumes)

        # Update layout
        title_text = (
            f"{symbol.upper()} Trading Volume ({start_date} to {end_date})<br>"
            f"<sub>Total Volume: {total_volume:,} shares</sub>"
        )
        fig.update_layout(
            title=title_text, xaxis_title="Date", yaxis_title="Volume",
            hovermode="x unified", template="plotly_white", height=400
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
        plot_volume_chart(symbol, start_date, end_date, output_file=output)
    else:
        plot_volume_chart("VNM", "2024-01-01", "2024-01-31", output_file="vnm_volume.html")
