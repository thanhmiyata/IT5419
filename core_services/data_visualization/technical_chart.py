"""
Technical Indicators Chart Visualization
=========================================

Create interactive charts with technical indicators overlay.
"""

import os
from datetime import datetime
from typing import List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import and_, create_engine
from sqlalchemy.orm import sessionmaker


def _calculate_subplot_config(indicators: List[str]):
    """Calculate subplot configuration based on indicators."""
    has_rsi = "RSI" in indicators
    has_macd = "MACD" in indicators
    subplot_count = 1 + (1 if has_rsi else 0) + (1 if has_macd else 0)

    subplot_titles = ["Price"]
    if has_rsi:
        subplot_titles.append("RSI")
    if has_macd:
        subplot_titles.append("MACD")

    if subplot_count == 1:
        row_heights = [1.0]
    elif subplot_count == 2:
        row_heights = [0.7, 0.3]
    else:
        row_heights = [0.6, 0.2, 0.2]

    return subplot_count, subplot_titles, row_heights, has_rsi, has_macd


def _add_moving_average(fig, prices, period, name, color, start_dt):
    """Add a moving average trace to the figure."""
    from app.services.technical_indicators import TechnicalIndicatorService

    ma_data = TechnicalIndicatorService.calculate_moving_averages(prices, start_date=start_dt, periods=[period])
    if ma_data and f"MA{period}" in ma_data:
        ma_values_list = ma_data[f"MA{period}"]
        ma_dates = [d["date"] for d in ma_values_list]
        ma_values = [d["value"] for d in ma_values_list]
        fig.add_trace(go.Scatter(
            x=ma_dates, y=ma_values, name=name, line=dict(color=color, width=1)
        ), row=1, col=1)


def _add_rsi_indicator(fig, prices, row, start_dt):
    """Add RSI indicator to the figure."""
    from app.services.technical_indicators import TechnicalIndicatorService

    rsi_data = TechnicalIndicatorService.calculate_rsi(prices, start_date=start_dt, period=14)
    if rsi_data:
        rsi_dates = [d["date"] for d in rsi_data]
        rsi_values = [d["value"] for d in rsi_data]
        fig.add_trace(
            go.Scatter(x=rsi_dates, y=rsi_values, name="RSI", line=dict(color="#1976d2", width=2)), row=row, col=1
        )

        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=row, col=1)


def _add_macd_indicator(fig, prices, row, start_dt):
    """Add MACD indicator to the figure."""
    from app.services.technical_indicators import TechnicalIndicatorService

    macd_result = TechnicalIndicatorService.calculate_macd(prices, start_date=start_dt)
    if macd_result and "macd" in macd_result:
        macd_data = macd_result["macd"]
        signal_data = macd_result["signal"]
        histogram_data = macd_result["histogram"]

        macd_dates = [d["date"] for d in macd_data]
        macd_values = [d["value"] for d in macd_data]
        signal_values = [d["value"] for d in signal_data]
        histogram_values = [d["value"] for d in histogram_data]

        fig.add_trace(go.Scatter(
            x=macd_dates, y=macd_values, name="MACD", line=dict(color="#1976d2", width=2)
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=macd_dates, y=signal_values, name="Signal", line=dict(color="#ff9800", width=2)
        ), row=row, col=1)

        fig.add_trace(go.Bar(
            x=macd_dates, y=histogram_values, name="Histogram",
            marker=dict(color=["#26a69a" if h >= 0 else "#ef5350" for h in histogram_values])
        ), row=row, col=1)


def _add_moving_averages(fig, indicators, prices, start_dt):
    """Add all requested moving averages to the figure."""
    ma_configs = {
        "MA20": (20, "MA20", "#1976d2"),
        "MA50": (50, "MA50", "#ff9800"),
        "MA200": (200, "MA200", "#9c27b0")
    }

    for indicator, (period, name, color) in ma_configs.items():
        if indicator in indicators:
            _add_moving_average(fig, prices, period, name, color, start_dt)


def _update_chart_layout(fig, symbol, start_date, end_date, subplot_count, has_rsi, has_macd):
    """Update the chart layout and axes."""
    title_text = f"{symbol.upper()} Technical Analysis ({start_date} to {end_date})"
    fig.update_layout(
        title=title_text, xaxis_rangeslider_visible=False, hovermode="x unified",
        template="plotly_white", height=600 + (200 * (subplot_count - 1)), showlegend=True
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if has_rsi:
        rsi_row = 2 if not has_macd else 2
        fig.update_yaxes(title_text="RSI", row=rsi_row, col=1)
    if has_macd:
        macd_row = subplot_count
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1)


def plot_technical_indicators(
    symbol: str,
    start_date: str,
    end_date: str,
    indicators: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    show: bool = True,
    database_url: Optional[str] = None
):
    """
    Plot candlestick chart with technical indicators overlay.

    Args:
        symbol: Stock symbol (e.g., "VNM")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        indicators: List of indicators to plot
                   (e.g., ["MA20", "MA50", "RSI", "MACD"])
        output_file: Optional path to save HTML file
        show: Whether to display the chart in browser
        database_url: Database connection string

    Returns:
        plotly.graph_objects.Figure
    """
    # Default indicators
    if indicators is None:
        indicators = ["MA20", "MA50", "RSI", "MACD"]

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

        # Get stock
        stock = session.query(Stock).filter(
            Stock.symbol == symbol.upper()
        ).first()
        if not stock:
            raise ValueError(f"Stock {symbol} not found in database")

        # Parse dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Query stock prices with buffer for technical indicators
        # MACD needs 26 + 9 = 35 days, MA200 needs 200 days
        # Fetch extra 250 trading days (~1 year) to ensure enough data
        from datetime import timedelta
        buffer_start = start_dt - timedelta(days=365)

        price_records = session.query(StockPrice).filter(
            and_(StockPrice.stock_id == stock.id, StockPrice.date >= buffer_start, StockPrice.date <= end_dt)
        ).order_by(StockPrice.date.asc()).all()

        if not price_records:
            raise ValueError(f"No price data found for {symbol} between {start_date} and {end_date}")

        # Filter display data to requested range
        display_records = [p for p in price_records if p.date >= start_dt.date()]
        if not display_records:
            raise ValueError(f"No price data found for {symbol} between {start_date} and {end_date}")

        # Prepare data for display (only requested range)
        dates = [p.date for p in display_records]
        opens = [float(p.open) if p.open else None for p in display_records]
        highs = [float(p.high) if p.high else None for p in display_records]
        lows = [float(p.low) if p.low else None for p in display_records]
        closes = [float(p.close) if p.close else None for p in display_records]

        # Calculate subplot configuration
        subplot_count, subplot_titles, row_heights, has_rsi, has_macd = _calculate_subplot_config(indicators)

        # Create subplots
        fig = make_subplots(
            rows=subplot_count, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=subplot_titles, row_heights=row_heights
        )

        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes, name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
        ), row=1, col=1)

        # Add technical indicators (use all price_records including buffer for calculation)
        # Add moving averages
        _add_moving_averages(fig, indicators, price_records, start_dt)

        # Add RSI
        current_row = 1
        if has_rsi:
            current_row += 1
            _add_rsi_indicator(fig, price_records, current_row, start_dt)

        # Add MACD
        if has_macd:
            current_row += 1
            _add_macd_indicator(fig, price_records, current_row, start_dt)

        # Update layout
        _update_chart_layout(fig, symbol, start_date, end_date, subplot_count, has_rsi, has_macd)

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
        plot_technical_indicators(
            symbol, start_date, end_date, output_file=output
        )
    else:
        plot_technical_indicators(
            "VNM", "2024-01-01", "2024-01-31",
            output_file="vnm_technical.html"
        )
