# Core Services - Danh sách Tools và Modules

## Tổng quan
Core Services là một hệ thống toàn diện cho việc thu thập, xử lý và phân tích dữ liệu thị trường chứng khoán Việt Nam. Hệ thống bao gồm 4 module chính:

## 1. Data Ingestion Module (`data_ingestion/`)

### 1.1 Pipeline Architecture
- **PipelineOrchestrator**: Quản lý và điều phối toàn bộ pipeline
- **PipelineFactory**: Tạo các pipeline cho các loại dữ liệu khác nhau
- **CrawlingPipeline**: Pipeline chính cho việc crawl dữ liệu
- **TaskManager**: Quản lý các task với priority và scheduling

### 1.2 Source Adapters
- **VNStockAdapter**: Tích hợp với thư viện VNStock
- **CafeFAdapter**: Scraping dữ liệu từ CafeF.vn
- **VnExpressNewsAdapter**: Thu thập tin tức từ VnExpress
- **InvestingComAdapter**: Tích hợp với Investing.com

### 1.3 Data Processing
- **DataNormalization**: Chuẩn hóa dữ liệu từ các nguồn khác nhau
- **DataQuality**: Kiểm tra và đảm bảo chất lượng dữ liệu
- **FieldMappings**: Mapping các trường dữ liệu giữa các nguồn
- **TextProcessor**: Xử lý văn bản và tin tức

### 1.4 Storage
- **DatabaseStorage**: Lưu trữ dữ liệu vào PostgreSQL
- **QdrantStorage**: Lưu trữ vector embeddings cho tìm kiếm semantic
- **BackendIntegration**: Tích hợp với backend API

### 1.5 Monitoring & Configuration
- **Monitoring**: Giám sát hiệu suất và health check
- **ConfigManager**: Quản lý cấu hình hệ thống
- **SourceSchema**: Quản lý schema cho các nguồn dữ liệu

## 2. Data Visualization Module (`data_visualization/`)

### 2.1 Chart Types
- **CandlestickChart**: Biểu đồ nến OHLC với volume overlay
- **LineChart**: Biểu đồ đường giá đóng cửa
- **VolumeChart**: Biểu đồ khối lượng giao dịch
- **TechnicalChart**: Biểu đồ kỹ thuật với các chỉ báo
- **ComparisonChart**: So sánh nhiều cổ phiếu

### 2.2 Features
- Export sang HTML interactive
- Hover tooltips chi tiết
- Zoom và pan capabilities
- Legend toggling
- Export to PNG

### 2.3 Technical Indicators
- Moving Averages (MA20, MA50, MA200)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators

## 3. ML Models Module (`ml_models/`)

### 3.1 Indicators (`indicators/`)
- **FundamentalIndicators**: Phân tích cơ bản (P/E, P/B, ROE, ROA, etc.)
- **TrendIndicators**: Chỉ báo xu hướng (MA, MACD, ADX)
- **MomentumIndicators**: Chỉ báo momentum (RSI, Stochastic, ROC, CCI)
- **VolumeIndicators**: Chỉ báo khối lượng (OBV, VWAP, MFI)
- **VolatilityIndicators**: Chỉ báo biến động (Bollinger Bands, ATR)
- **SupportResistanceIndicators**: Hỗ trợ/kháng cự (Fibonacci, Pivot Points)
- **MarketBreadthIndicators**: Độ rộng thị trường (A/D Line, Put-Call Ratio)
- **FinancialStrengthIndicators**: Sức mạnh tài chính (liquidity, solvency ratios)
- **RiskManagementIndicators**: Quản lý rủi ro (VaR, Sharpe ratio, drawdown)
- **PortfolioManagementIndicators**: Quản lý danh mục (diversification, allocation)

### 3.2 Quantitative Models (`quant_models/`)

#### 3.2.1 Deep Learning (`deep_learning/`)
- **LSTMModel**: Mô hình LSTM cho dự đoán giá
- **GRUModel**: Mô hình GRU cho time series
- **TransformerModel**: Mô hình Transformer cho financial data

#### 3.2.2 Ensemble Methods (`ensemble/`)
- **RandomForest**: Random Forest cho classification/regression
- **XGBoostModel**: XGBoost cho gradient boosting

#### 3.2.3 Time Series (`time_series/`)
- **ARIMAModel**: Mô hình ARIMA cho time series
- **GARCHModel**: Mô hình GARCH cho volatility
- **KalmanFilter**: Kalman Filter cho state estimation

#### 3.2.4 Risk Management (`risk/`)
- **ExtremeValueTheory**: Phân tích extreme values
- **MonteCarloSimulation**: Mô phỏng Monte Carlo

#### 3.2.5 Portfolio Optimization (`optimization/`)
- **MarkowitzOptimizer**: Tối ưu hóa Markowitz Mean-Variance
- **BlackLittermanModel**: Mô hình Black-Litterman
- **RiskParityOptimizer**: Tối ưu hóa Risk Parity
- **KellyCriterion**: Kelly Criterion cho position sizing

#### 3.2.6 Trading Strategies (`strategies/`)
- **MomentumStrategy**: Chiến lược momentum
- **MeanReversionStrategy**: Chiến lược mean reversion
- **PairsTradingStrategy**: Chiến lược pairs trading
- **BreakoutStrategy**: Chiến lược breakout

#### 3.2.7 Factor Models (`factors/`)
- **FamaFrenchModel**: Mô hình Fama-French 3 factors
- **MomentumFactor**: Momentum factor analysis
- **MultiFactorModel**: Multi-factor models

#### 3.2.8 Regime Detection (`regime/`)
- **GMMRegimeModel**: Gaussian Mixture Model cho regime detection
- **HMMRegimeModel**: Hidden Markov Model cho regime switching

#### 3.2.9 Microstructure (`microstructure/`)
- **HawkesProcess**: Hawkes process cho high-frequency data

#### 3.2.10 Model Evaluation (`evaluation/`)
- **Evaluator**: Đánh giá hiệu suất mô hình
- **HyperparameterTuning**: Tuning siêu tham số
- **Metrics**: Các metrics đánh giá (Sharpe, Sortino, etc.)
- **TrainUtils**: Utilities cho training

### 3.3 Examples (`examples/`)
- **BacktestStrategyExample**: Ví dụ backtest strategy
- **HyperparameterTuningExample**: Ví dụ tuning hyperparameters
- **TrainLSTMExample**: Ví dụ training LSTM model

## 4. Utils Module (`utils/`)

### 4.1 Common Utilities
- **Common**: Constants, enumerations, shared types
- **LoggerUtils**: Logging utilities và configuration

### 4.2 Key Constants
- Data source types (VNSTOCK, CAFEF, etc.)
- Data types (STOCK_PRICE, INDEX_DATA, etc.)
- Task priorities và status
- Rate limiting configurations
- Database table names
- Vietnamese stock market specific constants

## 5. Scripts (`scripts/`)

### 5.1 Execution Scripts
- **run_data_ingestion.py**: Chạy data ingestion pipeline
- **run_bctc.py**: Chạy xử lý báo cáo tài chính
- **test_bctc_normalization.py**: Test normalization
- **psql_test.sh**: Test PostgreSQL connection

## 6. Dependencies

### 6.1 Core Dependencies
- **asyncio-mqtt**: MQTT messaging
- **redis**: Redis client
- **sqlalchemy**: ORM
- **aiohttp**: Async HTTP client
- **vnstock**: Vietnamese stock data library
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### 6.2 ML Dependencies
- **plotly**: Interactive charts
- **sentence-transformers**: Text embeddings
- **qdrant-client**: Vector database

### 6.3 Development Dependencies
- **pytest**: Testing framework
- **mypy**: Type checking
- **black**: Code formatting
- **flake8**: Linting

## 7. Cách sử dụng

### 7.1 Data Ingestion
```python
from core_services.data_ingestion.pipeline_runner import PipelineManager

# Chạy pipeline
manager = PipelineManager()
await manager.run()
```

### 7.2 Data Visualization
```python
from core_services.data_visualization import plot_candlestick

# Tạo biểu đồ nến
fig = plot_candlestick(
    symbol="VNM",
    start_date="2024-01-01",
    end_date="2024-01-31",
    output_file="vnm_chart.html"
)
```

### 7.3 ML Models
```python
from core_services.ml_models.indicators import TrendIndicators
from core_services.ml_models.quant_models.strategies import MomentumStrategy

# Sử dụng indicators
trend = TrendIndicators()
ma20 = trend.simple_moving_average(prices, 20)

# Sử dụng strategies
strategy = MomentumStrategy()
signals = strategy.generate_signals(data)
```

## 8. Cấu hình

### 8.1 Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `ENVIRONMENT`: development/testing/production

### 8.2 Configuration Files
- Schema files trong `schemas/`
- Configuration presets cho các môi trường khác nhau

## 9. Monitoring

### 9.1 Health Checks
- HTTP health check endpoint
- Redis connectivity check
- Database connectivity check
- Task metrics monitoring

### 9.2 Metrics
- Task success rate
- Response times
- Queue sizes
- Error rates

## 10. Lưu ý

- Hệ thống được thiết kế cho thị trường chứng khoán Việt Nam
- Hỗ trợ multiple data sources với fallback
- Rate limiting để tuân thủ các nguồn dữ liệu
- Async/await pattern cho performance tốt
- Comprehensive error handling và retry logic
- Modular design cho dễ mở rộng và bảo trì
