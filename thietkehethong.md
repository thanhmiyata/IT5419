# THIẾT KẾ HỆ THỐNG BOT CHAT TƯ VẤN CỔ PHIẾU

## 📋 TỔNG QUAN DỰ ÁN

### Mục tiêu
Xây dựng hệ thống Bot chat tư vấn cổ phiếu thông minh, cung cấp phân tích kỹ thuật, dự đoán xu hướng và khuyến nghị đầu tư cho người dùng.

### Phạm vi
- **Backend**: API services, AI agents, data processing
- **Frontend**: Chat interface, chart visualization, admin dashboard
- **Core Services**: ML models, data ingestion, indicators, visualization

---

## 🏗️ KIẾN TRÚC TỔNG THỂ

### 1. Kiến trúc Master-Slave
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │ Core Services   │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   AI Agents     │
                    │   (Gemini 2.0)  │
                    └─────────────────┘
                              ▲
                              │
                    ┌─────────────────┐
                    │    Backend      │
                    │   (FastAPI)    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Databases     │
                    │PostgreSQL+Qdrant│
                    └─────────────────┘
```

### 2. Luồng xử lý chính
```
User Query → OrchestrationAgent → PlanningAgent → Tool Execution → ResponseGenerator → User
```

---

## 🔧 BACKEND ARCHITECTURE

### 1. Core Components

#### 1.1 OrchestrationAgent (Master Agent)
**Chức năng:**
- Tiếp nhận và phân tích câu hỏi từ user
- Quản lý conversation context (tối đa 5 câu hỏi trước)
- Điều phối các Agent khác
- Xử lý error handling và retry logic

**API Endpoints:**
```python
POST /api/v1/chat/query
GET  /api/v1/chat/history/{session_id}
POST /api/v1/chat/context/update
```

#### 1.2 PlanningAgent
**Chức năng:**
- Phân tích câu hỏi và tạo execution plan
- Xác định các tool cần thiết
- Đánh giá độ phức tạp của task
- Tối đa 3 lần call cho 1 câu hỏi
- Sử dụng Qdrant để tìm similar financial reports khi cần

**Integration:**
- Gemini 2.0 Flash API
- Tool registry và capability mapping
- Qdrant vector search cho financial reports similarity

#### 1.3 ResponseGenerator
**Chức năng:**
- Tạo câu trả lời hoàn chỉnh
- Kết hợp text và visual content
- Format response theo template
- Xử lý multi-modal content
- Sử dụng Qdrant để tìm relevant financial reports examples

**Integration:**
- Gemini 2.0 Flash API
- Qdrant vector search cho financial reports examples
- PostgreSQL cho structured data

### 2. Tool Ecosystem

#### 2.1 Data Tools
```python
class DataTools:
    def get_stock_data(symbol: str, start_date: str, end_date: str)
    def get_financial_data(symbol: str, period: str)
    def get_company_info(symbol: str)
    def get_financial_statements(symbol: str, period: str)
    def search_company_info(query: str)
    def search_financial_reports_semantic(query: str, symbol: str = None, year: int = None)
```

#### 2.2 Analysis Tools
```python
class AnalysisTools:
    def calculate_indicators(symbol: str, indicators: List[str])
    def compare_stocks(symbols: List[str], period: str)
    def get_top_movers(period: str, limit: int)
    def analyze_market_trends(period: str)
```

#### 2.3 Prediction Tools
```python
class PredictionTools:
    def predict_price_trend(symbol: str, model_version: str)
    def predict_buy_sell_signals(symbol: str, model_version: str)
    def get_portfolio_recommendations(criteria: dict)
    def analyze_risk_factors(symbol: str)
```

#### 2.4 Visualization Tools
```python
class VisualizationTools:
    def generate_chart(data: dict, chart_type: str)
    def create_comparison_chart(data: dict)
    def generate_technical_analysis_chart(symbol: str)
    def create_portfolio_visualization(portfolio: dict)
```

### 3. Database Design

#### 3.1 PostgreSQL Schema
```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat sessions
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat messages
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id),
    message_type VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stock data
CREATE TABLE stock_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- Financial data
CREATE TABLE financial_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    period VARCHAR(20) NOT NULL, -- 'Q1', 'Q2', 'Q3', 'Q4', 'YEAR'
    year INTEGER NOT NULL,
    revenue DECIMAL(15,2),
    net_income DECIMAL(15,2),
    assets DECIMAL(15,2),
    liabilities DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, period, year)
);
```

#### 3.2 Qdrant Vector Database Structure
```python
# Qdrant Collections - Chỉ lưu embeddings từ core_services
class QdrantCollections:
    # Financial reports embeddings (từ core_services)
    FINANCIAL_REPORTS_EMBEDDINGS = {
        "name": "financial_reports_embeddings",
        "vector_size": 768,  # Vietnamese SBERT dimension
        "distance": "Cosine",
        "description": "Lưu embeddings của các báo cáo tài chính đã được chunk và embed",
        "fields": [
            "symbol",           # Mã cổ phiếu
            "fiscal_year",      # Năm tài chính
            "fiscal_quarter",  # Quý (nếu có)
            "period_type",      # Loại báo cáo (Q/Y)
            "chunk_type",       # Loại chunk (summary, income_statement, balance_sheet, cash_flow, ratios)
            "text",             # Nội dung text đã được chunk
            "created_at",       # Thời gian tạo
            # Financial metrics for filtering
            "revenue",          # Doanh thu
            "net_profit",       # Lợi nhuận ròng
            "eps",              # Thu nhập/cổ phiếu
            "roe",              # ROE
            "roa",              # ROA
            "total_assets",     # Tổng tài sản
            "shareholders_equity" # Vốn chủ sở hữu
        ]
    }
    
    # Usage trong core_services:
    # - FinancialReportChunker tạo chunks từ raw financial data
    # - SentenceTransformer encode chunks thành vectors
    # - QdrantFinancialReportStorage lưu vectors vào Qdrant
    # - Backend sử dụng để semantic search financial reports
```

### 4. API Design

#### 4.1 RESTful API Structure
```python
# Chat API
POST /api/v1/chat/query
GET  /api/v1/chat/history/{session_id}
POST /api/v1/chat/context/update
DELETE /api/v1/chat/session/{session_id}

# Data API
GET  /api/v1/data/stock/{symbol}
GET  /api/v1/data/financial/{symbol}
GET  /api/v1/data/company/{symbol}
POST /api/v1/data/search

# Analysis API
POST /api/v1/analysis/indicators
POST /api/v1/analysis/compare
GET  /api/v1/analysis/top-movers
POST /api/v1/analysis/trends

# Prediction API
POST /api/v1/prediction/price-trend
POST /api/v1/prediction/signals
POST /api/v1/prediction/portfolio
GET  /api/v1/prediction/models

# Visualization API
POST /api/v1/charts/generate
POST /api/v1/charts/compare
GET  /api/v1/charts/templates
POST /api/v1/charts/export

# Admin API
GET  /api/v1/admin/users
POST /api/v1/admin/users
PUT  /api/v1/admin/users/{user_id}
DELETE /api/v1/admin/users/{user_id}
GET  /api/v1/admin/config
POST /api/v1/admin/config
GET  /api/v1/admin/logs
GET  /api/v1/admin/metrics
```

#### 4.2 Qdrant Vector Search API
```python
# Vector Search API - Chỉ cho financial reports từ core_services
POST /api/v1/vector/search/financial-reports
GET  /api/v1/vector/collections/financial_reports_embeddings/info
POST /api/v1/vector/collections/financial_reports_embeddings/upsert
POST /api/v1/vector/collections/financial_reports_embeddings/delete
```

#### 4.3 WebSocket API
```python
# Real-time updates
ws://localhost:8000/ws/chat/{session_id}
ws://localhost:8000/ws/admin/notifications
ws://localhost:8000/ws/data/updates
```

---

## 🎨 FRONTEND ARCHITECTURE

### 1. Component Structure

#### 1.1 Chat Interface
```typescript
interface ChatInterface {
  // Main chat components
  ChatContainer: React.FC
  MessageList: React.FC<{ messages: Message[] }>
  MessageInput: React.FC<{ onSend: (message: string) => void }>
  TypingIndicator: React.FC<{ isTyping: boolean }>
  
  // Message types
  TextMessage: React.FC<{ content: string }>
  ChartMessage: React.FC<{ chartData: ChartData }>
  TableMessage: React.FC<{ tableData: TableData }>
  MixedMessage: React.FC<{ content: MixedContent }>
}
```

#### 1.2 Chart Viewport
```typescript
interface ChartViewport {
  // Chart components
  ChartContainer: React.FC<{ chartType: string }>
  CandlestickChart: React.FC<{ data: StockData[] }>
  LineChart: React.FC<{ data: TimeSeriesData[] }>
  ComparisonChart: React.FC<{ data: ComparisonData }>
  VolumeChart: React.FC<{ data: VolumeData[] }>
  
  // Chart controls
  ChartControls: React.FC<{ onExport: () => void }>
  TimeframeSelector: React.FC<{ onTimeframeChange: (tf: string) => void }>
  IndicatorSelector: React.FC<{ onIndicatorChange: (ind: string[]) => void }>
}
```

#### 1.3 Authentication
```typescript
interface AuthComponents {
  LoginForm: React.FC<{ onLogin: (credentials: LoginCredentials) => void }>
  RegisterForm: React.FC<{ onRegister: (userData: UserData) => void }>
  ForgotPasswordForm: React.FC<{ onReset: (email: string) => void }>
  ProfileSettings: React.FC<{ user: User }>
}
```

#### 1.4 Admin Dashboard
```typescript
interface AdminDashboard {
  // User management
  UserManagement: React.FC
  UserList: React.FC<{ users: User[] }>
  UserForm: React.FC<{ user?: User }>
  
  // Configuration
  ConfigManagement: React.FC
  CrawlConfig: React.FC<{ config: CrawlConfig }>
  SystemSettings: React.FC<{ settings: SystemSettings }>
  
  // Monitoring
  SystemMonitoring: React.FC
  MetricsDashboard: React.FC<{ metrics: SystemMetrics }>
  LogViewer: React.FC<{ logs: LogEntry[] }>
}
```

### 2. State Management

#### 2.1 Redux Store Structure
```typescript
interface RootState {
  auth: {
    user: User | null
    isAuthenticated: boolean
    isLoading: boolean
  }
  
  chat: {
    currentSession: string | null
    messages: Message[]
    isTyping: boolean
    context: ConversationContext
  }
  
  charts: {
    activeChart: ChartData | null
    chartHistory: ChartData[]
    viewportVisible: boolean
  }
  
  admin: {
    users: User[]
    config: SystemConfig
    metrics: SystemMetrics
    logs: LogEntry[]
  }
}
```

#### 2.2 API Integration
```typescript
// API service layer
class ApiService {
  // Chat API
  async sendMessage(message: string, sessionId: string): Promise<ChatResponse>
  async getChatHistory(sessionId: string): Promise<Message[]>
  async updateContext(sessionId: string, context: ConversationContext): Promise<void>
  
  // Data API
  async getStockData(symbol: string, timeframe: string): Promise<StockData[]>
  async getFinancialData(symbol: string, period: string): Promise<FinancialData>
  async searchCompany(query: string): Promise<CompanyInfo[]>
  
  // Analysis API
  async calculateIndicators(symbol: string, indicators: string[]): Promise<IndicatorData>
  async compareStocks(symbols: string[], period: string): Promise<ComparisonData>
  async getTopMovers(period: string, limit: number): Promise<TopMoverData[]>
  
  // Prediction API
  async predictPriceTrend(symbol: string, modelVersion: string): Promise<PredictionData>
  async getBuySellSignals(symbol: string, modelVersion: string): Promise<SignalData>
  async getPortfolioRecommendations(criteria: PortfolioCriteria): Promise<PortfolioData>
  
  // Visualization API
  async generateChart(data: ChartRequest): Promise<ChartData>
  async exportChart(chartId: string, format: string): Promise<Blob>
}
```

### 3. UI/UX Design

#### 3.1 Design System
```typescript
// Color palette
const colors = {
  primary: '#1976d2',
  secondary: '#dc004e',
  success: '#2e7d32',
  warning: '#ed6c02',
  error: '#d32f2f',
  info: '#0288d1',
  
  // Chart colors
  chart: {
    bullish: '#00c853',
    bearish: '#ff1744',
    neutral: '#757575',
    volume: '#2196f3'
  }
}

// Typography
const typography = {
  fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  h1: { fontSize: '2.5rem', fontWeight: 300 },
  h2: { fontSize: '2rem', fontWeight: 300 },
  h3: { fontSize: '1.75rem', fontWeight: 400 },
  h4: { fontSize: '1.5rem', fontWeight: 400 },
  h5: { fontSize: '1.25rem', fontWeight: 400 },
  h6: { fontSize: '1rem', fontWeight: 500 },
  body1: { fontSize: '1rem', fontWeight: 400 },
  body2: { fontSize: '0.875rem', fontWeight: 400 },
  caption: { fontSize: '0.75rem', fontWeight: 400 }
}

// Spacing
const spacing = {
  xs: '4px',
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '32px',
  xxl: '48px'
}
```

#### 3.2 Responsive Design
```typescript
// Breakpoints
const breakpoints = {
  xs: '0px',
  sm: '600px',
  md: '960px',
  lg: '1280px',
  xl: '1920px'
}

// Layout components
const Layout = {
  Mobile: '100% width, single column',
  Tablet: '768px max-width, two columns',
  Desktop: '1200px max-width, three columns',
  Large: '1920px max-width, four columns'
}
```

---

## 🔬 CORE SERVICES ARCHITECTURE

### 1. Data Ingestion Pipeline

#### 1.1 Data Sources
```python
class DataSources:
    # Stock data sources
    VNSTOCK_API = "https://api.vnstock.vn"
    FINNHUB_API = "https://finnhub.io/api"
    ALPHA_VANTAGE_API = "https://www.alphavantage.co"
    
    # Financial data sources
    BCTC_SOURCES = [
        "https://bctc.vn",
        "https://finance.vietstock.vn",
        "https://cafef.vn"
    ]
    
    # News and sentiment
    NEWS_SOURCES = [
        "https://vnexpress.net",
        "https://tuoitre.vn",
        "https://thanhnien.vn"
    ]
```

#### 1.2 Data Processing Pipeline
```python
class DataProcessingPipeline:
    def __init__(self):
        self.stages = [
            DataValidation(),
            DataNormalization(),
            DataQualityCheck(),
            DataEnrichment(),
            DataStorage()
        ]
    
    def process(self, raw_data: dict) -> ProcessedData:
        for stage in self.stages:
            raw_data = stage.process(raw_data)
        return raw_data
```

### 2. ML Models Architecture

#### 2.1 Model Registry
```python
class ModelRegistry:
    def __init__(self):
        self.models = {
            'price_prediction': {
                'lstm': LSTMPricePredictor(),
                'gru': GRUPricePredictor(),
                'transformer': TransformerPredictor()
            },
            'signal_generation': {
                'ensemble': EnsembleSignalGenerator(),
                'xgboost': XGBoostSignalGenerator()
            },
            'sentiment_analysis': {
                'bert': BERTSentimentAnalyzer(),
                'roberta': RoBERTaSentimentAnalyzer()
            }
        }
    
    def get_model(self, task: str, model_type: str):
        return self.models[task][model_type]
```

#### 2.2 Model Training Pipeline
```python
class ModelTrainingPipeline:
    def __init__(self):
        self.stages = [
            DataPreprocessing(),
            FeatureEngineering(),
            ModelTraining(),
            ModelValidation(),
            ModelDeployment()
        ]
    
    def train(self, training_data: Dataset) -> TrainedModel:
        for stage in self.stages:
            training_data = stage.process(training_data)
        return training_data
```

### 3. Indicators and Analysis

#### 3.1 Technical Indicators
```python
class TechnicalIndicators:
    def __init__(self):
        self.indicators = {
            'trend': [
                'SMA', 'EMA', 'MACD', 'ADX', 'Parabolic_SAR'
            ],
            'momentum': [
                'RSI', 'Stochastic', 'Williams_R', 'CCI', 'ROC'
            ],
            'volume': [
                'OBV', 'VWAP', 'Volume_SMA', 'Volume_Ratio'
            ],
            'volatility': [
                'Bollinger_Bands', 'ATR', 'Keltner_Channels'
            ],
            'support_resistance': [
                'Pivot_Points', 'Fibonacci_Retracement'
            ]
        }
    
    def calculate(self, symbol: str, indicator: str, period: int) -> IndicatorData:
        # Implementation
        pass
```

#### 3.2 Fundamental Analysis
```python
class FundamentalAnalysis:
    def __init__(self):
        self.metrics = {
            'profitability': [
                'ROE', 'ROA', 'ROIC', 'Gross_Margin', 'Net_Margin'
            ],
            'liquidity': [
                'Current_Ratio', 'Quick_Ratio', 'Cash_Ratio'
            ],
            'leverage': [
                'Debt_to_Equity', 'Debt_to_Assets', 'Interest_Coverage'
            ],
            'efficiency': [
                'Asset_Turnover', 'Inventory_Turnover', 'Receivables_Turnover'
            ]
        }
    
    def analyze(self, symbol: str, period: str) -> FundamentalData:
        # Implementation
        pass
```

---

## 🔄 WORKFLOW PROCESSES

### 1. Chat Query Processing Workflow

```mermaid
graph TD
    A[User Query] --> B[OrchestrationAgent]
    B --> C[Context Analysis]
    C --> D[PlanningAgent Call]
    D --> E[Plan Generation]
    E --> F[Tool Execution]
    F --> G[Data Collection]
    G --> H{Plan Complete?}
    H -->|No| I[Additional PlanningAgent Call]
    I --> F
    H -->|Yes| J[ResponseGenerator]
    J --> K[Response Formatting]
    K --> L[User Response]
```

### 2. Data Ingestion Workflow

```mermaid
graph TD
    A[Data Source] --> B[Data Extraction]
    B --> C[Data Validation]
    C --> D[Data Normalization]
    D --> E[Data Quality Check]
    E --> F[Data Enrichment]
    F --> G[Database Storage]
    G --> H[Cache Update]
    H --> I[Notification]
```

### 3. ML Prediction Workflow

```mermaid
graph TD
    A[Prediction Request] --> B[Data Retrieval]
    B --> C[Feature Engineering]
    C --> D[Model Selection]
    D --> E[Prediction Generation]
    E --> F[Confidence Scoring]
    F --> G[Result Validation]
    G --> H[Response Formatting]
    H --> I[User Response]
```

---

## 🔒 SECURITY & AUTHENTICATION

### 1. Authentication Flow

```mermaid
graph TD
    A[User Login] --> B[Credential Validation]
    B --> C[JWT Token Generation]
    C --> D[Token Storage]
    D --> E[API Request]
    E --> F[Token Validation]
    F --> G[Authorization Check]
    G --> H[Resource Access]
```

### 2. Security Measures

#### 2.1 API Security
```python
class SecurityConfig:
    # Rate limiting
    RATE_LIMIT = "100 requests per minute"
    
    # Input validation
    INPUT_VALIDATION = {
        'max_query_length': 1000,
        'allowed_characters': r'[a-zA-Z0-9\s.,!?-]',
        'sql_injection_protection': True
    }
    
    # CORS configuration
    CORS_ORIGINS = [
        "http://localhost:3000",
        "https://yourdomain.com"
    ]
    
    # HTTPS enforcement
    HTTPS_ONLY = True
    HSTS_HEADERS = True
```

#### 2.2 Data Protection
```python
class DataProtection:
    # Encryption
    ENCRYPTION_KEY = "your-encryption-key"
    ENCRYPTION_ALGORITHM = "AES-256-GCM"
    
    # Data masking
    SENSITIVE_FIELDS = [
        'password', 'email', 'phone', 'ssn'
    ]
    
    # Audit logging
    AUDIT_LOG_ENABLED = True
    AUDIT_LOG_RETENTION = "90 days"
```

---

## 📊 MONITORING & OBSERVABILITY

### 1. Metrics Collection

#### 1.1 Application Metrics
```python
class ApplicationMetrics:
    # Performance metrics
    RESPONSE_TIME = "response_time_seconds"
    THROUGHPUT = "requests_per_second"
    ERROR_RATE = "error_rate_percentage"
    
    # Business metrics
    USER_ENGAGEMENT = "active_users_count"
    QUERY_SUCCESS_RATE = "query_success_rate"
    CHART_GENERATION_TIME = "chart_generation_seconds"
    
    # System metrics
    CPU_USAGE = "cpu_usage_percentage"
    MEMORY_USAGE = "memory_usage_bytes"
    DISK_USAGE = "disk_usage_bytes"
```

#### 1.2 Custom Metrics
```python
class CustomMetrics:
    # Chat metrics
    CHAT_SESSIONS_ACTIVE = "chat_sessions_active"
    MESSAGES_PER_SESSION = "messages_per_session"
    AVERAGE_RESPONSE_TIME = "average_response_time"
    
    # ML metrics
    MODEL_PREDICTION_ACCURACY = "model_prediction_accuracy"
    MODEL_INFERENCE_TIME = "model_inference_time"
    FEATURE_IMPORTANCE_SCORE = "feature_importance_score"
    
    # Data metrics
    DATA_FRESHNESS = "data_freshness_hours"
    DATA_QUALITY_SCORE = "data_quality_score"
    CACHE_HIT_RATE = "cache_hit_rate"
```

### 2. Logging Strategy

#### 2.1 Log Levels
```python
class LogLevels:
    DEBUG = "DEBUG"      # Detailed information for debugging
    INFO = "INFO"        # General information about system operation
    WARNING = "WARNING"  # Warning messages for potential issues
    ERROR = "ERROR"      # Error messages for system failures
    CRITICAL = "CRITICAL" # Critical errors that require immediate attention
```

#### 2.2 Log Structure
```python
class LogEntry:
    timestamp: datetime
    level: str
    service: str
    component: str
    message: str
    context: dict
    user_id: str
    session_id: str
    request_id: str
    duration: float
    status_code: int
```

### 3. Alerting System

#### 3.1 Alert Rules
```python
class AlertRules:
    # Performance alerts
    HIGH_RESPONSE_TIME = {
        'metric': 'response_time_seconds',
        'threshold': 5.0,
        'severity': 'WARNING'
    }
    
    # Error alerts
    HIGH_ERROR_RATE = {
        'metric': 'error_rate_percentage',
        'threshold': 5.0,
        'severity': 'CRITICAL'
    }
    
    # System alerts
    HIGH_CPU_USAGE = {
        'metric': 'cpu_usage_percentage',
        'threshold': 80.0,
        'severity': 'WARNING'
    }
    
    # Business alerts
    LOW_USER_ENGAGEMENT = {
        'metric': 'active_users_count',
        'threshold': 10,
        'severity': 'INFO'
    }
```

---

## 🚀 DEPLOYMENT & INFRASTRUCTURE

### 1. Containerization

#### 1.1 Docker Configuration
```dockerfile
# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

#### 1.2 Docker Compose
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/stockbot
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - db
      - qdrant

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=stockbot
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  qdrant_data:
```

### 2. Kubernetes Deployment

#### 2.1 Backend Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: stockbot/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

#### 2.2 Frontend Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: stockbot/frontend:latest
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
```

### 3. CI/CD Pipeline

#### 3.1 GitHub Actions
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=./ --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t stockbot/backend:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push stockbot/backend:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/backend backend=stockbot/backend:${{ github.sha }}
        kubectl rollout status deployment/backend
```

---

## 📈 PERFORMANCE OPTIMIZATION

### 1. Caching Strategy

#### 1.1 Application-Level Caching
```python
class ApplicationCache:
    def __init__(self):
        self.cache = {}
        self.ttl = {
            'stock_data': 3600,      # 1 hour
            'financial_data': 86400, # 24 hours
            'company_info': 604800,  # 7 days
            'ml_predictions': 3600,  # 1 hour
            'vector_search': 1800    # 30 minutes
        }
    
    def get_cache_key(self, service: str, params: dict) -> str:
        return f"{service}:{hash(str(params))}"
    
    def cache_data(self, key: str, data: any, ttl: int = 3600):
        # Implementation
        pass
```

#### 1.2 Qdrant Vector Caching
```python
class VectorCache:
    def __init__(self):
        self.vector_cache = {}
        self.similarity_threshold = 0.85
    
    def cache_vector_search(self, query_vector: list, results: list):
        # Cache vector search results
        pass
    
#### 1.3 Cache Invalidation
```python
class CacheInvalidation:
    def __init__(self):
        self.invalidation_rules = {
            'stock_data': 'TTL: 1 hour',
            'financial_data': 'TTL: 24 hours',
            'company_info': 'TTL: 7 days',
            'ml_predictions': 'TTL: 1 hour',
            'vector_search': 'TTL: 30 minutes'
        }
    
    def invalidate(self, pattern: str):
        # Implementation
        pass
    
    def invalidate_vector_cache(self, collection: str, symbol: str = None):
        # Invalidate vector cache for specific collection/symbol
        pass
```

### 2. Database Optimization

#### 2.1 Indexing Strategy
```sql
-- Stock data indexes
CREATE INDEX idx_stock_data_symbol_date ON stock_data(symbol, date);
CREATE INDEX idx_stock_data_date ON stock_data(date);

-- Financial data indexes
CREATE INDEX idx_financial_data_symbol_period ON financial_data(symbol, period, year);
CREATE INDEX idx_financial_data_year ON financial_data(year);

-- Chat data indexes
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at);
```

#### 2.2 Query Optimization
```python
class QueryOptimization:
    def __init__(self):
        self.optimization_rules = {
            'use_indexes': True,
            'limit_results': True,
            'avoid_select_star': True,
            'use_prepared_statements': True,
            'connection_pooling': True
        }
    
    def optimize_query(self, query: str) -> str:
        # Implementation
        pass
```

### 3. API Performance

#### 3.1 Response Optimization
```python
class ResponseOptimization:
    def __init__(self):
        self.optimization_techniques = {
            'compression': 'gzip',
            'pagination': True,
            'field_selection': True,
            'response_caching': True,
            'async_processing': True
        }
    
    def optimize_response(self, data: dict) -> dict:
        # Implementation
        pass
```

---

## 🔧 CONFIGURATION MANAGEMENT

### 1. Environment Configuration

#### 1.1 Configuration Files
```python
# config/development.py
class DevelopmentConfig:
    DEBUG = True
    DATABASE_URL = "postgresql://user:pass@localhost:5432/stockbot_dev"
    REDIS_URL = "redis://localhost:6379"
    GEMINI_API_KEY = "your-dev-api-key"
    LOG_LEVEL = "DEBUG"

# config/production.py
class ProductionConfig:
    DEBUG = False
    DATABASE_URL = "postgresql://user:pass@prod-db:5432/stockbot"
    REDIS_URL = "redis://prod-redis:6379"
    GEMINI_API_KEY = "your-prod-api-key"
    LOG_LEVEL = "INFO"
```

#### 1.2 Configuration Validation
```python
class ConfigValidation:
    def __init__(self):
        self.required_fields = [
            'DATABASE_URL',
            'REDIS_URL',
            'GEMINI_API_KEY',
            'SECRET_KEY'
        ]
    
    def validate(self, config: dict) -> bool:
        for field in self.required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        return True
```

### 2. Feature Flags

#### 2.1 Feature Toggle System
```python
class FeatureFlags:
    def __init__(self):
        self.flags = {
            'enable_ml_predictions': True,
            'enable_real_time_data': False,
            'enable_advanced_charts': True,
            'enable_voice_interface': False,
            'enable_mobile_app': False
        }
    
    def is_enabled(self, feature: str) -> bool:
        return self.flags.get(feature, False)
    
    def toggle_feature(self, feature: str, enabled: bool):
        self.flags[feature] = enabled
```

---

## 📚 TESTING STRATEGY

### 1. Testing Pyramid

#### 1.1 Unit Tests
```python
class UnitTests:
    def test_data_validation(self):
        # Test data validation logic
        pass
    
    def test_indicator_calculation(self):
        # Test technical indicator calculations
        pass
    
    def test_ml_model_prediction(self):
        # Test ML model predictions
        pass
```

#### 1.2 Integration Tests
```python
class IntegrationTests:
    def test_api_endpoints(self):
        # Test API endpoint integration
        pass
    
    def test_database_operations(self):
        # Test database integration
        pass
    
    def test_external_api_calls(self):
        # Test external API integration
        pass
```

#### 1.3 End-to-End Tests
```python
class E2ETests:
    def test_chat_workflow(self):
        # Test complete chat workflow
        pass
    
    def test_chart_generation(self):
        # Test chart generation workflow
        pass
    
    def test_admin_dashboard(self):
        # Test admin dashboard functionality
        pass
```

### 2. Performance Testing

#### 2.1 Load Testing
```python
class LoadTesting:
    def __init__(self):
        self.test_scenarios = {
            'normal_load': {
                'users': 100,
                'duration': '10 minutes',
                'ramp_up': '2 minutes'
            },
            'peak_load': {
                'users': 500,
                'duration': '5 minutes',
                'ramp_up': '1 minute'
            },
            'stress_test': {
                'users': 1000,
                'duration': '3 minutes',
                'ramp_up': '30 seconds'
            }
        }
    
    def run_load_test(self, scenario: str):
        # Implementation
        pass
```

---

## 🚀 DEPLOYMENT ROADMAP

### Phase 1: MVP (2-3 tháng)
- [ ] Basic chat interface
- [ ] Core data retrieval APIs
- [ ] Simple chart generation
- [ ] Basic authentication
- [ ] PostgreSQL database setup
- [ ] Basic ML predictions

### Phase 2: Enhancement (2-3 tháng)
- [ ] Advanced ML predictions
- [ ] Complex chart types
- [ ] Admin dashboard
- [ ] Performance optimization
- [ ] Caching implementation
- [ ] Error handling improvements

### Phase 3: Scale (2-3 tháng)
- [ ] Real-time data streaming
- [ ] Advanced analytics
- [ ] Mobile app
- [ ] Enterprise features
- [ ] Multi-tenant support
- [ ] Advanced security features

### Phase 4: Innovation (3-4 tháng)
- [ ] Voice interface
- [ ] AI-powered insights
- [ ] Social features
- [ ] Integration with broker APIs
- [ ] Advanced portfolio management
- [ ] Machine learning model improvements

---

## 📊 SUCCESS METRICS

### 1. Technical Metrics
- **Response Time**: < 2 seconds for 95% of requests
- **Uptime**: 99.9% availability
- **Error Rate**: < 1% of all requests
- **Throughput**: 1000+ requests per second

### 2. Business Metrics
- **User Engagement**: 80%+ daily active users
- **Query Success Rate**: 95%+ successful responses
- **User Satisfaction**: 4.5+ stars rating
- **Revenue Growth**: 20%+ month-over-month

### 3. ML Metrics
- **Prediction Accuracy**: 70%+ for price predictions
- **Signal Quality**: 60%+ profitable signals
- **Model Performance**: < 100ms inference time
- **Data Quality**: 99%+ data accuracy

---

## 🔮 FUTURE ENHANCEMENTS

### 1. Advanced AI Features
- **Multi-modal AI**: Text, voice, and image processing
- **Personalized Recommendations**: User-specific investment advice
- **Sentiment Analysis**: Real-time market sentiment tracking
- **Risk Assessment**: Advanced risk modeling

### 2. Integration Opportunities
- **Broker APIs**: Direct trading integration
- **News APIs**: Real-time news integration
- **Social Media**: Sentiment analysis from social platforms
- **Economic Data**: Macro-economic indicators

### 3. Platform Expansion
- **Mobile App**: Native iOS and Android apps
- **Web App**: Progressive Web App (PWA)
- **Desktop App**: Electron-based desktop application
- **API Platform**: Third-party developer API

---

## 📝 CONCLUSION

Thiết kế hệ thống này cung cấp một foundation mạnh mẽ cho việc xây dựng Bot chat tư vấn cổ phiếu. Với kiến trúc modular, scalable và maintainable, hệ thống có thể phát triển từ MVP đến một platform hoàn chỉnh.

**Điểm mạnh chính:**
- Kiến trúc Master-Slave rõ ràng và dễ maintain
- Separation of concerns tốt
- Scalability cao với microservices
- Security và performance được quan tâm từ đầu

**Khuyến nghị tiếp theo:**
1. Bắt đầu với Phase 1 (MVP) để validate concept
2. Focus vào user experience và data quality
3. Implement monitoring và alerting sớm
4. Plan cho scalability từ đầu

Hệ thống này có tiềm năng trở thành một platform tư vấn đầu tư hàng đầu tại Việt Nam với khả năng mở rộng ra thị trường khu vực.
