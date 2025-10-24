// API endpoints
export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh',
    PROFILE: '/auth/profile',
    CHANGE_PASSWORD: '/auth/change-password',
    FORGOT_PASSWORD: '/auth/forgot-password',
    RESET_PASSWORD: '/auth/reset-password',
  },
  CHAT: {
    SESSIONS: '/chat/sessions',
    MESSAGES: '/chat/sessions/:sessionId/messages',
    SEARCH: '/chat/search',
  },
  STOCKS: {
    LIST: '/stocks',
    DETAIL: '/stocks/:symbol',
    CHART: '/stocks/:symbol/chart',
    ANALYSIS: '/stocks/:symbol/analysis',
    SEARCH: '/stocks/search',
    WATCHLIST: '/stocks/watchlist',
    TOP_GAINERS: '/stocks/top-gainers',
    TOP_LOSERS: '/stocks/top-losers',
    MOST_ACTIVE: '/stocks/most-active',
    MARKET_OVERVIEW: '/stocks/market-overview',
  },
} as const

// WebSocket events
export const WS_EVENTS = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  CHAT_MESSAGE: 'chat_message',
  TYPING: 'typing',
  USER_JOINED: 'user_joined',
  USER_LEFT: 'user_left',
  STOCK_UPDATE: 'stock_update',
  MARKET_UPDATE: 'market_update',
  ERROR: 'error',
} as const

// Chart timeframes
export const CHART_TIMEFRAMES = [
  { value: '1D', label: '1 Ngày' },
  { value: '1W', label: '1 Tuần' },
  { value: '1M', label: '1 Tháng' },
  { value: '3M', label: '3 Tháng' },
  { value: '6M', label: '6 Tháng' },
  { value: '1Y', label: '1 Năm' },
  { value: '2Y', label: '2 Năm' },
  { value: '5Y', label: '5 Năm' },
] as const

// Chart types
export const CHART_TYPES = {
  LINE: 'line',
  CANDLESTICK: 'candlestick',
  VOLUME: 'volume',
  AREA: 'area',
} as const

// Message types
export const MESSAGE_TYPES = {
  TEXT: 'text',
  CHART: 'chart',
  TABLE: 'table',
  ERROR: 'error',
} as const

// User roles
export const USER_ROLES = {
  USER: 'user',
  ADMIN: 'admin',
} as const

// Notification types
export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info',
} as const

// Stock analysis recommendations
export const STOCK_RECOMMENDATIONS = {
  BUY: 'buy',
  SELL: 'sell',
  HOLD: 'hold',
} as const

// Technical indicators
export const TECHNICAL_INDICATORS = {
  RSI: 'RSI',
  MACD: 'MACD',
  SMA: 'SMA',
  EMA: 'EMA',
  BOLLINGER_BANDS: 'BOLLINGER_BANDS',
  STOCHASTIC: 'STOCHASTIC',
  WILLIAMS_R: 'WILLIAMS_R',
  CCI: 'CCI',
} as const

// Market status
export const MARKET_STATUS = {
  OPEN: 'open',
  CLOSED: 'closed',
  PRE_MARKET: 'pre_market',
  AFTER_HOURS: 'after_hours',
} as const

// Error codes
export const ERROR_CODES = {
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  NOT_FOUND: 'NOT_FOUND',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  SERVER_ERROR: 'SERVER_ERROR',
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT: 'TIMEOUT',
} as const

// Local storage keys
export const STORAGE_KEYS = {
  TOKEN: 'token',
  USER: 'user',
  THEME: 'theme',
  LANGUAGE: 'language',
  SETTINGS: 'settings',
} as const

// Pagination
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100,
  DEFAULT_PAGE: 1,
} as const

// Debounce delay
export const DEBOUNCE_DELAY = {
  SEARCH: 300,
  INPUT: 500,
  API_CALL: 1000,
} as const

// Animation durations
export const ANIMATION_DURATION = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
} as const

// Breakpoints
export const BREAKPOINTS = {
  XS: 0,
  SM: 600,
  MD: 900,
  LG: 1200,
  XL: 1536,
} as const

// Z-index values
export const Z_INDEX = {
  MODAL: 1300,
  SNACKBAR: 1400,
  TOOLTIP: 1500,
  LOADING: 1600,
} as const
