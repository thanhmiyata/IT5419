import { apiService } from './api'
import { StockData, StockChartData, StockAnalysis, ApiResponse } from '@/types'

class StockService {
  async getStockData(symbol: string): Promise<StockData> {
    const response = await apiService.get<StockData>(`/stocks/${symbol}`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to get stock data')
    }
    return response.data
  }

  async getStockChart(symbol: string, timeframe: string): Promise<StockChartData> {
    const response = await apiService.get<StockChartData>(`/stocks/${symbol}/chart`, {
      params: { timeframe },
    })
    if (!response.success) {
      throw new Error(response.error || 'Failed to get chart data')
    }
    return response.data
  }

  async getStockAnalysis(symbol: string): Promise<StockAnalysis> {
    const response = await apiService.get<StockAnalysis>(`/stocks/${symbol}/analysis`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to get stock analysis')
    }
    return response.data
  }

  async searchStocks(query: string): Promise<StockData[]> {
    const response = await apiService.get<StockData[]>('/stocks/search', {
      params: { q: query },
    })
    if (!response.success) {
      throw new Error(response.error || 'Failed to search stocks')
    }
    return response.data
  }

  async getTopGainers(): Promise<StockData[]> {
    const response = await apiService.get<StockData[]>('/stocks/top-gainers')
    if (!response.success) {
      throw new Error(response.error || 'Failed to get top gainers')
    }
    return response.data
  }

  async getTopLosers(): Promise<StockData[]> {
    const response = await apiService.get<StockData[]>('/stocks/top-losers')
    if (!response.success) {
      throw new Error(response.error || 'Failed to get top losers')
    }
    return response.data
  }

  async getMostActive(): Promise<StockData[]> {
    const response = await apiService.get<StockData[]>('/stocks/most-active')
    if (!response.success) {
      throw new Error(response.error || 'Failed to get most active stocks')
    }
    return response.data
  }

  async getMarketOverview(): Promise<{
    totalStocks: number;
    gainers: number;
    losers: number;
    unchanged: number;
    totalVolume: number;
    totalValue: number;
  }> {
    const response = await apiService.get('/stocks/market-overview')
    if (!response.success) {
      throw new Error(response.error || 'Failed to get market overview')
    }
    return response.data
  }

  async getWatchlist(): Promise<string[]> {
    const response = await apiService.get<string[]>('/stocks/watchlist')
    if (!response.success) {
      throw new Error(response.error || 'Failed to get watchlist')
    }
    return response.data
  }

  async addToWatchlist(symbol: string): Promise<{ symbol: string }> {
    const response = await apiService.post<{ symbol: string }>('/stocks/watchlist', { symbol })
    if (!response.success) {
      throw new Error(response.error || 'Failed to add to watchlist')
    }
    return response.data
  }

  async removeFromWatchlist(symbol: string): Promise<{ symbol: string }> {
    const response = await apiService.delete<{ symbol: string }>(`/stocks/watchlist/${symbol}`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to remove from watchlist')
    }
    return response.data
  }

  async getWatchlistData(): Promise<StockData[]> {
    const response = await apiService.get<StockData[]>('/stocks/watchlist/data')
    if (!response.success) {
      throw new Error(response.error || 'Failed to get watchlist data')
    }
    return response.data
  }

  async getSectorPerformance(): Promise<{
    sector: string;
    change: number;
    changePercent: number;
    volume: number;
  }[]> {
    const response = await apiService.get('/stocks/sectors/performance')
    if (!response.success) {
      throw new Error(response.error || 'Failed to get sector performance')
    }
    return response.data
  }

  async getCompanyInfo(symbol: string): Promise<{
    symbol: string;
    name: string;
    sector: string;
    industry: string;
    description: string;
    website: string;
    employees: number;
    marketCap: number;
    pe: number;
    eps: number;
    dividend: number;
    dividendYield: number;
  }> {
    const response = await apiService.get(`/stocks/${symbol}/company`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to get company info')
    }
    return response.data
  }

  async getFinancials(symbol: string): Promise<{
    revenue: number[];
    netIncome: number[];
    assets: number[];
    liabilities: number[];
    equity: number[];
    years: string[];
  }> {
    const response = await apiService.get(`/stocks/${symbol}/financials`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to get financials')
    }
    return response.data
  }

  async getNews(symbol: string, limit: number = 10): Promise<{
    title: string;
    summary: string;
    url: string;
    publishedAt: string;
    source: string;
    sentiment: 'positive' | 'negative' | 'neutral';
  }[]> {
    const response = await apiService.get(`/stocks/${symbol}/news`, {
      params: { limit },
    })
    if (!response.success) {
      throw new Error(response.error || 'Failed to get news')
    }
    return response.data
  }
}

export const stockService = new StockService()
