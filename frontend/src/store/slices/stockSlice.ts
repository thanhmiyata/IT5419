import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { StockData, StockChartData, StockAnalysis } from '@/types'
import { stockService } from '@/services/stockService'

interface StockState {
  currentStock: StockData | null;
  chartData: StockChartData | null;
  analysis: StockAnalysis | null;
  watchlist: string[];
  isLoading: boolean;
  error: string | null;
}

const initialState: StockState = {
  currentStock: null,
  chartData: null,
  analysis: null,
  watchlist: [],
  isLoading: false,
  error: null,
}

// Async thunks
export const getStockData = createAsyncThunk(
  'stock/getStockData',
  async (symbol: string, { rejectWithValue }) => {
    try {
      const response = await stockService.getStockData(symbol)
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to get stock data')
    }
  }
)

export const getStockChart = createAsyncThunk(
  'stock/getStockChart',
  async ({ symbol, timeframe }: { symbol: string; timeframe: string }, { rejectWithValue }) => {
    try {
      const response = await stockService.getStockChart(symbol, timeframe)
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to get chart data')
    }
  }
)

export const getStockAnalysis = createAsyncThunk(
  'stock/getStockAnalysis',
  async (symbol: string, { rejectWithValue }) => {
    try {
      const response = await stockService.getStockAnalysis(symbol)
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to get stock analysis')
    }
  }
)

export const addToWatchlist = createAsyncThunk(
  'stock/addToWatchlist',
  async (symbol: string, { rejectWithValue }) => {
    try {
      const response = await stockService.addToWatchlist(symbol)
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to add to watchlist')
    }
  }
)

export const removeFromWatchlist = createAsyncThunk(
  'stock/removeFromWatchlist',
  async (symbol: string, { rejectWithValue }) => {
    try {
      const response = await stockService.removeFromWatchlist(symbol)
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to remove from watchlist')
    }
  }
)

export const getWatchlist = createAsyncThunk(
  'stock/getWatchlist',
  async (_, { rejectWithValue }) => {
    try {
      const response = await stockService.getWatchlist()
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to get watchlist')
    }
  }
)

const stockSlice = createSlice({
  name: 'stock',
  initialState,
  reducers: {
    setCurrentStock: (state, action: PayloadAction<StockData | null>) => {
      state.currentStock = action.payload
    },
    clearError: (state) => {
      state.error = null
    },
    clearStockData: (state) => {
      state.currentStock = null
      state.chartData = null
      state.analysis = null
    },
  },
  extraReducers: (builder) => {
    builder
      // Get stock data
      .addCase(getStockData.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(getStockData.fulfilled, (state, action) => {
        state.isLoading = false
        state.currentStock = action.payload
        state.error = null
      })
      .addCase(getStockData.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })
      // Get stock chart
      .addCase(getStockChart.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(getStockChart.fulfilled, (state, action) => {
        state.isLoading = false
        state.chartData = action.payload
        state.error = null
      })
      .addCase(getStockChart.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })
      // Get stock analysis
      .addCase(getStockAnalysis.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(getStockAnalysis.fulfilled, (state, action) => {
        state.isLoading = false
        state.analysis = action.payload
        state.error = null
      })
      .addCase(getStockAnalysis.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })
      // Add to watchlist
      .addCase(addToWatchlist.fulfilled, (state, action) => {
        if (!state.watchlist.includes(action.payload.symbol)) {
          state.watchlist.push(action.payload.symbol)
        }
      })
      // Remove from watchlist
      .addCase(removeFromWatchlist.fulfilled, (state, action) => {
        state.watchlist = state.watchlist.filter(symbol => symbol !== action.payload.symbol)
      })
      // Get watchlist
      .addCase(getWatchlist.fulfilled, (state, action) => {
        state.watchlist = action.payload
      })
  },
})

export const { setCurrentStock, clearError, clearStockData } = stockSlice.actions
export default stockSlice.reducer
