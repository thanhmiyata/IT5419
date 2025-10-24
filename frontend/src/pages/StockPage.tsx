import React, { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
} from '@mui/material'
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Search as SearchIcon,
} from '@mui/icons-material'
import { StockData } from '@/types'
import LoadingSpinner from '@/components/UI/LoadingSpinner'
import { TechnicalChart } from '@/components/Charts'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`stock-tabpanel-${index}`}
      aria-labelledby={`stock-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )
}

const StockPage: React.FC = () => {
  const { symbol } = useParams<{ symbol: string }>()
  
  // Mock data for demo
  const currentStock: StockData | null = symbol ? {
    symbol: symbol,
    name: 'Demo Stock Company',
    price: 45000,
    change: 2500,
    changePercent: 5.88,
    volume: 1000000,
    marketCap: 1000000000000,
    high: 46000,
    low: 44000,
    open: 44500,
    previousClose: 42500,
    timestamp: new Date().toISOString(),
  } : null
  
  const chartData = null
  const analysis = null
  const isLoading = false
  
  const [searchSymbol, setSearchSymbol] = useState(symbol || '')
  const [tabValue, setTabValue] = useState(0)

  useEffect(() => {
    if (symbol) {
      // Mock stock data loading for demo
      console.log('Loading stock data for:', symbol)
    }
  }, [symbol])

  const handleSearch = () => {
    if (searchSymbol.trim()) {
      // Mock search for demo
      console.log('Searching for stock:', searchSymbol.trim().toUpperCase())
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('vi-VN', {
      style: 'currency',
      currency: 'VND',
    }).format(value)
  }

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('vi-VN').format(value)
  }

  const getChangeColor = (change: number) => {
    return change >= 0 ? 'success.main' : 'error.main'
  }

  const getChangeIcon = (change: number) => {
    return change >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />
  }

  if (isLoading && !currentStock) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <LoadingSpinner message="Đang tải dữ liệu cổ phiếu..." />
      </Box>
    )
  }

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Phân tích Cổ phiếu
        </Typography>
        
        {/* Search */}
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
          <TextField
            label="Mã cổ phiếu"
            value={searchSymbol}
            onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
            onKeyPress={handleKeyPress}
            placeholder="VD: VIC, VHM, MSN..."
            sx={{ minWidth: 200 }}
          />
          <Button
            variant="contained"
            startIcon={<SearchIcon />}
            onClick={handleSearch}
            disabled={!searchSymbol.trim()}
          >
            Tìm kiếm
          </Button>
        </Box>
      </Paper>

      {currentStock ? (
        <>
          {/* Stock Overview */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box>
                      <Typography variant="h4" component="h2" gutterBottom>
                        {currentStock.symbol}
                      </Typography>
                      <Typography variant="h6" color="text.secondary">
                        {currentStock.name}
                      </Typography>
                    </Box>
                    <Chip
                      label={currentStock.change >= 0 ? 'Tăng' : 'Giảm'}
                      color={currentStock.change >= 0 ? 'success' : 'error'}
                      icon={getChangeIcon(currentStock.change)}
                    />
                  </Box>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                    <Typography variant="h3" component="span">
                      {formatCurrency(currentStock.price)}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getChangeIcon(currentStock.change)}
                      <Typography
                        variant="h6"
                        sx={{ color: getChangeColor(currentStock.change) }}
                      >
                        {currentStock.change >= 0 ? '+' : ''}{formatCurrency(currentStock.change)}
                      </Typography>
                      <Typography
                        variant="h6"
                        sx={{ color: getChangeColor(currentStock.change) }}
                      >
                        ({currentStock.changePercent >= 0 ? '+' : ''}{currentStock.changePercent.toFixed(2)}%)
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Thông tin cơ bản
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell>Giá mở cửa</TableCell>
                          <TableCell align="right">{formatCurrency(currentStock.open)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Giá cao nhất</TableCell>
                          <TableCell align="right">{formatCurrency(currentStock.high)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Giá thấp nhất</TableCell>
                          <TableCell align="right">{formatCurrency(currentStock.low)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Giá đóng cửa trước</TableCell>
                          <TableCell align="right">{formatCurrency(currentStock.previousClose)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Khối lượng</TableCell>
                          <TableCell align="right">{formatNumber(currentStock.volume)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Vốn hóa</TableCell>
                          <TableCell align="right">{formatCurrency(currentStock.marketCap)}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Tabs */}
          <Paper>
            <Tabs
              value={tabValue}
              onChange={(e, newValue) => setTabValue(newValue)}
              aria-label="stock analysis tabs"
            >
              <Tab label="Biểu đồ" />
              <Tab label="Phân tích" />
              <Tab label="Tin tức" />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
              <TechnicalChart
                symbol={currentStock.symbol}
                data={chartData?.data || []}
                onTimeframeChange={(timeframe) => {
                  console.log('Timeframe changed to:', timeframe)
                }}
              />
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              {analysis ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Phân tích AI
                  </Typography>
                  <Typography variant="body1">
                    {analysis.analysis}
                  </Typography>
                </Box>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
                  <LoadingSpinner message="Đang phân tích..." />
                </Box>
              )}
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              <Typography variant="h6" gutterBottom>
                Tin tức liên quan
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Tính năng tin tức sẽ được phát triển trong phiên bản tiếp theo.
              </Typography>
            </TabPanel>
          </Paper>
        </>
      ) : (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            Chưa có dữ liệu cổ phiếu
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Hãy nhập mã cổ phiếu để xem thông tin chi tiết
          </Typography>
        </Paper>
      )}
    </Box>
  )
}

export default StockPage
