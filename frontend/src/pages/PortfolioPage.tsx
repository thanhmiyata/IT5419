import React from 'react'
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material'
import {
  Add as AddIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material'

const PortfolioPage: React.FC = () => {
  // Mock data - will be replaced with real data from Redux store
  const portfolioData = {
    totalValue: 100000000,
    totalGain: 5000000,
    totalGainPercent: 5.26,
    positions: [
      {
        symbol: 'VIC',
        name: 'Vingroup JSC',
        shares: 1000,
        avgPrice: 45000,
        currentPrice: 47500,
        marketValue: 47500000,
        gain: 2500000,
        gainPercent: 5.56,
      },
      {
        symbol: 'VHM',
        name: 'Vinhomes JSC',
        shares: 500,
        avgPrice: 60000,
        currentPrice: 58000,
        marketValue: 29000000,
        gain: -1000000,
        gainPercent: -3.33,
      },
      {
        symbol: 'MSN',
        name: 'Masan Group',
        shares: 200,
        avgPrice: 80000,
        currentPrice: 85000,
        marketValue: 17000000,
        gain: 1000000,
        gainPercent: 6.25,
      },
    ],
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

  const getGainColor = (gain: number) => {
    return gain >= 0 ? 'success.main' : 'error.main'
  }

  const getGainIcon = (gain: number) => {
    return gain >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />
  }

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" component="h1">
            Danh mục Đầu tư
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            sx={{ minWidth: 120 }}
          >
            Thêm cổ phiếu
          </Button>
        </Box>
        <Typography variant="body2" color="text.secondary">
          Theo dõi và quản lý danh mục đầu tư của bạn
        </Typography>
      </Paper>

      {/* Portfolio Summary */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Tổng giá trị
              </Typography>
              <Typography variant="h4" component="div">
                {formatCurrency(portfolioData.totalValue)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Lãi/Lỗ
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {getGainIcon(portfolioData.totalGain)}
                <Typography
                  variant="h4"
                  component="div"
                  sx={{ color: getGainColor(portfolioData.totalGain) }}
                >
                  {portfolioData.totalGain >= 0 ? '+' : ''}{formatCurrency(portfolioData.totalGain)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Tỷ lệ lãi/lỗ
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {getGainIcon(portfolioData.totalGain)}
                <Typography
                  variant="h4"
                  component="div"
                  sx={{ color: getGainColor(portfolioData.totalGain) }}
                >
                  {portfolioData.totalGainPercent >= 0 ? '+' : ''}{portfolioData.totalGainPercent.toFixed(2)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Positions Table */}
      <Paper>
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Vị thế đầu tư
          </Typography>
        </Box>
        
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Mã cổ phiếu</TableCell>
                <TableCell>Tên công ty</TableCell>
                <TableCell align="right">Số lượng</TableCell>
                <TableCell align="right">Giá TB</TableCell>
                <TableCell align="right">Giá hiện tại</TableCell>
                <TableCell align="right">Giá trị thị trường</TableCell>
                <TableCell align="right">Lãi/Lỗ</TableCell>
                <TableCell align="right">Tỷ lệ</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {portfolioData.positions.map((position) => (
                <TableRow key={position.symbol} hover>
                  <TableCell>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      {position.symbol}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {position.name}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    {formatNumber(position.shares)}
                  </TableCell>
                  <TableCell align="right">
                    {formatCurrency(position.avgPrice)}
                  </TableCell>
                  <TableCell align="right">
                    {formatCurrency(position.currentPrice)}
                  </TableCell>
                  <TableCell align="right">
                    {formatCurrency(position.marketValue)}
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                      {getGainIcon(position.gain)}
                      <Typography
                        variant="body2"
                        sx={{ color: getGainColor(position.gain) }}
                      >
                        {position.gain >= 0 ? '+' : ''}{formatCurrency(position.gain)}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right">
                    <Chip
                      label={`${position.gainPercent >= 0 ? '+' : ''}${position.gainPercent.toFixed(2)}%`}
                      color={position.gain >= 0 ? 'success' : 'error'}
                      size="small"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Performance Chart Placeholder */}
      <Paper sx={{ mt: 3, p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Biểu đồ hiệu suất
        </Typography>
        <Box
          sx={{
            height: 300,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'grey.50',
            borderRadius: 1,
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Biểu đồ hiệu suất danh mục sẽ được hiển thị ở đây
          </Typography>
        </Box>
      </Paper>
    </Box>
  )
}

export default PortfolioPage
