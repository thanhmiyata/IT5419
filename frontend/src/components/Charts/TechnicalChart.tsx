import React, { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
} from '@mui/material'
import {
  TrendingUp as TrendingUpIcon,
  ShowChart as ShowChartIcon,
  BarChart as BarChartIcon,
} from '@mui/icons-material'

import LineChart from './LineChart'
import CandlestickChart from './CandlestickChart'
import VolumeChart from './VolumeChart'

interface TechnicalChartProps {
  symbol: string
  data: any
  onTimeframeChange?: (timeframe: string) => void
}

const TechnicalChart: React.FC<TechnicalChartProps> = ({
  symbol,
  data,
  onTimeframeChange,
}) => {
  const [activeTab, setActiveTab] = useState(0)
  const [timeframe, setTimeframe] = useState('1M')

  const timeframes = [
    { value: '1D', label: '1 Ngày' },
    { value: '1W', label: '1 Tuần' },
    { value: '1M', label: '1 Tháng' },
    { value: '3M', label: '3 Tháng' },
    { value: '6M', label: '6 Tháng' },
    { value: '1Y', label: '1 Năm' },
  ]

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
  }

  const handleTimeframeChange = (event: any) => {
    const newTimeframe = event.target.value
    setTimeframe(newTimeframe)
    if (onTimeframeChange) {
      onTimeframeChange(newTimeframe)
    }
  }

  // Mock data processing - replace with real data
  const processChartData = () => {
    if (!data) return null

    const labels = data.map((item: any) => 
      new Date(item.timestamp).toLocaleDateString('vi-VN')
    )

    return {
      labels,
      datasets: [
        {
          label: 'Giá đóng cửa',
          data: data.map((item: any) => item.close),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          fill: true,
        },
      ],
    }
  }

  const chartData = processChartData()

  const tabs = [
    {
      label: 'Biểu đồ đường',
      icon: <ShowChartIcon />,
      component: (
        <LineChart
          data={chartData || { labels: [], datasets: [] }}
          title={`Biểu đồ giá ${symbol}`}
          height={400}
        />
      ),
    },
    {
      label: 'Nến Nhật',
      icon: <BarChartIcon />,
      component: (
        <CandlestickChart
          data={data || []}
          title={`Biểu đồ nến ${symbol}`}
          height={400}
        />
      ),
    },
    {
      label: 'Khối lượng',
      icon: <TrendingUpIcon />,
      component: (
        <VolumeChart
          data={data || []}
          title={`Khối lượng giao dịch ${symbol}`}
          height={200}
        />
      ),
    },
  ]

  return (
    <Paper sx={{ p: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h2">
          Biểu đồ kỹ thuật - {symbol}
        </Typography>
        
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Khung thời gian</InputLabel>
          <Select
            value={timeframe}
            label="Khung thời gian"
            onChange={handleTimeframeChange}
          >
            {timeframes.map((tf) => (
              <MenuItem key={tf.value} value={tf.value}>
                {tf.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {/* Tabs */}
      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        aria-label="chart type tabs"
        sx={{ mb: 2 }}
      >
        {tabs.map((tab, index) => (
          <Tab
            key={index}
            icon={tab.icon}
            label={tab.label}
            iconPosition="start"
          />
        ))}
      </Tabs>

      {/* Chart Content */}
      <Box>
        {tabs[activeTab].component}
      </Box>

      {/* Technical Indicators */}
      <Box sx={{ mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Chỉ báo kỹ thuật
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                RSI (14)
              </Typography>
              <Typography variant="h6" color="primary">
                65.4
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                MACD
              </Typography>
              <Typography variant="h6" color="success.main">
                +2.5
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                SMA (20)
              </Typography>
              <Typography variant="h6">
                45,250
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                EMA (12)
              </Typography>
              <Typography variant="h6">
                45,180
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  )
}

export default TechnicalChart
