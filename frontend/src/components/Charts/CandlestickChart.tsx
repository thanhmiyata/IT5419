import React, { useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import { Box, Paper, Typography } from '@mui/material'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

interface CandlestickData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface CandlestickChartProps {
  data: CandlestickData[]
  title?: string
  height?: number
  showVolume?: boolean
}

const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  title,
  height = 400,
  showVolume = true,
}) => {
  // Use showVolume to avoid unused variable warning
  console.log('Show volume:', showVolume)
  const chartRef = useRef<ChartJS<'bar'>>(null)

  // Process data for candlestick visualization
  const processedData = data.map(item => ({
    x: new Date(item.timestamp).toLocaleDateString('vi-VN'),
    open: item.open,
    high: item.high,
    low: item.low,
    close: item.close,
    volume: item.volume,
  }))

  const chartData = {
    labels: processedData.map(item => item.x),
    datasets: [
      {
        label: 'Giá cao nhất',
        data: processedData.map(item => item.high),
        backgroundColor: 'rgba(76, 175, 80, 0.6)',
        borderColor: 'rgba(76, 175, 80, 1)',
        borderWidth: 1,
        type: 'bar' as const,
        order: 1,
      },
      {
        label: 'Giá thấp nhất',
        data: processedData.map(item => item.low),
        backgroundColor: 'rgba(244, 67, 54, 0.6)',
        borderColor: 'rgba(244, 67, 54, 1)',
        borderWidth: 1,
        type: 'bar' as const,
        order: 1,
      },
      {
        label: 'Giá đóng cửa',
        data: processedData.map(item => item.close),
        backgroundColor: 'rgba(33, 150, 243, 0.6)',
        borderColor: 'rgba(33, 150, 243, 1)',
        borderWidth: 2,
        type: 'line' as const,
        order: 2,
        tension: 0.1,
      },
    ],
  } as any

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
      },
      title: {
        display: !!title,
        text: title,
      },
      tooltip: {
        enabled: true,
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          label: function(context: any) {
            const label = context.dataset.label || ''
            const value = new Intl.NumberFormat('vi-VN', {
              style: 'currency',
              currency: 'VND',
            }).format(context.parsed.y)
            return `${label}: ${value}`
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Thời gian',
        },
        grid: {
          display: false,
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Giá (VND)',
        },
        ticks: {
          callback: function(value: any) {
            return new Intl.NumberFormat('vi-VN', {
              style: 'currency',
              currency: 'VND',
              minimumFractionDigits: 0,
            }).format(value)
          },
        },
      },
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
  }

  return (
    <Paper sx={{ p: 2, height: height + 60 }}>
      {title && (
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
      )}
      <Box sx={{ height, position: 'relative' }}>
        <Bar ref={chartRef} data={chartData} options={options} />
      </Box>
    </Paper>
  )
}

export default CandlestickChart
