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

interface VolumeData {
  timestamp: string
  volume: number
  price: number
}

interface VolumeChartProps {
  data: VolumeData[]
  title?: string
  height?: number
}

const VolumeChart: React.FC<VolumeChartProps> = ({
  data,
  title,
  height = 200,
}) => {
  const chartRef = useRef<ChartJS<'bar'>>(null)

  // Process data for volume visualization
  const processedData = data.map(item => ({
    x: new Date(item.timestamp).toLocaleDateString('vi-VN'),
    volume: item.volume,
    price: item.price,
  }))

  const chartData = {
    labels: processedData.map(item => item.x),
    datasets: [
      {
        label: 'Khối lượng giao dịch',
        data: processedData.map(item => item.volume),
        backgroundColor: processedData.map((item, index) => {
          // Color based on price movement (green for up, red for down)
          if (index === 0) return 'rgba(158, 158, 158, 0.6)'
          const currentPrice = item.price
          const previousPrice = processedData[index - 1].price
          return currentPrice >= previousPrice 
            ? 'rgba(76, 175, 80, 0.6)' 
            : 'rgba(244, 67, 54, 0.6)'
        }),
        borderColor: processedData.map((item, index) => {
          if (index === 0) return 'rgba(158, 158, 158, 1)'
          const currentPrice = item.price
          const previousPrice = processedData[index - 1].price
          return currentPrice >= previousPrice 
            ? 'rgba(76, 175, 80, 1)' 
            : 'rgba(244, 67, 54, 1)'
        }),
        borderWidth: 1,
      },
    ],
  }

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
            const value = new Intl.NumberFormat('vi-VN').format(context.parsed.y)
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
          text: 'Khối lượng',
        },
        ticks: {
          callback: function(value: any) {
            if (value >= 1000000) {
              return (value / 1000000).toFixed(1) + 'M'
            } else if (value >= 1000) {
              return (value / 1000).toFixed(1) + 'K'
            }
            return value.toString()
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

export default VolumeChart
