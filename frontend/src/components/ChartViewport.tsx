import React from 'react'
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Tooltip,
} from '@mui/material'
import {
  Close as CloseIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material'
import { TechnicalChart } from '@/components/Charts'

interface ChartViewportProps {
  isVisible: boolean
  onClose: () => void
  symbol?: string
  data?: any[]
}

const ChartViewport: React.FC<ChartViewportProps> = ({
  isVisible,
  onClose,
  symbol = 'VN-INDEX',
  data = []
}) => {
  if (!isVisible) return null

  return (
    <Paper
      sx={{
        position: 'fixed',
        top: 0,
        right: 0,
        width: '50%',
        height: '100vh',
        zIndex: 1300,
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '-4px 0 20px rgba(0,0,0,0.15)',
        borderRadius: 0,
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          backgroundColor: 'background.paper',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TrendingUpIcon color="primary" />
          <Typography variant="h6" component="h2">
            Biểu đồ {symbol}
          </Typography>
        </Box>
        
        <Tooltip title="Đóng biểu đồ">
          <IconButton
            onClick={onClose}
            size="small"
            sx={{
              bgcolor: 'grey.100',
              '&:hover': {
                bgcolor: 'grey.200',
              },
            }}
          >
            <CloseIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Chart Content */}
      <Box
        sx={{
          flexGrow: 1,
          p: 2,
          overflow: 'auto',
          backgroundColor: 'background.default',
        }}
      >
        <TechnicalChart
          symbol={symbol}
          data={data}
          onTimeframeChange={(timeframe) => {
            console.log('Timeframe changed to:', timeframe)
          }}
        />
      </Box>
    </Paper>
  )
}

export default ChartViewport
