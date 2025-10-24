import React from 'react'
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Avatar,
} from '@mui/material'
import {
  TrendingUp as TrendingUpIcon,
  Chat as ChatIcon,
  Assessment as AssessmentIcon,
  AccountBalance as PortfolioIcon,
} from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'

const HomePage: React.FC = () => {
  const navigate = useNavigate()

  const quickActions = [
    {
      title: 'Bắt đầu Chat',
      description: 'Tư vấn cổ phiếu thông minh với AI',
      icon: <ChatIcon sx={{ fontSize: 40 }} />,
      color: '#1976d2',
      action: () => navigate('/chat'),
    },
    {
      title: 'Phân tích Cổ phiếu',
      description: 'Xem phân tích kỹ thuật và cơ bản',
      icon: <TrendingUpIcon sx={{ fontSize: 40 }} />,
      color: '#2e7d32',
      action: () => navigate('/stock'),
    },
    {
      title: 'Quản lý Danh mục',
      description: 'Theo dõi và tối ưu hóa danh mục đầu tư',
      icon: <PortfolioIcon sx={{ fontSize: 40 }} />,
      color: '#ed6c02',
      action: () => navigate('/portfolio'),
    },
    {
      title: 'Báo cáo & Thống kê',
      description: 'Xem báo cáo hiệu suất và thống kê',
      icon: <AssessmentIcon sx={{ fontSize: 40 }} />,
      color: '#9c27b0',
      action: () => navigate('/portfolio'),
    },
  ]

  const features = [
    {
      title: 'Phân tích AI thông minh',
      description: 'Sử dụng trí tuệ nhân tạo để phân tích xu hướng thị trường và đưa ra khuyến nghị đầu tư chính xác.',
    },
    {
      title: 'Dữ liệu thời gian thực',
      description: 'Cập nhật dữ liệu cổ phiếu và thị trường theo thời gian thực từ các nguồn uy tín.',
    },
    {
      title: 'Giao diện thân thiện',
      description: 'Thiết kế giao diện đơn giản, dễ sử dụng phù hợp với mọi đối tượng nhà đầu tư.',
    },
    {
      title: 'Bảo mật cao',
      description: 'Hệ thống bảo mật đa lớp đảm bảo thông tin cá nhân và dữ liệu đầu tư được bảo vệ an toàn.',
    },
  ]

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      {/* Welcome Section */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
          Chào mừng đến với Stock Chat Bot
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
          Hệ thống tư vấn cổ phiếu thông minh với AI
        </Typography>
        <Chip
          label="Phiên bản Beta"
          color="primary"
          variant="outlined"
          sx={{ mb: 2 }}
        />
      </Box>

      {/* Quick Actions */}
      <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Thao tác nhanh
      </Typography>
      <Grid container spacing={3} sx={{ mb: 6 }}>
        {quickActions.map((action, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                cursor: 'pointer',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4,
                },
              }}
              onClick={action.action}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center', pt: 3 }}>
                <Avatar
                  sx={{
                    bgcolor: action.color,
                    width: 80,
                    height: 80,
                    mx: 'auto',
                    mb: 2,
                  }}
                >
                  {action.icon}
                </Avatar>
                <Typography variant="h6" component="h3" gutterBottom>
                  {action.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {action.description}
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: 'center', pb: 2 }}>
                <Button
                  variant="contained"
                  sx={{
                    bgcolor: action.color,
                    '&:hover': {
                      bgcolor: action.color,
                      opacity: 0.9,
                    },
                  }}
                >
                  Bắt đầu
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Features Section */}
      <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Tính năng nổi bật
      </Typography>
      <Grid container spacing={3}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" component="h3" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Getting Started */}
      <Box sx={{ mt: 6, p: 3, bgcolor: 'background.paper', borderRadius: 2 }}>
        <Typography variant="h5" component="h2" gutterBottom sx={{ fontWeight: 600 }}>
          Bắt đầu sử dụng
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Để bắt đầu sử dụng Stock Chat Bot, bạn có thể:
        </Typography>
        <Box component="ol" sx={{ pl: 2 }}>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            Nhấn vào "Bắt đầu Chat" để bắt đầu cuộc trò chuyện với AI
          </Typography>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            Tìm kiếm và phân tích cổ phiếu cụ thể
          </Typography>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            Tạo danh mục đầu tư và theo dõi hiệu suất
          </Typography>
          <Typography component="li" variant="body2">
            Xem báo cáo và thống kê chi tiết
          </Typography>
        </Box>
      </Box>
    </Box>
  )
}

export default HomePage
