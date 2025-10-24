import React from 'react'
import {
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Divider,
  Tooltip,
  IconButton,
} from '@mui/material'
import {
  Home as HomeIcon,
  Chat as ChatIcon,
  TrendingUp as TrendingUpIcon,
  AccountBalance as PortfolioIcon,
  Person as PersonIcon,
  Close as CloseIcon,
} from '@mui/icons-material'
import { useNavigate, useLocation } from 'react-router-dom'

interface SidebarProps {
  onClose?: () => void
}

const Sidebar: React.FC<SidebarProps> = ({ onClose }) => {
  const navigate = useNavigate()
  const location = useLocation()

  const menuItems = [
    {
      text: 'Trang chủ',
      icon: <HomeIcon />,
      path: '/',
    },
    {
      text: 'Chat',
      icon: <ChatIcon />,
      path: '/chat',
    },
    {
      text: 'Cổ phiếu',
      icon: <TrendingUpIcon />,
      path: '/stock',
    },
    {
      text: 'Danh mục',
      icon: <PortfolioIcon />,
      path: '/portfolio',
    },
    {
      text: 'Hồ sơ',
      icon: <PersonIcon />,
      path: '/profile',
    },
  ]

  const handleNavigation = (path: string) => {
    navigate(path)
    if (onClose) {
      onClose()
    }
  }

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/'
    }
    return location.pathname.startsWith(path)
  }

  return (
    <Box sx={{ 
      height: '100vh', 
      display: 'flex', 
      flexDirection: 'column',
      width: '100%'
    }}>
      {/* Header */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          minHeight: 64,
        }}
      >
        <Typography variant="h6" component="div" sx={{ fontWeight: 600 }}>
          Stock Bot
        </Typography>
        {onClose && (
          <Tooltip title="Đóng menu">
            <IconButton onClick={onClose} size="small">
              <CloseIcon />
            </IconButton>
          </Tooltip>
        )}
      </Box>

      <Divider />

      {/* Navigation Menu */}
      <List sx={{ 
        flexGrow: 1, 
        pt: 1,
        overflow: 'auto',
        minHeight: 0
      }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              onClick={() => handleNavigation(item.path)}
              selected={isActive(item.path)}
              sx={{
                mx: 1,
                borderRadius: 1,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'primary.contrastText',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'primary.contrastText',
                  },
                },
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 40,
                  color: isActive(item.path) ? 'primary.contrastText' : 'inherit',
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: '0.875rem',
                  fontWeight: isActive(item.path) ? 600 : 400,
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      {/* Footer */}
      <Box sx={{ 
        p: 2,
        mt: 'auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 'auto'
      }}>
        <Typography variant="caption" color="text.secondary" align="center" sx={{ mb: 0.5 }}>
          © 2024 Stock Chat Bot
        </Typography>
        <Typography variant="caption" color="text.secondary" align="center">
          Phiên bản 1.0.0
        </Typography>
      </Box>
    </Box>
  )
}

export default Sidebar
