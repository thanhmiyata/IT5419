import React from 'react'
import {
  Box,
  Typography,
  IconButton,
  Badge,
  Tooltip,
  Avatar,
  Menu,
  MenuItem,
  Divider,
} from '@mui/material'
import {
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
  Logout as LogoutIcon,
  Person as PersonIcon,
  Menu as MenuIcon,
} from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'

interface HeaderProps {
  onDesktopDrawerToggle?: () => void
  desktopOpen?: boolean
}

const Header: React.FC<HeaderProps> = ({ onDesktopDrawerToggle, desktopOpen = true }) => {
  const navigate = useNavigate()
  
  // Mock data for demo
  const user = {
    id: '1',
    email: 'demo@example.com',
    username: 'demo',
    fullName: 'Demo User',
    avatar: '',
    role: 'user' as const,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  }
  const notifications: any[] = []

  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null)
  const [notificationAnchor, setNotificationAnchor] = React.useState<null | HTMLElement>(null)

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleProfileMenuClose = () => {
    setAnchorEl(null)
  }

  const handleNotificationMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationAnchor(event.currentTarget)
  }

  const handleNotificationMenuClose = () => {
    setNotificationAnchor(null)
  }

  const handleLogout = async () => {
    try {
      // Mock logout for demo
      console.log('Logging out...')
      navigate('/login')
    } catch (error) {
      console.error('Logout failed:', error)
    }
    handleProfileMenuClose()
  }

  const handleProfileClick = () => {
    navigate('/profile')
    handleProfileMenuClose()
  }

  const handleSettingsClick = () => {
    // TODO: Open settings modal
    handleProfileMenuClose()
  }

  const unreadNotifications = notifications.filter(n => !n.read).length

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        width: '100%',
      }}
    >
      {/* Left side - Desktop drawer toggle */}
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        {onDesktopDrawerToggle && (
          <Tooltip title={desktopOpen ? "Đóng menu" : "Mở menu"}>
            <IconButton
              color="inherit"
              onClick={onDesktopDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          </Tooltip>
        )}
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Stock Chat Bot
        </Typography>
      </Box>

      {/* Right side actions */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        {/* Notifications */}
        <Tooltip title="Thông báo">
          <IconButton
            color="inherit"
            onClick={handleNotificationMenuOpen}
            sx={{ position: 'relative' }}
          >
            <Badge badgeContent={unreadNotifications} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>
        </Tooltip>

        {/* Profile Menu */}
        <Tooltip title="Tài khoản">
          <IconButton
            size="large"
            edge="end"
            aria-label="account of current user"
            aria-controls="primary-search-account-menu"
            aria-haspopup="true"
            onClick={handleProfileMenuOpen}
            color="inherit"
          >
            <Avatar
              sx={{ width: 32, height: 32 }}
              src={user?.avatar}
              alt={user?.fullName}
            >
              {user?.fullName?.charAt(0)?.toUpperCase()}
            </Avatar>
          </IconButton>
        </Tooltip>
      </Box>

      {/* Profile Menu */}
      <Menu
        anchorEl={anchorEl}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        keepMounted
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        open={Boolean(anchorEl)}
        onClose={handleProfileMenuClose}
      >
        <MenuItem onClick={handleProfileClick}>
          <PersonIcon sx={{ mr: 1 }} />
          Hồ sơ
        </MenuItem>
        <MenuItem onClick={handleSettingsClick}>
          <SettingsIcon sx={{ mr: 1 }} />
          Cài đặt
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleLogout}>
          <LogoutIcon sx={{ mr: 1 }} />
          Đăng xuất
        </MenuItem>
      </Menu>

      {/* Notifications Menu */}
      <Menu
        anchorEl={notificationAnchor}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        keepMounted
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        open={Boolean(notificationAnchor)}
        onClose={handleNotificationMenuClose}
        sx={{ maxHeight: 400 }}
      >
        {notifications.length === 0 ? (
          <MenuItem disabled>
            <Typography variant="body2" color="text.secondary">
              Không có thông báo nào
            </Typography>
          </MenuItem>
        ) : (
          notifications.slice(0, 5).map((notification) => (
            <MenuItem key={notification.id}>
              <Box>
                <Typography variant="body2">
                  {notification.message}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {new Date(notification.timestamp).toLocaleString('vi-VN')}
                </Typography>
              </Box>
            </MenuItem>
          ))
        )}
        {notifications.length > 5 && (
          <>
            <Divider />
            <MenuItem onClick={() => console.log('Clear all notifications')}>
              <Typography variant="body2" color="primary">
                Xem tất cả thông báo
              </Typography>
            </MenuItem>
          </>
        )}
      </Menu>
    </Box>
  )
}

export default Header
