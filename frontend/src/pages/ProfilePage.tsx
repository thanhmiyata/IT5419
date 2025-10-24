import React, { useState } from 'react'
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  Avatar,
  Divider,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material'
import {
  Person as PersonIcon,
  Security as SecurityIcon,
  Notifications as NotificationsIcon,
} from '@mui/icons-material'
// Mock imports for demo

const ProfilePage: React.FC = () => {
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
  
  const [activeTab, setActiveTab] = useState(0)
  const [profileData, setProfileData] = useState({
    fullName: user?.fullName || '',
    username: user?.username || '',
    email: user?.email || '',
  })
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  })
  const [settings, setSettings] = useState({
    emailNotifications: true,
    pushNotifications: false,
    language: 'vi',
    theme: 'light',
  })

  const handleProfileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setProfileData(prev => ({
      ...prev,
      [name]: value,
    }))
  }

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setPasswordData(prev => ({
      ...prev,
      [name]: value,
    }))
  }

  const handleSettingsChange = (setting: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [setting]: value,
    }))
  }

  const handleUpdateProfile = async () => {
    try {
      // Mock update profile for demo
      console.log('Updating profile:', profileData)
      // Show success message
    } catch (error) {
      // Handle error
    }
  }

  const handleChangePassword = async () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      // Show error message
      return
    }
    
    try {
      // Mock change password for demo
      console.log('Changing password...')
      // Show success message and clear form
      setPasswordData({
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      })
    } catch (error) {
      // Handle error
    }
  }

  const tabs = [
    { label: 'Thông tin cá nhân', icon: <PersonIcon /> },
    { label: 'Bảo mật', icon: <SecurityIcon /> },
    { label: 'Thông báo', icon: <NotificationsIcon /> },
  ]

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Avatar
            sx={{ width: 80, height: 80, bgcolor: 'primary.main' }}
            src={user?.avatar}
          >
            {user?.fullName?.charAt(0)?.toUpperCase()}
          </Avatar>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              {user?.fullName}
            </Typography>
            <Typography variant="body1" color="text.secondary">
              {user?.email}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Thành viên từ {new Date(user?.createdAt || '').toLocaleDateString('vi-VN')}
            </Typography>
          </Box>
        </Box>
      </Paper>

      <Grid container spacing={3}>
        {/* Navigation */}
        <Grid item xs={12} md={3}>
          <Paper>
            <Box sx={{ p: 2 }}>
              {tabs.map((tab, index) => (
                <Button
                  key={index}
                  fullWidth
                  variant={activeTab === index ? 'contained' : 'text'}
                  startIcon={tab.icon}
                  onClick={() => setActiveTab(index)}
                  sx={{
                    justifyContent: 'flex-start',
                    mb: 1,
                    textTransform: 'none',
                  }}
                >
                  {tab.label}
                </Button>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Content */}
        <Grid item xs={12} md={9}>
          <Paper>
            {/* Personal Information */}
            {activeTab === 0 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Thông tin cá nhân
                </Typography>
                <Divider sx={{ mb: 3 }} />
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Họ và tên"
                      name="fullName"
                      value={profileData.fullName}
                      onChange={handleProfileChange}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Tên đăng nhập"
                      name="username"
                      value={profileData.username}
                      onChange={handleProfileChange}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Email"
                      name="email"
                      type="email"
                      value={profileData.email}
                      onChange={handleProfileChange}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      variant="contained"
                      onClick={handleUpdateProfile}
                      sx={{ minWidth: 120 }}
                    >
                      Cập nhật
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            )}

            {/* Security */}
            {activeTab === 1 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Bảo mật
                </Typography>
                <Divider sx={{ mb: 3 }} />
                
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Mật khẩu hiện tại"
                      name="currentPassword"
                      type="password"
                      value={passwordData.currentPassword}
                      onChange={handlePasswordChange}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Mật khẩu mới"
                      name="newPassword"
                      type="password"
                      value={passwordData.newPassword}
                      onChange={handlePasswordChange}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Xác nhận mật khẩu mới"
                      name="confirmPassword"
                      type="password"
                      value={passwordData.confirmPassword}
                      onChange={handlePasswordChange}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      variant="contained"
                      onClick={handleChangePassword}
                      sx={{ minWidth: 120 }}
                    >
                      Đổi mật khẩu
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            )}

            {/* Notifications */}
            {activeTab === 2 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Cài đặt thông báo
                </Typography>
                <Divider sx={{ mb: 3 }} />
                
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.emailNotifications}
                          onChange={(e) => handleSettingsChange('emailNotifications', e.target.checked)}
                        />
                      }
                      label="Nhận thông báo qua email"
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.pushNotifications}
                          onChange={(e) => handleSettingsChange('pushNotifications', e.target.checked)}
                        />
                      }
                      label="Nhận thông báo đẩy"
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Ngôn ngữ</InputLabel>
                      <Select
                        value={settings.language}
                        label="Ngôn ngữ"
                        onChange={(e) => handleSettingsChange('language', e.target.value)}
                      >
                        <MenuItem value="vi">Tiếng Việt</MenuItem>
                        <MenuItem value="en">English</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Giao diện</InputLabel>
                      <Select
                        value={settings.theme}
                        label="Giao diện"
                        onChange={(e) => handleSettingsChange('theme', e.target.value)}
                      >
                        <MenuItem value="light">Sáng</MenuItem>
                        <MenuItem value="dark">Tối</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      variant="contained"
                      sx={{ minWidth: 120 }}
                    >
                      Lưu cài đặt
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default ProfilePage
