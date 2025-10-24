import React, { useState } from 'react'
import { Outlet } from 'react-router-dom'
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  IconButton,
} from '@mui/material'
import {
  Menu as MenuIcon,
} from '@mui/icons-material'

import Sidebar from './Sidebar'
import Header from './Header'

const drawerWidth = 280

const Layout: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [desktopOpen, setDesktopOpen] = useState(true)

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen)
  }

  const handleDesktopDrawerToggle = () => {
    setDesktopOpen(!desktopOpen)
  }

  const drawer = <Sidebar onClose={handleDrawerToggle} />
  const desktopDrawer = <Sidebar onClose={handleDesktopDrawerToggle} />

  return (
    <Box sx={{ 
      display: 'flex',
      width: '100%',
      height: '100vh',
      overflow: 'hidden'
    }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { 
            xs: '100%',
            md: desktopOpen ? `calc(100% - ${drawerWidth}px)` : '100%' 
          },
          ml: { md: desktopOpen ? `${drawerWidth}px` : 0 },
          backgroundColor: 'background.paper',
          color: 'text.primary',
          boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
          transition: 'width 0.3s ease, margin-left 0.3s ease',
          zIndex: (theme) => theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Header 
            onDesktopDrawerToggle={handleDesktopDrawerToggle}
            desktopOpen={desktopOpen}
          />
        </Toolbar>
      </AppBar>

      {/* Navigation Drawer */}
      <Box
        component="nav"
        sx={{ 
          width: { md: desktopOpen ? drawerWidth : 0 }, 
          flexShrink: { md: 0 },
          transition: 'width 0.3s ease',
          overflow: 'hidden'
        }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
            },
          }}
        >
          {drawer}
        </Drawer>

        {/* Desktop drawer */}
        <Drawer
          variant="persistent"
          open={desktopOpen}
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              transition: 'width 0.3s ease, transform 0.3s ease',
              transform: desktopOpen ? 'translateX(0)' : `translateX(-${drawerWidth}px)`,
              position: 'relative',
            },
          }}
        >
          {desktopDrawer}
        </Drawer>
      </Box>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { 
            xs: '100%',
            md: desktopOpen ? `calc(100% - ${drawerWidth}px)` : '100%' 
          },
          height: '100vh',
          backgroundColor: 'background.default',
          transition: 'width 0.3s ease, margin-left 0.3s ease',
          ml: { md: desktopOpen ? 0 : 0 },
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  )
}

export default Layout
