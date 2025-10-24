import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'

// Layout components
import Layout from './components/Layout/Layout'

// Page components
import HomePage from './pages/HomePage'
import ChatPage from './pages/ChatPage'
import StockPage from './pages/StockPage'
import PortfolioPage from './pages/PortfolioPage'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import ProfilePage from './pages/ProfilePage'
import NotFoundPage from './pages/NotFoundPage'

// Loading component
import LoadingSpinner from './components/UI/LoadingSpinner'

function App() {
  // Mock authentication state for demo
  const isAuthenticated = true // Set to true to bypass login for demo
  const isLoading = false

  if (isLoading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <LoadingSpinner size={60} />
      </Box>
    )
  }

  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      {/* Main app routes - bypassing authentication for demo */}
      <Route path="/" element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="chat/:sessionId" element={<ChatPage />} />
        <Route path="stock/:symbol" element={<StockPage />} />
        <Route path="portfolio" element={<PortfolioPage />} />
        <Route path="profile" element={<ProfilePage />} />
      </Route>

      {/* Catch all route */}
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  )
}

export default App
