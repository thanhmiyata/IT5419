// Email validation
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

// Password validation
export const isValidPassword = (password: string): { isValid: boolean; errors: string[] } => {
  const errors: string[] = []
  
  if (password.length < 6) {
    errors.push('Mật khẩu phải có ít nhất 6 ký tự')
  }
  
  if (!/[A-Z]/.test(password)) {
    errors.push('Mật khẩu phải có ít nhất 1 chữ hoa')
  }
  
  if (!/[a-z]/.test(password)) {
    errors.push('Mật khẩu phải có ít nhất 1 chữ thường')
  }
  
  if (!/\d/.test(password)) {
    errors.push('Mật khẩu phải có ít nhất 1 số')
  }
  
  return {
    isValid: errors.length === 0,
    errors,
  }
}

// Username validation
export const isValidUsername = (username: string): { isValid: boolean; errors: string[] } => {
  const errors: string[] = []
  
  if (username.length < 3) {
    errors.push('Tên đăng nhập phải có ít nhất 3 ký tự')
  }
  
  if (username.length > 20) {
    errors.push('Tên đăng nhập không được quá 20 ký tự')
  }
  
  if (!/^[a-zA-Z0-9_]+$/.test(username)) {
    errors.push('Tên đăng nhập chỉ được chứa chữ cái, số và dấu gạch dưới')
  }
  
  return {
    isValid: errors.length === 0,
    errors,
  }
}

// Stock symbol validation
export const isValidStockSymbol = (symbol: string): boolean => {
  const symbolRegex = /^[A-Z]{3,5}$/
  return symbolRegex.test(symbol.toUpperCase())
}

// Phone number validation (Vietnamese format)
export const isValidPhoneNumber = (phone: string): boolean => {
  const phoneRegex = /^(\+84|84|0)[1-9][0-9]{8,9}$/
  return phoneRegex.test(phone.replace(/\s/g, ''))
}

// URL validation
export const isValidUrl = (url: string): boolean => {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

// Number validation
export const isValidNumber = (value: string | number): boolean => {
  return !isNaN(Number(value)) && isFinite(Number(value))
}

// Positive number validation
export const isValidPositiveNumber = (value: string | number): boolean => {
  const num = Number(value)
  return isValidNumber(value) && num > 0
}

// Form validation helper
export const validateForm = (data: Record<string, any>, rules: Record<string, any>): Record<string, string> => {
  const errors: Record<string, string> = {}
  
  Object.keys(rules).forEach(field => {
    const value = data[field]
    const rule = rules[field]
    
    // Required validation
    if (rule.required && (!value || value.toString().trim() === '')) {
      errors[field] = rule.requiredMessage || `${field} là bắt buộc`
      return
    }
    
    // Skip other validations if field is empty and not required
    if (!value || value.toString().trim() === '') {
      return
    }
    
    // Email validation
    if (rule.type === 'email' && !isValidEmail(value)) {
      errors[field] = rule.message || 'Email không hợp lệ'
    }
    
    // Password validation
    if (rule.type === 'password') {
      const passwordValidation = isValidPassword(value)
      if (!passwordValidation.isValid) {
        errors[field] = passwordValidation.errors[0]
      }
    }
    
    // Username validation
    if (rule.type === 'username') {
      const usernameValidation = isValidUsername(value)
      if (!usernameValidation.isValid) {
        errors[field] = usernameValidation.errors[0]
      }
    }
    
    // Min length validation
    if (rule.minLength && value.toString().length < rule.minLength) {
      errors[field] = rule.message || `${field} phải có ít nhất ${rule.minLength} ký tự`
    }
    
    // Max length validation
    if (rule.maxLength && value.toString().length > rule.maxLength) {
      errors[field] = rule.message || `${field} không được quá ${rule.maxLength} ký tự`
    }
    
    // Number validation
    if (rule.type === 'number' && !isValidNumber(value)) {
      errors[field] = rule.message || `${field} phải là số`
    }
    
    // Positive number validation
    if (rule.type === 'positiveNumber' && !isValidPositiveNumber(value)) {
      errors[field] = rule.message || `${field} phải là số dương`
    }
    
    // Custom validation
    if (rule.custom && typeof rule.custom === 'function') {
      const customResult = rule.custom(value)
      if (customResult !== true) {
        errors[field] = customResult || `${field} không hợp lệ`
      }
    }
  })
  
  return errors
}
