# Hướng dẫn Chạy Frontend

## 🚀 Cài đặt và Chạy

### 1. Cài đặt Dependencies

```bash
cd frontend
npm install
```

### 2. Cấu hình Environment

```bash
cp env.example .env
```

Chỉnh sửa file `.env`:
```env
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000
VITE_APP_NAME=Stock Chat Bot
VITE_APP_VERSION=1.0.0
VITE_DEBUG=true
```

### 3. Chạy Development Server

```bash
npm run dev
```

Ứng dụng sẽ chạy tại: `http://localhost:3000`

## 📁 Cấu trúc Project

```
frontend/
├── public/                 # Static files
├── src/
│   ├── components/         # React components
│   │   ├── Auth/          # Authentication components
│   │   ├── Charts/        # Chart components
│   │   ├── Layout/        # Layout components
│   │   └── UI/            # Reusable UI components
│   ├── hooks/             # Custom React hooks
│   ├── pages/             # Page components
│   ├── services/          # API services
│   ├── store/             # Redux store
│   ├── types/             # TypeScript types
│   ├── utils/             # Utility functions
│   ├── App.tsx            # Main App component
│   └── main.tsx           # Entry point
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## 🔧 Available Scripts

- `npm run dev` - Chạy development server
- `npm run build` - Build production
- `npm run preview` - Preview production build
- `npm run lint` - Chạy ESLint
- `npm run lint:fix` - Fix ESLint errors

## 🎨 Features

### ✅ Đã hoàn thành:
- [x] React 18 + TypeScript setup
- [x] Material-UI theme và components
- [x] Redux Toolkit state management
- [x] React Router navigation
- [x] Authentication system (Login/Register)
- [x] Protected routes
- [x] Chat interface với WebSocket
- [x] Stock analysis pages
- [x] Portfolio management
- [x] Chart components (Line, Candlestick, Volume)
- [x] Responsive design
- [x] Custom hooks
- [x] Utility functions
- [x] Error handling
- [x] Loading states

### 🔄 Đang phát triển:
- [ ] Real-time stock data
- [ ] Advanced chart indicators
- [ ] Mobile app
- [ ] Push notifications
- [ ] Offline support

## 🌐 API Integration

Frontend tích hợp với backend qua:

### REST API Endpoints:
- `POST /api/auth/login` - Đăng nhập
- `POST /api/auth/register` - Đăng ký
- `GET /api/auth/me` - Lấy thông tin user
- `GET /api/stocks/:symbol` - Lấy thông tin cổ phiếu
- `GET /api/stocks/:symbol/chart` - Lấy dữ liệu biểu đồ
- `POST /api/chat/sessions` - Tạo chat session
- `POST /api/chat/sessions/:id/messages` - Gửi tin nhắn

### WebSocket Events:
- `chat_message` - Tin nhắn chat
- `typing` - Trạng thái đang gõ
- `stock_update` - Cập nhật giá cổ phiếu
- `market_update` - Cập nhật thị trường

## 🎯 Usage Examples

### 1. Sử dụng Custom Hooks

```typescript
import { useAuth, useNotifications } from '@/hooks'

function MyComponent() {
  const { user, logout } = useAuth()
  const { showSuccess, showError } = useNotifications()
  
  const handleAction = () => {
    showSuccess('Thành công!')
  }
}
```

### 2. Sử dụng Formatters

```typescript
import { formatCurrency, formatPercentage } from '@/utils'

const price = formatCurrency(45000) // "45.000 ₫"
const change = formatPercentage(5.2) // "+5.20%"
```

### 3. Sử dụng Charts

```typescript
import { TechnicalChart } from '@/components/Charts'

<TechnicalChart
  symbol="VIC"
  data={chartData}
  onTimeframeChange={(timeframe) => {
    // Handle timeframe change
  }}
/>
```

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **Module not found**
   ```bash
   npm install
   ```

2. **Port already in use**
   ```bash
   # Thay đổi port trong vite.config.ts
   server: { port: 3001 }
   ```

3. **API connection failed**
   - Kiểm tra backend server đang chạy
   - Kiểm tra VITE_API_URL trong .env

4. **WebSocket connection failed**
   - Kiểm tra VITE_WS_URL trong .env
   - Kiểm tra backend WebSocket server

## 📱 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🔒 Security

- JWT tokens được lưu trong localStorage
- API calls có authentication headers
- XSS protection với React
- CSRF protection với SameSite cookies

## 📊 Performance

- Code splitting với React.lazy
- Bundle optimization với Vite
- Image lazy loading
- Memoization với React.memo
- Virtual scrolling cho large lists

## 🧪 Testing

```bash
# Chạy tests
npm test

# Chạy tests với coverage
npm run test:coverage
```

## 🚀 Deployment

### Build Production

```bash
npm run build
```

### Deploy to Vercel

```bash
npm install -g vercel
vercel --prod
```

### Deploy to Netlify

```bash
npm run build
# Upload dist/ folder to Netlify
```

## 📞 Support

Nếu gặp vấn đề, vui lòng:

1. Kiểm tra console logs
2. Xem Network tab trong DevTools
3. Tạo issue trên GitHub
4. Liên hệ team development

---

**Happy Coding! 🎉**
