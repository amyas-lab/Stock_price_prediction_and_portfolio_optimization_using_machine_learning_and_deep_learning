import { Routes, Route } from 'react-router-dom'
import Navbar    from './components/Navbar.jsx'
import Dashboard from './pages/Dashboard.jsx'
import PredictPrice from './pages/PredictPrice.jsx'
import Portfolio from './pages/Portfolio.jsx'

export default function App() {
  return (
    <div className="app">
      <Navbar />
      <main className="main-content">
        <Routes>
          <Route path="/"          element={<Dashboard />} />
          <Route path="/predict"   element={<PredictPrice />} />
          <Route path="/portfolio" element={<Portfolio />} />
        </Routes>
      </main>
      <footer className="footer">
        <span>© 2026 InvestNature. Đầu tư bền vững như thiên nhiên.</span>
        <span>☀ Phát triển với sự cân bằng tự nhiên</span>
      </footer>
    </div>
  )
}
