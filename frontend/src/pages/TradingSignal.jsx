import { useState } from 'react'
import StockSearch    from '../components/StockSearch.jsx'
import TickerBadge    from '../components/TickerBadge.jsx'
import LoadingSpinner from '../components/LoadingSpinner.jsx'
import { predictSignal } from '../api/stockApi.js'

const KNOWN = new Set([
  'FPT','VCB','VHM','VNM','HPG','VIC','TCB',
  'MSN','MWG','VND','BID','CTG','MBB','ACB',
  'HDB','TPB','SHB','PDR','KDH','DXG','GAS',
  'HSG','PNJ','SAB','CMG','ELC','SGT',
])

const SIGNAL_COLORS = { BUY: 'green', SELL: 'red', HOLD: 'yellow' }

export default function TradingSignal() {
  const [ticker,    setTicker]    = useState('')
  const [threshold, setThreshold] = useState(0.55)
  const [loading,   setLoading]   = useState(false)
  const [result,    setResult]    = useState(null)
  const [error,     setError]     = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    const t = ticker.trim().toUpperCase()
    if (!t) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await predictSignal(t, threshold)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Lỗi không xác định')
    } finally {
      setLoading(false)
    }
  }

  const isLive = ticker && !KNOWN.has(ticker.toUpperCase())

  return (
    <div>
      <h1 className="page-title">Tín hiệu giao dịch</h1>
      <p className="page-subtitle">
        Phân loại BUY / SELL / HOLD. Điều chỉnh ngưỡng conviction để kiểm soát độ nhạy.
      </p>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Mã cổ phiếu</label>
            <StockSearch value={ticker} onChange={setTicker} />
          </div>

          <div className="form-group">
            <label className="form-label">
              Ngưỡng conviction: {(threshold * 100).toFixed(0)}%
            </label>
            <div className="slider-wrap">
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>40%</span>
              <input
                type="range"
                min="0.40" max="0.80" step="0.01"
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
              />
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>80%</span>
              <span className="slider-value">{(threshold * 100).toFixed(0)}%</span>
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={!ticker.trim() || loading}
          >
            {loading ? 'Đang phân tích...' : 'Phân tích tín hiệu'}
          </button>
        </form>
      </div>

      {loading && (
        <div className="card" style={{ marginTop: 16 }}>
          <LoadingSpinner ticker={ticker.toUpperCase()} isLive={isLive} />
        </div>
      )}

      {error && <div className="error-box">Lỗi: {error}</div>}

      {result && !loading && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="result-header">
            <span className="result-ticker">{result.ticker}</span>
            <TickerBadge isKnown={result.is_known_ticker} />
            <span className={`badge badge-${result.signal.toLowerCase()}`} style={{ fontSize: 18, padding: '6px 18px' }}>
              {result.signal}
            </span>
          </div>

          {!result.is_known_ticker && (
            <p style={{ fontSize: 12, color: 'var(--orange)', marginBottom: 16 }}>
              ⚠ Mã nằm ngoài tập huấn luyện. Tín hiệu từ phân loại tổng quát của MTL model.
            </p>
          )}

          {/* Probability bars */}
          <div className="prob-bars">
            {[
              { label: 'BUY',  key: 'p_buy',  cls: 'buy'  },
              { label: 'HOLD', key: 'p_hold', cls: 'hold' },
              { label: 'SELL', key: 'p_sell', cls: 'sell' },
            ].map(({ label, key, cls }) => (
              <div key={key} className="prob-row">
                <span className="prob-label" style={{
                  color: cls === 'buy' ? 'var(--green)' : cls === 'sell' ? 'var(--red)' : 'var(--yellow)',
                }}>
                  {label}
                </span>
                <div className="prob-track">
                  <div
                    className={`prob-fill ${cls}`}
                    style={{ width: `${(result[key] * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="prob-pct">{(result[key] * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>

          <div className="recommendation">{result.recommendation}</div>

          <div style={{ marginTop: 16 }}>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Conviction</div>
                <div className="metric-value">{(result.conviction * 100).toFixed(1)}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Ngưỡng đã dùng</div>
                <div className="metric-value">{(result.threshold_used * 100).toFixed(0)}%</div>
              </div>
            </div>
          </div>

          <p style={{ marginTop: 14, fontSize: 12, color: 'var(--text-muted)' }}>
            Nguồn dữ liệu: {result.data_source} · Ngày tín hiệu: {result.signal_date}
          </p>
        </div>
      )}
    </div>
  )
}
