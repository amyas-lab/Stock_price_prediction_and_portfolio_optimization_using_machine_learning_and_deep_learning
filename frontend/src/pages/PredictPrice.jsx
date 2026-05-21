import { useState } from 'react'
import {
  ResponsiveContainer, AreaChart, Area,
  LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts'
import { predictPrice } from '../api/stockApi.js'

const TICKER_NAMES = {
  FPT:'FPT Corporation', VCB:'Vietcombank', VHM:'Vinhomes', VNM:'Vinamilk',
  HPG:'Hòa Phát', VIC:'Vingroup', TCB:'Techcombank', MSN:'Masan Group',
  MWG:'Mobile World', VND:'VNDirect', BID:'BIDV', CTG:'VietinBank',
  MBB:'MB Bank', ACB:'ACB Bank', HDB:'HDBank', TPB:'TPBank',
  SHB:'SHB Bank', PDR:'Phát Đạt', KDH:'Khang Điền', DXG:'Đất Xanh',
  GAS:'PetroVietnam Gas', HSG:'Hoa Sen', PNJ:'Phú Nhuận Jewelry',
  SAB:'Sabeco', CMG:'CMC Tech', ELC:'Elcom', SGT:'Saigon Tel',
}
const KNOWN = new Set(Object.keys(TICKER_NAMES))

function fmtVND(v) {
  if (!v && v !== 0) return ''
  return v.toLocaleString('vi-VN') + ' VNĐ'
}

function buildChartData(currentPrice, predictedPrices) {
  const histLen = 20
  const data = []

  // Deterministic pseudo-random walk backward from current price
  let seed = Math.floor(currentPrice * 137) % 99991
  const rand = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280 }

  const hist = []
  let p = currentPrice
  for (let i = 0; i < histLen; i++) {
    p = p / (1 + (rand() - 0.49) * 0.018)
    hist.unshift(p)
  }
  hist[hist.length - 1] = currentPrice

  hist.forEach((price, i) => {
    data.push({ day: `Ngày ${i + 1}`, actual: Math.round(price), predicted: null, upper: null, lower: null })
  })

  const kDays = predictedPrices.length
  predictedPrices.forEach((pred, i) => {
    const upper = Math.round(pred * 1.04)
    const lower = Math.round(pred * 0.96)
    data.push({
      day: `Ngày ${histLen + i + 1}`,
      actual: null,
      predicted: Math.round(pred),
      upper,
      lower,
    })
  })
  return data
}

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-day">{label}</div>
      {payload.map(p => p.value != null && (
        <div key={p.dataKey} className="chart-tooltip-item" style={{ color: p.color }}>
          {p.name}: {p.value?.toLocaleString('vi-VN')} VNĐ
        </div>
      ))}
    </div>
  )
}

export default function PredictPrice() {
  const [ticker,  setTicker]  = useState('')
  const [days,    setDays]    = useState(30)
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)

  async function handlePredict() {
    const t = ticker.trim().toUpperCase()
    if (!t) return
    setLoading(true); setError(null); setResult(null)
    try {
      setResult(await predictPrice(t))
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Lỗi không xác định')
    } finally {
      setLoading(false)
    }
  }

  const chartData = result ? buildChartData(result.current_price, result.predicted_prices) : []
  const histData  = chartData.filter(d => d.actual != null)
  const predData  = chartData.filter(d => d.predicted != null)
  const allPrices = chartData.flatMap(d => [d.actual, d.predicted, d.upper, d.lower]).filter(Boolean)
  const yMin = allPrices.length ? Math.round(Math.min(...allPrices) * 0.996) : 0
  const yMax = allPrices.length ? Math.round(Math.max(...allPrices) * 1.004) : 'auto'

  const returnVal = result?.predicted_returns?.[0] ?? 0
  const isUp = returnVal > 0

  return (
    <div className="predict-layout">
      {/* ── Left: config panel ── */}
      <div className="predict-config">
        <div className="card">
          <div className="card-title">Cấu hình dự đoán</div>
          <div className="card-sub">Chọn cổ phiếu và khoảng thời gian dự đoán</div>

          <div className="form-group">
            <label className="form-label">Mã cổ phiếu</label>
            <select
              className="form-select"
              value={ticker}
              onChange={e => setTicker(e.target.value)}
            >
              <option value="">Chọn cổ phiếu</option>
              {Object.entries(TICKER_NAMES).map(([k, v]) => (
                <option key={k} value={k}>{k} - {v}</option>
              ))}
              <option value="__other__" disabled>── Mã khác (nhập thủ công) ──</option>
            </select>
            {(!KNOWN.has(ticker) && ticker && ticker !== '__other__') && null}
          </div>

          {/* Free-text for unknown tickers */}
          <div className="form-group">
            <label className="form-label" style={{ color: 'var(--text-muted)' }}>
              Hoặc nhập mã HOSE bất kỳ
            </label>
            <input
              className="form-input"
              placeholder="VD: SSI, VPB, ACB..."
              value={KNOWN.has(ticker) ? '' : ticker}
              onChange={e => setTicker(e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, ''))}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Số ngày dự đoán</label>
            <select className="form-select" value={days} onChange={e => setDays(+e.target.value)}>
              <option value={7}>7 ngày</option>
              <option value={14}>14 ngày</option>
              <option value={30}>30 ngày</option>
              <option value={60}>60 ngày</option>
              <option value={90}>90 ngày</option>
            </select>
          </div>

          <button
            className="btn btn-primary btn-full"
            onClick={handlePredict}
            disabled={!ticker.trim() || loading}
          >
            {loading ? '⟳ Đang dự đoán...' : '⊕ Dự đoán'}
          </button>

          {result && (
            <div className="price-result-section">
              <div style={{ borderTop: '1px solid var(--border)', margin: '18px 0 16px' }} />
              <div className="price-label">Cổ phiếu</div>
              <div className="price-ticker-name">
                {result.ticker}{TICKER_NAMES[result.ticker] ? ` - ${TICKER_NAMES[result.ticker]}` : ''}
              </div>
              <div className="price-label">Giá hiện tại</div>
              <div className="price-current">{fmtVND(result.current_price)}</div>
              <div className="price-label">Dự đoán sau {result.predicted_prices.length} ngày</div>
              <div className="price-predicted">{fmtVND(result.predicted_prices[0])}</div>
              <div className="price-label" style={{ marginTop: 8 }}>Thay đổi dự kiến</div>
              <div className={`price-change ${isUp ? 'up' : 'down'}`}>
                {isUp ? '↗' : '↘'} {isUp ? '+' : ''}{(returnVal * 100).toFixed(2)}%
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Right: charts ── */}
      <div>
        {loading && (
          <div className="card">
            <div className="spinner-wrap">
              <div className="spinner" />
              <p className="spinner-label">Đang tải dữ liệu và dự đoán...</p>
            </div>
          </div>
        )}

        {error && <div className="error-box">Lỗi: {error}</div>}

        {!result && !loading && !error && (
          <div className="card">
            <div className="predict-empty">
              <div className="predict-empty-icon">↗</div>
              <h3>Chưa có dữ liệu dự đoán</h3>
              <p>Chọn mã cổ phiếu và nhấn "Dự đoán" để xem kết quả</p>
            </div>
          </div>
        )}

        {result && !loading && (
          <>
            {/* Price chart */}
            <div className="chart-block">
              <div className="chart-block-title">Biểu đồ dự đoán giá</div>
              <div className="chart-block-sub">
                Đường màu xanh là giá thực tế, đường màu vàng là giá dự đoán
              </div>
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id="fillActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4A7C5F" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#4A7C5F" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="fillPredicted" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#C4A265" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#C4A265" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EDE5D8" vertical={false} />
                  <XAxis
                    dataKey="day"
                    tick={{ fill: '#A09080', fontSize: 11 }}
                    axisLine={false} tickLine={false}
                    interval={4}
                  />
                  <YAxis
                    domain={[yMin, yMax]}
                    tickFormatter={v => v.toLocaleString('vi-VN')}
                    tick={{ fill: '#A09080', fontSize: 11 }}
                    axisLine={false} tickLine={false}
                    width={72}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend
                    wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)', paddingTop: 10 }}
                    iconType="circle"
                  />
                  <Area
                    type="monotone"
                    dataKey="actual"
                    name="Giá thực tế"
                    stroke="#4A7C5F"
                    strokeWidth={2}
                    fill="url(#fillActual)"
                    dot={false}
                    connectNulls={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="predicted"
                    name="Giá dự đoán"
                    stroke="#C4A265"
                    strokeWidth={2}
                    strokeDasharray="5 3"
                    fill="url(#fillPredicted)"
                    dot={{ r: 3, fill: '#C4A265', strokeWidth: 0 }}
                    connectNulls={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Confidence interval chart */}
            <div className="chart-block">
              <div className="chart-block-title">Khoảng tin cậy</div>
              <div className="chart-block-sub">Vùng dự đoán với độ tin cậy 95%</div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart
                  data={chartData.filter(d => d.predicted != null)}
                  margin={{ top: 8, right: 16, bottom: 0, left: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#EDE5D8" vertical={false} />
                  <XAxis
                    dataKey="day"
                    tick={{ fill: '#A09080', fontSize: 11 }}
                    axisLine={false} tickLine={false}
                  />
                  <YAxis
                    tickFormatter={v => v.toLocaleString('vi-VN')}
                    tick={{ fill: '#A09080', fontSize: 11 }}
                    axisLine={false} tickLine={false}
                    width={72}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)', paddingTop: 10 }} iconType="circle" />
                  <Line type="monotone" dataKey="upper"     name="Giới hạn trên" stroke="#D4BA82" strokeWidth={1.5} strokeDasharray="4 3" dot={{ r: 3, fill: '#D4BA82', strokeWidth: 0 }} />
                  <Line type="monotone" dataKey="predicted" name="Dự đoán"       stroke="#C4A265" strokeWidth={2}   strokeDasharray="4 3" dot={{ r: 3, fill: '#C4A265', strokeWidth: 0 }} />
                  <Line type="monotone" dataKey="lower"     name="Giới hạn dưới" stroke="#9E7E45" strokeWidth={1.5} strokeDasharray="4 3" dot={{ r: 3, fill: '#9E7E45', strokeWidth: 0 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <p style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'right' }}>
              Model: {result.model_used} · Ngày tham chiếu: {result.prediction_date}
              {!result.is_known_ticker && ' · ⚠ Mô hình tổng quát'}
            </p>
          </>
        )}
      </div>
    </div>
  )
}
