import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip as PieTip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTip,
} from 'recharts'
import { fetchPortfolio, fetchProfitabilityScores, fetchRiskScores } from '../api/stockApi.js'
import { TargetIcon, TrendUpIcon, ShieldIcon, PieIcon } from '../components/Icons.jsx'

const PROFILE_MAP = { 0: 'prudent', 1: 'equal_weight', 2: 'risk_taking' }
const RISK_LABELS = ['Thận trọng', 'Cân bằng', 'Tích cực']

const PIE_COLORS = [
  '#C4A265','#5C7A45','#8B7355','#D4C090','#2D5A3D','#C8BA9A',
  '#4A7C5F','#9E7E45','#7A6845','#3D6B4F',
]

const TICKER_NAMES = {
  FPT:'FPT Corp', VCB:'Vietcombank', VHM:'Vinhomes', VNM:'Vinamilk',
  HPG:'Hòa Phát', VIC:'Vingroup', TCB:'Techcombank', MSN:'Masan',
  MWG:'Mobile World', VND:'VNDirect', BID:'BIDV', CTG:'VietinBank',
  MBB:'MB Bank', ACB:'ACB', HDB:'HDBank', TPB:'TPBank',
  SHB:'SHB', PDR:'Phát Đạt', KDH:'Khang Điền', DXG:'Đất Xanh',
  GAS:'PV Gas', HSG:'Hoa Sen', PNJ:'PNJ', SAB:'Sabeco',
  CMG:'CMC', ELC:'Elcom', SGT:'Saigon Tel',
}

const PieCustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <strong>{payload[0].name}</strong>: {(payload[0].value * 100).toFixed(1)}%
    </div>
  )
}

const BarCustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-day">{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} className="chart-tooltip-item" style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(1)}%
        </div>
      ))}
    </div>
  )
}

export default function Portfolio() {
  const [riskIdx,  setRiskIdx]  = useState(1)
  const [target,   setTarget]   = useState(10)
  const [data,     setData]     = useState(null)
  const [profData, setProfData] = useState(null)
  const [riskData, setRiskData] = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)

  async function loadPortfolio() {
    setLoading(true); setError(null)
    try {
      const profile = PROFILE_MAP[riskIdx]
      const [port, prof, risk] = await Promise.all([
        fetchPortfolio(profile),
        fetchProfitabilityScores(),
        fetchRiskScores(),
      ])
      setData(port); setProfData(prof); setRiskData(risk)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadPortfolio() }, [])

  const pieData = data?.stocks?.map(s => ({ name: s.ticker, value: s.weight })) ?? []

  // Bar chart: profitability rank (0–100) vs risk score normalised to same 0–100 scale
  // composite_score is a percentile rank in [0,1] → multiply by 100
  // composite_risk (= final_risk_score) is on a 0–10 scale → multiply by 10 to reach 0–100
  const barData = data?.stocks?.map(s => {
    const ticker = s.ticker
    const prof = profData?.scores?.find(x => x.ticker === ticker)
    const rsk  = riskData?.scores?.find(x => x.ticker === ticker)
    return {
      ticker,
      diem_ln: prof ? +(prof.composite_score * 100).toFixed(1) : null,
      diem_rr: rsk  ? +(rsk.composite_risk * 10).toFixed(1)    : null,
    }
  }) ?? []

  return (
    <div>
      <h1 className="page-title"><PieIcon size={26} color="var(--gold)" /> Tối ưu hóa danh mục đầu tư</h1>
      <p className="page-subtitle">Phân bổ tài sản thông minh để đạt mục tiêu lợi nhuận và kiểm soát rủi ro</p>

      {/* Stat cards */}
      <div className="stat-cards">
        <div className="stat-card">
          <div className="stat-card-label"><TargetIcon size={15} color="var(--text-muted)" /> Tổng phân bổ</div>
          <div className="stat-card-value">100.0%</div>
          <div className="stat-card-check">✓ Đã cân bằng</div>
        </div>
        <div className="stat-card">
          <div className="stat-card-label"><TrendUpIcon size={15} color="var(--text-muted)" /> Lợi nhuận kỳ vọng</div>
          <div className="stat-card-value value-green">
            {data ? (data.expected_return * 100).toFixed(2) : '—'}%
          </div>
          <div className="stat-card-note">Hàng năm</div>
        </div>
        <div className="stat-card">
          <div className="stat-card-label"><ShieldIcon size={15} color="var(--text-muted)" /> Mức độ rủi ro</div>
          <div className="stat-card-value value-gold">
            {data ? (data.expected_vol * 100).toFixed(2) : '—'}%
          </div>
          <div className="stat-card-note">Độ lệch chuẩn</div>
        </div>
      </div>

      <div className="portfolio-layout">
        {/* ── Left panel ── */}
        <div className="portfolio-left">
          {/* Objective card */}
          <div className="card">
            <div className="card-title">Mục tiêu đầu tư</div>
            <div className="card-sub">Thiết lập khẩu vị rủi ro và mục tiêu lợi nhuận</div>

            <div className="form-group">
              <div className="risk-label">
                <span className="risk-label-key">Khẩu vị rủi ro</span>
                <span className="risk-label-val">{RISK_LABELS[riskIdx]}</span>
              </div>
              <input
                type="range"
                min={0} max={2} step={1}
                value={riskIdx}
                onChange={e => setRiskIdx(+e.target.value)}
              />
              <div className="slider-marks">
                <span>Thận trọng</span>
                <span>Cân bằng</span>
                <span>Tích cực</span>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Lợi nhuận mục tiêu (%/năm)</label>
              <input
                type="number"
                className="form-input-num"
                value={target}
                min={0} max={50}
                onChange={e => setTarget(+e.target.value)}
              />
            </div>

            <button
              className="btn btn-primary btn-full"
              onClick={loadPortfolio}
              disabled={loading}
            >
              ◎ {loading ? 'Đang tối ưu...' : 'Tối ưu danh mục'}
            </button>
          </div>

          {/* Asset list */}
          {data && (
            <div className="card">
              <div className="card-title">Quản lý tài sản</div>
              <div className="asset-list-scroll" style={{ marginTop: 12 }}>
                {data.stocks.map((s, i) => {
                  const prof = profData?.scores?.find(x => x.ticker === s.ticker)
                  const rsk  = riskData?.scores?.find(x => x.ticker === s.ticker)
                  return (
                    <div key={s.ticker} className="asset-card">
                      <div className="asset-card-header">
                        <div className="asset-dot" style={{ background: PIE_COLORS[i % PIE_COLORS.length] }} />
                        <div>
                          <div className="asset-ticker">{s.ticker}</div>
                          <div className="asset-name">{TICKER_NAMES[s.ticker] || s.sector}</div>
                        </div>
                      </div>
                      <div className="asset-fields">
                        <div>
                          <div className="asset-field-label">Phân bổ</div>
                          <div className="asset-field-value">{(s.weight * 100).toFixed(0)}%</div>
                        </div>
                        <div>
                          <div className="asset-field-label">Điểm LN</div>
                          <div className="asset-field-value" style={{ color: 'var(--green)' }}>
                            {prof ? (prof.composite_score * 100).toFixed(0) + '/100' : '—'}
                          </div>
                        </div>
                        <div>
                          <div className="asset-field-label">Rủi ro</div>
                          <div className="asset-field-value" style={{ color: 'var(--gold-dark)' }}>
                            {rsk ? rsk.composite_risk.toFixed(1) + '/10' : s.risk_score.toFixed(1) + '/10'}
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
                <button className="add-asset-btn">+ Thêm tài sản</button>
              </div>
            </div>
          )}
        </div>

        {/* ── Right panel ── */}
        <div>
          {loading && (
            <div className="card">
              <div className="spinner-wrap">
                <div className="spinner" />
                <p className="spinner-label">Đang tối ưu danh mục...</p>
              </div>
            </div>
          )}
          {error && <div className="error-box">Lỗi: {error}</div>}

          {data && !loading && (
            <>
              {/* Pie chart */}
              <div className="chart-block" style={{ marginBottom: 16 }}>
                <div className="chart-block-title">Phân bổ danh mục</div>
                <div className="chart-block-sub">Tỷ trọng từng tài sản trong danh mục</div>
                <ResponsiveContainer width="100%" height={320}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%" cy="42%"
                      innerRadius={0} outerRadius={82}
                      paddingAngle={2}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${(value * 100).toFixed(0)}%`}
                      labelLine={{ stroke: '#A09080', strokeWidth: 1, length: 8 }}
                    >
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <PieTip content={<PieCustomTooltip />} />
                    <Legend
                      iconSize={10}
                      wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Bar chart */}
              <div className="chart-block">
                <div className="chart-block-title">Điểm lợi nhuận vs rủi ro</div>
                <div className="chart-block-sub">
                  Điểm LN: percentile rank 0–100 · Điểm RR: risk score 0–10 (×10 để so sánh)
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={barData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#EDE5D8" vertical={false} />
                    <XAxis dataKey="ticker" tick={{ fill: '#A09080', fontSize: 12 }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#A09080', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <BarTip content={<BarCustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)', paddingTop: 10 }} iconType="circle" />
                    <Bar dataKey="diem_ln" name="Điểm LN (0–100)" fill="#4A7C5F" radius={[3,3,0,0]} />
                    <Bar dataKey="diem_rr" name="Điểm RR (×10)"   fill="#C4A265" radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={{
                display: 'flex', justifyContent: 'flex-end', gap: 20,
                marginTop: 10, padding: '8px 12px',
                background: '#FAF7F2', borderRadius: 8,
              }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Sharpe Ratio</div>
                  <div style={{ fontSize: 17, fontWeight: 700, color: 'var(--gold-dark)' }}>
                    {data.sharpe_ratio?.toFixed(2) ?? '—'}
                  </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Chiến lược</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
                    {RISK_LABELS[riskIdx]}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
