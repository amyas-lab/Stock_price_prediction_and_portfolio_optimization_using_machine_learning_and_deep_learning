import { useState, useEffect, useRef } from 'react'
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip as PieTip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTip,
} from 'recharts'
import {
  fetchPortfolio, fetchProfitabilityScores, fetchRiskScores, computePortfolio,
} from '../api/stockApi.js'
import { TargetIcon, TrendUpIcon, ShieldIcon, PieIcon } from '../components/Icons.jsx'

const PROFILE_MAP = { 0: 'prudent', 1: 'equal_weight', 2: 'risk_taking' }
const RISK_LABELS = ['Thận trọng', 'Cân bằng', 'Tích cực']

const PROFILE_COPY = {
  equal_weight: {
    title: 'Equal-Weight Top 10',
    badge: '★ Tốt nhất',
    badgeColor: '#2E7D32',
    desc: 'Phân bổ đều 10% mỗi cổ phiếu được mô hình chọn lọc. Chiến lược này đạt Sharpe 1.62 và lợi nhuận 65.90% trong backtest 2025–2026, vượt trội nhờ tránh được rủi ro ước lượng ma trận hiệp phương sai (DeMiguel et al., 2009).',
  },
  risk_taking: {
    title: 'Markowitz Tích Cực',
    badge: 'Rủi ro cao',
    badgeColor: '#C0392B',
    desc: 'MVO tối đa hóa Sharpe với ràng buộc không short-sell. Phân bổ tập trung vào các cổ phiếu có Sharpe cao trong giai đoạn train (2020–2024) nhưng dễ overfit do lỗi ước lượng covariance. Lợi nhuận realized 35.73%.',
  },
  prudent: {
    title: 'Markowitz Thận Trọng',
    badge: 'Rủi ro thấp',
    badgeColor: '#1565C0',
    desc: 'MVO tối thiểu hóa biến động với ràng buộc lợi nhuận tối thiểu. Thiên về cổ phiếu phòng thủ ổn định. Phù hợp nhà đầu tư bảo toàn vốn. Lợi nhuận realized 16.44%, Sharpe 0.455.',
  },
}

const BACKTEST_ROWS = [
  { label: 'Chiến lược',         ew: 'Equal-Weight Top 10', rt: 'Tích cực (MVO)', pr: 'Thận trọng (MVO)', vni: 'VN-Index (BM)' },
  { label: 'Tổng lợi nhuận',     ew: '65.90%',  rt: '35.73%', pr: '16.44%', vni: '40.74%' },
  { label: 'Lợi nhuận / năm',    ew: '45.49%',  rt: '29.88%', pr: '15.79%', vni: '31.03%' },
  { label: 'Biến động / năm',    ew: '25.30%',  rt: '29.52%', pr: '24.80%', vni: '22.69%' },
  { label: 'Sharpe Ratio',       ew: '1.6201',  rt: '0.8596', pr: '0.4550', vni: '1.1694' },
  { label: 'Max Drawdown',       ew: '-19.87%', rt: '-23.20%',pr: '-21.97%',vni: 'N/A' },
  { label: 'Win Rate',           ew: '59.27%',  rt: '55.63%', pr: '53.97%', vni: 'N/A' },
]

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

const RF_ANNUAL = 0.045

function fmtPct(v, digits = 2) {
  return v === null || v === undefined ? '—' : (v * 100).toFixed(digits) + '%'
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

function AccSection({ id, title, open, onToggle, children }) {
  return (
    <div style={{ border: '1px solid #E8DFD0', borderRadius: 8, marginBottom: 8, overflow: 'hidden' }}>
      <button
        onClick={() => onToggle(id)}
        style={{
          width: '100%', textAlign: 'left', padding: '10px 14px',
          background: open ? '#F5EFE6' : '#FAF7F2',
          border: 'none', cursor: 'pointer',
          fontSize: 13, fontWeight: 600, color: 'var(--text-primary)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}
      >
        <span>{title}</span>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div style={{ padding: '12px 14px', fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
          {children}
        </div>
      )}
    </div>
  )
}

export default function Portfolio() {
  const [riskIdx,       setRiskIdx]       = useState(1)
  const [data,          setData]          = useState(null)
  const [profData,      setProfData]      = useState(null)
  const [riskData,      setRiskData]      = useState(null)
  const [loading,       setLoading]       = useState(false)
  const [error,         setError]         = useState(null)
  const [lastFetched,   setLastFetched]   = useState(null)

  // Cash allocation slider (0–80%)
  const [cashPct,       setCashPct]       = useState(0)

  // Editable weights for What-If
  const [editWeights,   setEditWeights]   = useState({})
  const [weightError,   setWeightError]   = useState(null)
  const [scenario,      setScenario]      = useState(null)
  const [scenarioLoad,  setScenarioLoad]  = useState(false)

  // Academic accordion
  const [accOpen,       setAccOpen]       = useState({})
  const toggleAcc = id => setAccOpen(prev => ({ ...prev, [id]: !prev[id] }))

  const profile = PROFILE_MAP[riskIdx]

  async function loadPortfolio() {
    setLoading(true); setError(null)
    try {
      const [port, prof, risk] = await Promise.all([
        fetchPortfolio(profile),
        fetchProfitabilityScores(),
        fetchRiskScores(),
      ])
      setData(port); setProfData(prof); setRiskData(risk)
      setLastFetched(new Date())
      // initialise editable weights from API response
      const initW = {}
      port.stocks.forEach(s => { initW[s.ticker] = +(s.weight * 100).toFixed(1) })
      setEditWeights(initW)
      setScenario(null)
      setWeightError(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadPortfolio() }, [riskIdx])

  // Cash-adjusted stats
  const stockRet = data?.expected_return ?? 0
  const stockVol = data?.expected_vol    ?? 0
  const wCash    = cashPct / 100
  const adjRet   = (1 - wCash) * stockRet + wCash * RF_ANNUAL
  const adjVol   = (1 - wCash) * stockVol
  const adjSharpe = adjVol > 1e-9 ? (adjRet - RF_ANNUAL) / adjVol : 0

  // Displayed stats (from scenario if available, else live data with cash adj)
  const dispRet    = scenario ? scenario.expected_return : adjRet
  const dispVol    = scenario ? scenario.expected_vol    : adjVol
  const dispSharpe = scenario ? scenario.sharpe_ratio    : adjSharpe

  const pieData = data?.stocks?.map(s => ({ name: s.ticker, value: s.weight })) ?? []

  const barData = data?.stocks?.map(s => {
    const prof = profData?.scores?.find(x => x.ticker === s.ticker)
    const rsk  = riskData?.scores?.find(x => x.ticker === s.ticker)
    return {
      ticker:   s.ticker,
      diem_ln:  prof ? +(prof.composite_score * 100).toFixed(1) : null,
      diem_rr:  rsk  ? +(rsk.composite_risk  * 10).toFixed(1)  : null,
    }
  }) ?? []

  function handleWeightChange(ticker, val) {
    setEditWeights(prev => ({ ...prev, [ticker]: val === '' ? '' : +val }))
    setWeightError(null)
    setScenario(null)
  }

  async function runScenario() {
    if (!data) return
    const tickers = data.stocks.map(s => s.ticker)
    const weights = {}
    let total = 0
    for (const t of tickers) {
      const v = parseFloat(editWeights[t] ?? 0)
      if (isNaN(v) || v < 0) { setWeightError(`Tỷ trọng không hợp lệ: ${t}`); return }
      weights[t] = v / 100
      total += v
    }
    if (Math.abs(total - 100) > 1) {
      setWeightError(`Tổng tỷ trọng = ${total.toFixed(1)}% (cần = 100%)`)
      return
    }
    setScenarioLoad(true); setWeightError(null)
    try {
      const res = await computePortfolio(tickers, weights)
      setScenario(res)
    } catch (err) {
      setWeightError(err.response?.data?.detail || err.message)
    } finally {
      setScenarioLoad(false)
    }
  }

  const profileInfo = PROFILE_COPY[profile]

  return (
    <div>
      <h1 className="page-title"><PieIcon size={26} color="var(--gold)" /> Tối ưu hóa danh mục đầu tư</h1>
      <p className="page-subtitle">Phân bổ tài sản thông minh để đạt mục tiêu lợi nhuận và kiểm soát rủi ro</p>

      {/* ── KPI cards ── */}
      <div className="stat-cards">
        <div className="stat-card">
          <div className="stat-card-label"><TargetIcon size={15} color="var(--text-muted)" /> Tổng phân bổ</div>
          <div className="stat-card-value">100.0%</div>
          <div className="stat-card-check">✓ Đã cân bằng</div>
        </div>
        <div className="stat-card">
          <div className="stat-card-label"><TrendUpIcon size={15} color="var(--text-muted)" /> Lợi nhuận / năm</div>
          <div className="stat-card-value value-green">
            {data ? fmtPct(dispRet) : '—'}
          </div>
          <div className="stat-card-note">
            {cashPct > 0 ? `Đã điều chỉnh ${cashPct}% tiền mặt` : scenario ? 'Kịch bản tùy chỉnh' : 'Backtest realized'}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-card-label"><ShieldIcon size={15} color="var(--text-muted)" /> Biến động / năm</div>
          <div className="stat-card-value value-gold">
            {data ? fmtPct(dispVol) : '—'}
          </div>
          <div className="stat-card-note">Độ lệch chuẩn</div>
        </div>
        <div className="stat-card">
          <div className="stat-card-label">Sharpe Ratio</div>
          <div className="stat-card-value" style={{ color: 'var(--gold-dark)' }}>
            {data ? dispSharpe.toFixed(4) : '—'}
          </div>
          <div className="stat-card-note">RF = 4.5%/năm</div>
        </div>
      </div>

      {/* ── Backtest leaderboard ── */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Kết quả Backtest — Walk-Forward 2025–2026</div>
        <div className="card-sub" style={{ marginBottom: 12 }}>
          Look-back 20 phiên · Conviction ≥ 0.55 · Nguồn: notebook cell 81
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#F5EFE6' }}>
                {['Chỉ số', 'EW Top 10 ★', 'Tích cực (MVO)', 'Thận trọng (MVO)', 'VN-Index'].map(h => (
                  <th key={h} style={{
                    padding: '8px 12px', textAlign: h === 'Chỉ số' ? 'left' : 'center',
                    fontSize: 12, fontWeight: 600, color: 'var(--text-primary)',
                    borderBottom: '2px solid #E8DFD0',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {BACKTEST_ROWS.slice(1).map((row, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#FDFAF6' : '#FAF7F2' }}>
                  <td style={{ padding: '7px 12px', fontWeight: 500, color: 'var(--text-primary)', fontSize: 13 }}>{row.label}</td>
                  <td style={{ padding: '7px 12px', textAlign: 'center', fontWeight: 700, color: '#2E7D32', fontSize: 13 }}>{row.ew}</td>
                  <td style={{ padding: '7px 12px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: 13 }}>{row.rt}</td>
                  <td style={{ padding: '7px 12px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: 13 }}>{row.pr}</td>
                  <td style={{ padding: '7px 12px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>{row.vni}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="portfolio-layout">
        {/* ── Left panel ── */}
        <div className="portfolio-left">

          {/* Strategy selector */}
          <div className="card">
            <div className="card-title">Chiến lược đầu tư</div>

            {/* Profile microcopy */}
            {profileInfo && (
              <div style={{
                background: '#F5EFE6', borderRadius: 8, padding: '10px 12px', marginBottom: 14,
                borderLeft: `3px solid ${profileInfo.badgeColor}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)' }}>{profileInfo.title}</span>
                  <span style={{
                    fontSize: 11, fontWeight: 600, color: '#fff',
                    background: profileInfo.badgeColor, borderRadius: 4, padding: '1px 6px',
                  }}>{profileInfo.badge}</span>
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.6 }}>
                  {profileInfo.desc}
                </p>
              </div>
            )}

            <div className="form-group">
              <div className="risk-label">
                <span className="risk-label-key">Khẩu vị rủi ro</span>
                <span className="risk-label-val">{RISK_LABELS[riskIdx]}</span>
              </div>
              <input
                type="range" min={0} max={2} step={1}
                value={riskIdx}
                onChange={e => setRiskIdx(+e.target.value)}
              />
              <div className="slider-marks">
                <span>Thận trọng</span>
                <span>Cân bằng</span>
                <span>Tích cực</span>
              </div>
            </div>

            {/* Cash allocation slider */}
            <div className="form-group">
              <div className="risk-label">
                <span className="risk-label-key">Tiền mặt (RF = 4.5%/năm)</span>
                <span className="risk-label-val">{cashPct}%</span>
              </div>
              <input
                type="range" min={0} max={80} step={5}
                value={cashPct}
                onChange={e => { setCashPct(+e.target.value); setScenario(null) }}
              />
              <div className="slider-marks">
                <span>0% (All-in)</span>
                <span>40%</span>
                <span>80%</span>
              </div>
              {cashPct > 0 && (
                <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-muted)' }}>
                  Cổ phiếu {100 - cashPct}% · Tiền mặt {cashPct}%
                  → LN ≈ {fmtPct(adjRet)} · Biến động ≈ {fmtPct(adjVol)}
                </div>
              )}
            </div>

            {/* Refresh button */}
            <button
              className="btn btn-primary btn-full"
              onClick={loadPortfolio}
              disabled={loading}
            >
              {loading ? '⟳ Đang tải...' : '⟳ Làm mới dữ liệu'}
            </button>
            {lastFetched && (
              <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                Cập nhật lúc {lastFetched.toLocaleTimeString('vi-VN')}
              </div>
            )}
          </div>

          {/* Asset list with editable weights */}
          {data && (
            <div className="card">
              <div className="card-title">Quản lý tài sản</div>
              <div className="card-sub" style={{ marginBottom: 8 }}>Chỉnh tỷ trọng → Tính kịch bản</div>

              <div className="asset-list-scroll" style={{ marginTop: 8 }}>
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
                          <div className="asset-field-label">Tỷ trọng (%)</div>
                          <input
                            type="number"
                            min={0} max={100} step={0.1}
                            value={editWeights[s.ticker] ?? (s.weight * 100).toFixed(1)}
                            onChange={e => handleWeightChange(s.ticker, e.target.value)}
                            style={{
                              width: 56, border: '1px solid #D4C090', borderRadius: 4,
                              padding: '2px 4px', fontSize: 13, textAlign: 'center',
                              background: '#FAF7F2', color: 'var(--text-primary)',
                            }}
                          />
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
              </div>

              {weightError && (
                <div style={{
                  background: '#FFF3F3', border: '1px solid #F0C0C0',
                  borderRadius: 6, padding: '8px 12px', marginTop: 8,
                  fontSize: 12, color: '#C0392B',
                }}>
                  {weightError}
                </div>
              )}

              {scenario && (
                <div style={{
                  background: '#F0F7F0', border: '1px solid #A8D5A8',
                  borderRadius: 6, padding: '8px 12px', marginTop: 8, fontSize: 13,
                }}>
                  <div style={{ fontWeight: 600, color: '#2E7D32', marginBottom: 4 }}>
                    Kết quả kịch bản (test period 2025+)
                  </div>
                  <div>LN/năm: <strong>{fmtPct(scenario.expected_return)}</strong></div>
                  <div>Biến động: <strong>{fmtPct(scenario.expected_vol)}</strong></div>
                  <div>Sharpe: <strong>{scenario.sharpe_ratio.toFixed(4)}</strong></div>
                </div>
              )}

              <button
                className="btn btn-primary btn-full"
                style={{ marginTop: 10 }}
                onClick={runScenario}
                disabled={scenarioLoad || !data}
              >
                {scenarioLoad ? '⟳ Đang tính...' : '🔬 Tính lại kịch bản'}
              </button>
            </div>
          )}
        </div>

        {/* ── Right panel ── */}
        <div>
          {loading && (
            <div className="card">
              <div className="spinner-wrap">
                <div className="spinner" />
                <p className="spinner-label">Đang tải dữ liệu danh mục...</p>
              </div>
            </div>
          )}
          {error && <div className="error-box">Lỗi: {error}</div>}

          {data && !loading && (
            <>
              {/* Pie chart */}
              <div className="chart-block" style={{ marginBottom: 16 }}>
                <div className="chart-block-title">Phân bổ danh mục</div>
                <div className="chart-block-sub">Tỷ trọng từng tài sản</div>
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
                    <Legend iconSize={10} wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Bar chart */}
              <div className="chart-block" style={{ marginBottom: 16 }}>
                <div className="chart-block-title">Điểm lợi nhuận vs rủi ro</div>
                <div className="chart-block-sub">
                  Điểm LN: percentile rank 0–100 · Điểm RR: risk score 0–10 (×10)
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

              {/* Academic accordion */}
              <div className="card">
                <div className="card-title" style={{ marginBottom: 10 }}>Phương pháp học thuật</div>
                <AccSection id="walkforward" title="Walk-Forward Backtest là gì?" open={!!accOpen.walkforward} onToggle={toggleAcc}>
                  <p>Walk-forward backtest mô phỏng giao dịch thực tế: tại mỗi thời điểm <em>t</em>, mô hình chỉ sử dụng dữ liệu đến <em>t</em> để ra tín hiệu, sau đó đánh giá kết quả tại <em>t+1</em> (1-day look-ahead = 0). Không có look-ahead bias.</p>
                  <p style={{ marginTop: 8 }}>Thiết lập: look-back = 20 phiên · conviction threshold = 0.55 · giai đoạn test = 2025-01-24 đến 2026-04.</p>
                </AccSection>
                <AccSection id="sharpe" title="Sharpe Ratio và ý nghĩa" open={!!accOpen.sharpe} onToggle={toggleAcc}>
                  <p>Sharpe = (μ − RF) / σ × √252, với RF = 4.5%/năm (lãi suất phi rủi ro VN). Sharpe {'>'} 1.0 được coi là tốt; {'>'} 1.5 là xuất sắc.</p>
                  <p style={{ marginTop: 8 }}>Equal-Weight Top 10 đạt Sharpe <strong>1.6201</strong> — vượt VN-Index (1.1694) và cả hai chiến lược Markowitz MVO.</p>
                </AccSection>
                <AccSection id="mvo" title="Tại sao Equal-Weight thắng Markowitz MVO?" open={!!accOpen.mvo} onToggle={toggleAcc}>
                  <p><strong>Lý thuyết (DeMiguel et al., 2009):</strong> MVO tối ưu hóa dựa trên ma trận hiệp phương sai ước lượng từ dữ liệu lịch sử. Sai số ước lượng này tích lũy và thường làm cho 1/N Equal-Weight thắng out-of-sample.</p>
                  <p style={{ marginTop: 8 }}><strong>Yếu tố cụ thể — VIC outlier:</strong> VIC tăng +678% trong test period. Equal-Weight tự động phân bổ 10% cho VIC; MVO chỉ phân bổ 2% do biến động lịch sử cao của VIC. Một cổ phiếu này đóng góp phần lớn chênh lệch hiệu suất.</p>
                  <p style={{ marginTop: 8 }}><em>"No investment strategy can be expected to consistently outperform the 1/N naive diversification rule."</em> — DeMiguel, Garlappi &amp; Uppal (2009), Review of Financial Studies.</p>
                </AccSection>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
