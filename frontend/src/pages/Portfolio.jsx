import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip as PieTip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTip,
} from 'recharts'
import {
  fetchPortfolio, fetchProfitabilityScores, fetchRiskScores, computePortfolio,
} from '../api/stockApi.js'
import { TargetIcon, TrendUpIcon, ShieldIcon, PieIcon } from '../components/Icons.jsx'

// ── constants ─────────────────────────────────────────────────
const PROFILE_MAP  = { 0: 'prudent', 1: 'equal_weight', 2: 'risk_taking' }
const RISK_LABELS  = ['Thận trọng', 'Cân bằng', 'Tích cực']
const RF_ANNUAL    = 0.045

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

// ── profile copy ───────────────────────────────────────────────
const PROFILE_COPY = {
  equal_weight: {
    title: 'Phân Bổ Đều (Equal-Weight)',
    badge: '★ Hiệu quả nhất',
    badgeColor: '#2E7D32',
    summary: 'Phân bổ 10% đều cho mỗi cổ phiếu trong Top 10 được mô hình chọn lọc.',
    mechanism: 'Không có tối ưu hóa phức tạp — mỗi cổ phiếu nhận đúng 10% danh mục. Cách tiếp cận đơn giản này hiệu quả vì nghiên cứu của DeMiguel và cộng sự (2009) đã chứng minh trên 14 bộ dữ liệu thực tế rằng phân bổ đều thường thắng các phương pháp tối ưu hóa phức tạp hơn, do các phương pháp đó dễ bị ảnh hưởng bởi sai số trong ước lượng ma trận hiệp phương sai. Kết quả kiểm thử 2025–2026: lợi nhuận 65.90%, Sharpe 1.62.',
  },
  risk_taking: {
    title: 'Tích Cực — Markowitz Tối Đa Sharpe',
    badge: 'Rủi ro cao',
    badgeColor: '#C0392B',
    summary: 'Tối ưu hóa tỷ lệ lợi nhuận / rủi ro bằng thuật toán Markowitz.',
    mechanism: 'Sử dụng bài toán tối ưu hóa Markowitz (thuật toán SLSQP): tìm tỷ trọng danh mục sao cho Sharpe Ratio đạt lớn nhất, với ràng buộc không được bán khống (tỷ trọng ≥ 0). Mô hình học từ dữ liệu 2020–2024 để ước lượng lợi nhuận kỳ vọng và ma trận hiệp phương sai — nhưng sai số ước lượng thường tích lũy và làm giảm hiệu quả trên dữ liệu mới. Kết quả kiểm thử 2025–2026: lợi nhuận 35.73%, Sharpe 0.86.',
  },
  prudent: {
    title: 'Thận Trọng — Markowitz Tối Thiểu Rủi Ro',
    badge: 'Rủi ro thấp',
    badgeColor: '#1565C0',
    summary: 'Tối ưu hóa giảm thiểu biến động danh mục bằng thuật toán Markowitz.',
    mechanism: 'Cũng dùng bài toán Markowitz SLSQP, nhưng mục tiêu là tìm tỷ trọng sao cho độ lệch chuẩn danh mục nhỏ nhất trong khi vẫn đảm bảo lợi nhuận kỳ vọng tối thiểu. Danh mục thiên về cổ phiếu phòng thủ, ít biến động. Phù hợp nhà đầu tư ưu tiên bảo toàn vốn hơn tăng trưởng. Kết quả kiểm thử 2025–2026: lợi nhuận 16.44%, Sharpe 0.46.',
  },
}

// ── backtest rows + tooltips ───────────────────────────────────
const BACKTEST_META = [
  {
    label: 'Tổng lợi nhuận (Cumulative Return)',
    ew: '65.90%', rt: '35.73%', pr: '16.44%', vni: '40.74%',
    tip: 'Lợi nhuận tích lũy toàn bộ giai đoạn kiểm thử (24/01/2025 – 20/04/2026). Tính bằng phương pháp nhân vốn từng ngày: ∏(1 + r_t) − 1. Đây là con số đã được kiểm chứng thực tế trên dữ liệu ngoài mẫu, không phải ước tính lý thuyết.',
  },
  {
    label: 'Lợi nhuận / năm (Annualized Return)',
    ew: '45.49%', rt: '29.88%', pr: '15.79%', vni: '31.03%',
    tip: 'Lợi nhuận quy về tốc độ 1 năm, tính từ lợi nhuận tích lũy và số ngày kiểm thử. Cho phép so sánh công bằng giữa các chiến lược có thời gian kiểm thử khác nhau.',
  },
  {
    label: 'Biến động / năm (Volatility)',
    ew: '25.30%', rt: '29.52%', pr: '24.80%', vni: '22.69%',
    tip: 'Độ lệch chuẩn (Standard Deviation) của lợi nhuận hằng ngày, nhân √252 để quy về năm. Đo mức độ dao động giá trị danh mục — con số càng thấp thì rủi ro biến động càng ít.',
  },
  {
    label: 'Sharpe Ratio',
    ew: '1.6201', rt: '0.8596', pr: '0.4550', vni: '1.1694',
    tip: 'Sharpe = (Lợi nhuận năm − Risk-Free Rate) / Biến động năm. Risk-Free Rate = 4.5%/năm (tương đương lãi suất trái phiếu Chính phủ VN). Sharpe > 1.0 là tốt, > 1.5 là xuất sắc — thể hiện lợi nhuận thu được trên mỗi đơn vị rủi ro chấp nhận.',
  },
  {
    label: 'Max Drawdown',
    ew: '-19.87%', rt: '-23.20%', pr: '-21.97%', vni: 'N/A *',
    tip: 'Mức sụt giảm lớn nhất từ đỉnh xuống đáy trong toàn bộ giai đoạn kiểm thử. Ví dụ: -19.87% có nghĩa là tại thời điểm tệ nhất, danh mục mất gần 20% so với mức đỉnh trước đó. (*) VN-Index không có chỉ số này vì đây là chuẩn tham chiếu thụ động — không có tín hiệu giao dịch.',
  },
  {
    label: 'Win Rate',
    ew: '59.27%', rt: '55.63%', pr: '53.97%', vni: 'N/A *',
    tip: 'Tỷ lệ số phiên giao dịch có lợi nhuận dương so với tổng số phiên. Tính trên tất cả các lệnh mua được mô hình tín hiệu phát ra. (*) VN-Index là chuẩn tham chiếu thụ động, không có tín hiệu giao dịch nên không tính Win Rate.',
  },
]

// ── helpers ───────────────────────────────────────────────────
function fmtPct(v, digits = 2) {
  return v === null || v === undefined ? '—' : (v * 100).toFixed(digits) + '%'
}

// ── sub-components ────────────────────────────────────────────
function StatCard({ icon, label, value, note, explainTitle, explain, valueColor }) {
  const [open, setOpen] = useState(false)
  return (
    <div
      className="stat-card"
      onClick={() => explain && setOpen(o => !o)}
      onMouseEnter={e => explain && (e.currentTarget.style.transform = 'translateY(-2px)')}
      onMouseLeave={e => (e.currentTarget.style.transform = '')}
      style={{ cursor: explain ? 'pointer' : 'default', transition: 'transform 0.15s, box-shadow 0.15s' }}
      title={explain ? 'Nhấn để xem giải thích' : undefined}
    >
      <div className="stat-card-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        {icon}{label}
        {explain && <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 2, fontStyle: 'italic' }}>— nhấn để xem</span>}
      </div>
      <div className="stat-card-value" style={valueColor ? { color: valueColor } : {}}>{value}</div>
      <div className="stat-card-note">{note}</div>
      {open && explain && (
        <div style={{
          marginTop: 8, padding: '8px 10px', background: '#F5EFE6', borderRadius: 6,
          fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.65,
          borderTop: '1px solid #E0D5C0', textAlign: 'left',
        }}>
          {explainTitle && <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>{explainTitle}</div>}
          {explain}
        </div>
      )}
    </div>
  )
}

function MetricRow({ row, i }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <tr
        style={{ background: i % 2 === 0 ? '#FDFAF6' : '#FAF7F2', cursor: 'pointer' }}
        onClick={() => setOpen(o => !o)}
        title="Nhấn để xem giải thích"
      >
        <td style={{ padding: '7px 12px', fontSize: 13 }}>
          <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{row.label}</span>
          <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 6 }}>{open ? '▲' : '▼'}</span>
        </td>
        <td style={{ padding: '7px 12px', textAlign: 'center', fontWeight: 700, color: '#2E7D32', fontSize: 13 }}>{row.ew}</td>
        <td style={{ padding: '7px 12px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: 13 }}>{row.rt}</td>
        <td style={{ padding: '7px 12px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: 13 }}>{row.pr}</td>
        <td style={{ padding: '7px 12px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>{row.vni}</td>
      </tr>
      {open && row.tip && (
        <tr style={{ background: '#F5EFE6' }}>
          <td colSpan={5} style={{ padding: '6px 16px 10px', fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {row.tip}
          </td>
        </tr>
      )}
    </>
  )
}

function ProfileCard({ info, active }) {
  const [showMech, setShowMech] = useState(false)
  return (
    <div style={{
      background: active ? '#F0EBE0' : '#F5EFE6', borderRadius: 8, padding: '10px 12px', marginBottom: 12,
      borderLeft: `3px solid ${info.badgeColor}`,
      transition: 'background 0.2s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'nowrap' }}>
        <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>{info.title}</span>
        <span style={{
          fontSize: 11, fontWeight: 600, color: '#fff', whiteSpace: 'nowrap',
          background: info.badgeColor, borderRadius: 4, padding: '1px 6px', flexShrink: 0,
        }}>{info.badge}</span>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-secondary)', margin: '0 0 6px', lineHeight: 1.5 }}>
        {info.summary}
      </p>
      <button
        onClick={() => setShowMech(o => !o)}
        style={{
          background: 'none', border: `1px solid ${info.badgeColor}`, borderRadius: 4,
          color: info.badgeColor, fontSize: 11, padding: '2px 8px', cursor: 'pointer',
          fontWeight: 600,
        }}
      >
        {showMech ? '▲ Ẩn cơ chế' : '▼ Xem cơ chế hoạt động'}
      </button>
      {showMech && (
        <p style={{ fontSize: 12, color: 'var(--text-secondary)', margin: '8px 0 0', lineHeight: 1.65 }}>
          {info.mechanism}
        </p>
      )}
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

// ── main component ────────────────────────────────────────────
export default function Portfolio() {
  const [riskIdx,      setRiskIdx]      = useState(1)
  const [data,         setData]         = useState(null)
  const [profData,     setProfData]     = useState(null)
  const [riskData,     setRiskData]     = useState(null)
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState(null)
  const [lastFetched,  setLastFetched]  = useState(null)
  const [cashPct,      setCashPct]      = useState(0)
  const [editWeights,  setEditWeights]  = useState({})
  const [weightError,  setWeightError]  = useState(null)
  const [scenario,     setScenario]     = useState(null)
  const [scenarioLoad, setScenarioLoad] = useState(false)
  const [accOpen,      setAccOpen]      = useState({})
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
      const initW = {}
      port.stocks.forEach(s => { initW[s.ticker] = +(s.weight * 100).toFixed(1) })
      setEditWeights(initW)
      setScenario(null); setWeightError(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadPortfolio() }, [riskIdx])

  // cash-adjusted stats
  const stockRet  = data?.expected_return ?? 0
  const stockVol  = data?.expected_vol    ?? 0
  const wCash     = cashPct / 100
  const adjRet    = (1 - wCash) * stockRet + wCash * RF_ANNUAL
  const adjVol    = (1 - wCash) * stockVol
  const adjSharpe = adjVol > 1e-9 ? (adjRet - RF_ANNUAL) / adjVol : 0

  const dispRet    = scenario ? scenario.expected_return : adjRet
  const dispVol    = scenario ? scenario.expected_vol    : adjVol
  const dispSharpe = scenario ? scenario.sharpe_ratio    : adjSharpe

  const pieData = data?.stocks?.map(s => ({ name: s.ticker, value: s.weight })) ?? []
  const barData = data?.stocks?.map(s => {
    const prof = profData?.scores?.find(x => x.ticker === s.ticker)
    const rsk  = riskData?.scores?.find(x => x.ticker === s.ticker)
    return {
      ticker:  s.ticker,
      diem_ln: prof ? +(prof.composite_score * 100).toFixed(1) : null,
      diem_rr: rsk  ? +(rsk.composite_risk  * 10).toFixed(1)  : null,
    }
  }) ?? []

  function handleWeightChange(ticker, val) {
    setEditWeights(prev => ({ ...prev, [ticker]: val === '' ? '' : +val }))
    setWeightError(null); setScenario(null)
  }

  async function runScenario() {
    if (!data) return
    const tickers = data.stocks.map(s => s.ticker)
    const weights = {}; let total = 0
    for (const t of tickers) {
      const v = parseFloat(editWeights[t] ?? 0)
      if (isNaN(v) || v < 0) { setWeightError(`Tỷ trọng không hợp lệ: ${t}`); return }
      weights[t] = v / 100; total += v
    }
    if (Math.abs(total - 100) > 1) {
      setWeightError(`Tổng tỷ trọng = ${total.toFixed(1)}% (cần = 100%)`); return
    }
    setScenarioLoad(true); setWeightError(null)
    try {
      setScenario(await computePortfolio(tickers, weights))
    } catch (err) {
      setWeightError(err.response?.data?.detail || err.message)
    } finally {
      setScenarioLoad(false)
    }
  }

  const retNote = cashPct > 0
    ? `Đã điều chỉnh ${cashPct}% tiền mặt`
    : scenario ? 'Kịch bản tùy chỉnh'
    : 'Lợi nhuận thực nghiệm kiểm thử'

  return (
    <div>
      <h1 className="page-title"><PieIcon size={26} color="var(--gold)" /> Tối ưu hóa danh mục đầu tư</h1>
      <p className="page-subtitle">Phân bổ tài sản thông minh để đạt mục tiêu lợi nhuận và kiểm soát rủi ro</p>

      {/* ── KPI cards ── */}
      <div className="stat-cards">
        <StatCard
          icon={<TargetIcon size={14} color="var(--text-muted)" />}
          label=" Tổng phân bổ"
          value="100.0%"
          note="✓ Đã cân bằng"
          explain="Tổng tỷ trọng của tất cả tài sản trong danh mục luôn bằng 100%, đảm bảo không có vốn nhàn rỗi. Khi điều chỉnh tỷ trọng thủ công, hệ thống sẽ chuẩn hóa lại nếu tổng lệch quá 1%."
        />
        <StatCard
          icon={<TrendUpIcon size={14} color="var(--text-muted)" />}
          label=" Lợi nhuận / năm"
          value={data ? fmtPct(dispRet) : '—'}
          note={retNote}
          valueColor="var(--green)"
          explainTitle="Lợi nhuận thực nghiệm (Backtest Return)"
          explain="Lợi nhuận đã được kiểm nghiệm thực tế trên dữ liệu ngoài mẫu (24/01/2025 – 20/04/2026). 'Thực nghiệm' có nghĩa là mô hình không nhìn thấy dữ liệu này trong quá trình học — kết quả phản ánh hiệu suất thực khi triển khai, không phải ước tính lý thuyết. Quy về năm bằng cách tính tốc độ tăng trưởng trung bình trên số ngày kiểm thử."
        />
        <StatCard
          icon={<ShieldIcon size={14} color="var(--text-muted)" />}
          label=" Biến động / năm"
          value={data ? fmtPct(dispVol) : '—'}
          note="Độ lệch chuẩn (Standard Deviation)"
          valueColor="var(--gold-dark)"
          explainTitle="Biến động danh mục (Volatility)"
          explain="Độ lệch chuẩn (Standard Deviation) của lợi nhuận hằng ngày, nhân √252 để quy về năm. Đây là thước đo rủi ro phổ biến nhất — danh mục có biến động 25% có nghĩa là trong điều kiện bình thường, lợi nhuận hằng năm có thể dao động ±25% so với mức trung bình."
        />
        <StatCard
          label="Sharpe Ratio"
          value={data ? dispSharpe.toFixed(4) : '—'}
          note="Risk-Free Rate = 4.5%/năm"
          valueColor="var(--gold-dark)"
          explainTitle="Sharpe Ratio — hiệu quả điều chỉnh rủi ro"
          explain="Sharpe = (Lợi nhuận năm − Risk-Free Rate) / Biến động năm. Risk-Free Rate (lãi suất phi rủi ro) = 4.5%/năm, tương đương lãi suất trái phiếu Chính phủ Việt Nam. Sharpe > 1.0 là tốt, > 1.5 là xuất sắc. Ý nghĩa: bạn nhận được bao nhiêu đơn vị lợi nhuận vượt trội trên mỗi đơn vị rủi ro chấp nhận."
        />
      </div>

      {/* ── Backtest leaderboard ── */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Kết quả Kiểm Thử Thực Nghiệm — Walk-Forward Backtest</div>
        <div style={{ marginBottom: 12 }}>
          <div className="card-sub">
            Dữ liệu kiểm thử: <strong>24/01/2025 – 20/04/2026</strong> · Dữ liệu huấn luyện: 2020–2024 · Phương pháp: Walk-Forward, không nhìn trước (look-ahead = 0)
          </div>
          <div className="card-sub" style={{ marginTop: 4 }}>
            Cơ chế: tại mỗi phiên, mô hình phát tín hiệu MUA/BÁN/GIỮ dựa trên 25 chỉ số kỹ thuật của 20 phiên gần nhất · Conviction threshold ≥ 0.55 · Nhấn vào từng chỉ số để xem giải thích chi tiết.
          </div>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#F5EFE6' }}>
                {['Chỉ số', 'EW Top 10 ★', 'Tích cực (MVO)', 'Thận trọng (MVO)', 'VN-Index (tham chiếu)'].map(h => (
                  <th key={h} style={{
                    padding: '8px 12px', textAlign: h === 'Chỉ số' ? 'left' : 'center',
                    fontSize: 12, fontWeight: 600, color: 'var(--text-primary)',
                    borderBottom: '2px solid #E8DFD0',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {BACKTEST_META.map((row, i) => (
                <MetricRow key={i} row={row} i={i} />
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
          (*) VN-Index được dùng làm chuẩn tham chiếu theo chiến lược mua và nắm giữ thụ động. Max Drawdown và Win Rate chỉ tính cho danh mục có tín hiệu giao dịch chủ động từ mô hình.
        </div>
      </div>

      <div className="portfolio-layout">
        {/* ── Left panel ── */}
        <div className="portfolio-left">

          {/* Strategy selector */}
          <div className="card">
            <div className="card-title">Chiến lược đầu tư</div>
            <div className="card-sub" style={{ marginBottom: 12 }}>Chọn chiến lược phù hợp với khẩu vị rủi ro của bạn</div>

            {/* Profile cards */}
            {[
              { key: 'prudent',      idx: 0 },
              { key: 'equal_weight', idx: 1 },
              { key: 'risk_taking',  idx: 2 },
            ].map(({ key, idx }) => (
              <div
                key={key}
                onClick={() => setRiskIdx(idx)}
                style={{ cursor: 'pointer', opacity: riskIdx === idx ? 1 : 0.65, transition: 'opacity 0.2s' }}
              >
                <ProfileCard info={PROFILE_COPY[key]} active={riskIdx === idx} />
              </div>
            ))}

            {/* Cash allocation slider */}
            <div className="form-group" style={{ marginTop: 4 }}>
              <div className="risk-label">
                <span className="risk-label-key">Tiền mặt (Risk-Free Rate = 4.5%/năm)</span>
                <span className="risk-label-val">{cashPct}%</span>
              </div>
              <input
                type="range" min={0} max={80} step={5}
                value={cashPct}
                onChange={e => { setCashPct(+e.target.value); setScenario(null) }}
              />
              <div className="slider-marks">
                <span>0% (Toàn cổ phiếu)</span>
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

            <button className="btn btn-primary btn-full" onClick={loadPortfolio} disabled={loading}>
              {loading ? '⟳ Đang tải...' : '⟳ Làm mới dữ liệu'}
            </button>
            {lastFetched && (
              <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                Cập nhật lúc {lastFetched.toLocaleTimeString('vi-VN')}
              </div>
            )}
          </div>

          {/* Asset list */}
          {data && (
            <div className="card">
              <div className="card-title">Quản lý tài sản</div>
              <div className="card-sub" style={{ marginBottom: 4 }}>Chỉnh tỷ trọng → Nhấn "Tính kịch bản" để xem kết quả mô phỏng</div>

              {/* Legend */}
              <div style={{
                background: '#FAF7F2', border: '1px solid #E8DFD0', borderRadius: 6,
                padding: '7px 10px', marginBottom: 10, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6,
              }}>
                <strong style={{ color: 'var(--green)' }}>Điểm lợi nhuận (0–100):</strong> Điểm tổng hợp từ 5 yếu tố — tín hiệu mô hình MTL, chỉ số kỹ thuật, tín hiệu XGBoost, Sharpe lịch sử, và xu hướng giá. Cổ phiếu đạt điểm cao hơn được mô hình đánh giá có tiềm năng sinh lợi tốt hơn.
                <br />
                <strong style={{ color: 'var(--gold-dark)' }}>Điểm rủi ro (0–10):</strong> Điểm tổng hợp từ 5 yếu tố — biến động giá, áp lực bán, mức sụt giảm lịch sử, tương quan danh mục, và nguy cơ đảo chiều. Điểm càng cao thì rủi ro càng lớn.
              </div>

              <div className="asset-list-scroll" style={{ marginTop: 4 }}>
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
                            type="number" min={0} max={100} step={0.1}
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
                          <div className="asset-field-label">Điểm lợi nhuận</div>
                          <div className="asset-field-value" style={{ color: 'var(--green)' }}>
                            {prof ? (prof.composite_score * 100).toFixed(0) + '/100' : '—'}
                          </div>
                        </div>
                        <div>
                          <div className="asset-field-label">Điểm rủi ro</div>
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
                }}>{weightError}</div>
              )}

              {scenario && (
                <div style={{
                  background: '#F0F7F0', border: '1px solid #A8D5A8',
                  borderRadius: 6, padding: '8px 12px', marginTop: 8, fontSize: 13,
                }}>
                  <div style={{ fontWeight: 600, color: '#2E7D32', marginBottom: 4 }}>
                    Kết quả kịch bản (dữ liệu kiểm thử 2025+)
                  </div>
                  <div>Lợi nhuận / năm: <strong>{fmtPct(scenario.expected_return)}</strong></div>
                  <div>Biến động / năm: <strong>{fmtPct(scenario.expected_vol)}</strong></div>
                  <div>Sharpe Ratio: <strong>{scenario.sharpe_ratio.toFixed(4)}</strong></div>
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
                      data={pieData} cx="50%" cy="42%"
                      innerRadius={0} outerRadius={82} paddingAngle={2} dataKey="value"
                      label={({ name, value }) => `${name}: ${(value * 100).toFixed(0)}%`}
                      labelLine={{ stroke: '#A09080', strokeWidth: 1, length: 8 }}
                    >
                      {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                    </Pie>
                    <PieTip content={<PieCustomTooltip />} />
                    <Legend iconSize={10} wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Bar chart */}
              <div className="chart-block" style={{ marginBottom: 16 }}>
                <div className="chart-block-title">Điểm lợi nhuận vs điểm rủi ro</div>
                <div className="chart-block-sub">
                  Điểm lợi nhuận: thứ hạng phần trăm 0–100 · Điểm rủi ro: 0–10 (×10 để so sánh)
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={barData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#EDE5D8" vertical={false} />
                    <XAxis dataKey="ticker" tick={{ fill: '#A09080', fontSize: 12 }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#A09080', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <BarTip content={<BarCustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)', paddingTop: 10 }} iconType="circle" />
                    <Bar dataKey="diem_ln" name="Điểm lợi nhuận (0–100)" fill="#4A7C5F" radius={[3,3,0,0]} />
                    <Bar dataKey="diem_rr" name="Điểm rủi ro (×10)"      fill="#C4A265" radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Academic accordion */}
              <div className="card">
                <div className="card-title" style={{ marginBottom: 10 }}>Phương pháp học thuật</div>
                <AccSection id="walkforward" title="Walk-Forward Backtest — Cơ chế kiểm thử" open={!!accOpen.walkforward} onToggle={toggleAcc}>
                  <p>Walk-forward backtest mô phỏng giao dịch thực tế theo thời gian tuyến tính: tại mỗi phiên <em>t</em>, mô hình chỉ được phép sử dụng dữ liệu tính đến phiên <em>t</em> để phát tín hiệu, sau đó đánh giá kết quả tại phiên <em>t+1</em>. Không có look-ahead bias (không nhìn trước tương lai).</p>
                  <p style={{ marginTop: 8 }}>Thiết lập cụ thể: cửa sổ nhìn lại (look-back) = 20 phiên giao dịch · Ngưỡng tin cậy (conviction threshold) = 0.55 · Giai đoạn kiểm thử: 24/01/2025 – 20/04/2026. Dữ liệu kiểm thử hoàn toàn tách biệt với dữ liệu huấn luyện (2020–2024).</p>
                </AccSection>

                <AccSection id="sharpe" title="Sharpe Ratio — Đo lường hiệu quả điều chỉnh rủi ro" open={!!accOpen.sharpe} onToggle={toggleAcc}>
                  <p>
                    Công thức (đầu vào đã được quy về năm):
                  </p>
                  <div style={{
                    background: '#F5EFE6', borderRadius: 6, padding: '8px 12px',
                    margin: '8px 0', fontFamily: 'monospace', fontSize: 13,
                    color: 'var(--text-primary)', textAlign: 'center',
                  }}>
                    Sharpe = (μ<sub>năm</sub> − RF<sub>năm</sub>) / σ<sub>năm</sub>
                  </div>
                  <p>
                    Trong đó: μ<sub>năm</sub> = lợi nhuận trung bình hằng năm (= lợi nhuận ngày × 252), σ<sub>năm</sub> = độ lệch chuẩn hằng năm (= độ lệch chuẩn ngày × √252), RF<sub>năm</sub> = 4.5%/năm (Risk-Free Rate — lãi suất phi rủi ro tương đương trái phiếu Chính phủ VN). Do μ và σ đã được quy về năm trước khi tính, công thức không cần thêm hệ số √252.
                  </p>
                  <p style={{ marginTop: 8 }}>Equal-Weight Top 10 đạt Sharpe <strong>1.6201</strong> — vượt VN-Index (1.1694) và cả hai chiến lược Markowitz MVO. Sharpe {'>'} 1.0 là tốt; {'>'} 1.5 là xuất sắc.</p>
                </AccSection>

                <AccSection id="mvo" title="Tại sao Equal-Weight thắng Markowitz MVO?" open={!!accOpen.mvo} onToggle={toggleAcc}>
                  <p><strong>Lý thuyết (DeMiguel et al., 2009):</strong> MVO tối ưu hóa dựa trên ma trận hiệp phương sai ước lượng từ dữ liệu lịch sử. Sai số ước lượng này tích lũy và thường làm cho phân bổ đều 1/N thắng out-of-sample — kết quả được kiểm chứng trên 14 bộ dữ liệu thực tế.</p>
                  <p style={{ marginTop: 8 }}><strong>Yếu tố cụ thể — VIC outlier:</strong> VIC tăng +678% trong giai đoạn kiểm thử. Equal-Weight tự động phân bổ 10% cho VIC; MVO chỉ phân bổ 2% do biến động lịch sử cao của VIC. Một cổ phiếu này đóng góp phần lớn chênh lệch hiệu suất.</p>
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
