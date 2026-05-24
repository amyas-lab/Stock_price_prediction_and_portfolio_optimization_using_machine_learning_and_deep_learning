import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip as PieTip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTip,
} from 'recharts'
import {
  fetchPortfolio, fetchProfitabilityScores, fetchRiskScores, computePortfolio,
} from '../api/stockApi.js'
import { TrendUpIcon, ShieldIcon, PieIcon } from '../components/Icons.jsx'

// ── constants ─────────────────────────────────────────────────
const PROFILE_MAP = { 0: 'prudent', 1: 'equal_weight', 2: 'risk_taking' }
const RISK_LABELS = ['Thận trọng', 'Cân bằng', 'Tích cực']
const RF_ANNUAL   = 0.045

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

// ── KPI card config ───────────────────────────────────────────
const KPI_CARDS = [
  {
    id: 'return',
    bg: '#EDFAF2', border: '#A8D5B8', valColor: '#2E7D32',
    label: 'Lợi nhuận / năm',
    icon: <TrendUpIcon size={13} color="#2E7D32" />,
    explainTitle: 'Lợi nhuận thực nghiệm (Backtest Return)',
    explain: 'Lợi nhuận đã được kiểm nghiệm thực tế trên dữ liệu ngoài mẫu (24/01/2025 – 20/04/2026). "Thực nghiệm" có nghĩa là mô hình không nhìn thấy dữ liệu này trong quá trình học - kết quả phản ánh hiệu suất thực khi triển khai, không phải ước tính lý thuyết. Quy về năm từ lợi nhuận tích lũy và số ngày kiểm thử.',
  },
  {
    id: 'vol',
    bg: '#FFF8E8', border: '#E8C97A', valColor: '#9E7E45',
    label: 'Biến động / năm',
    icon: <ShieldIcon size={13} color="#9E7E45" />,
    explainTitle: 'Biến động danh mục (Volatility / Standard Deviation)',
    explain: 'Độ lệch chuẩn (Standard Deviation) của lợi nhuận hằng ngày, nhân √252 để quy về năm. Đây là thước đo rủi ro phổ biến nhất - danh mục có biến động 25% có nghĩa là trong điều kiện bình thường, lợi nhuận hằng năm có thể dao động ±25% so với mức trung bình.',
  },
  {
    id: 'sharpe',
    bg: '#EFF4FF', border: '#A8C0F0', valColor: '#3A5FA0',
    label: 'Sharpe Ratio',
    explainTitle: 'Sharpe Ratio - hiệu quả điều chỉnh rủi ro',
    explain: 'Sharpe = (Lợi nhuận năm − Risk-Free Rate) / Biến động năm. Risk-Free Rate = 4.5%/năm (lãi suất phi rủi ro, tương đương trái phiếu Chính phủ VN). Sharpe > 1.0 là tốt, > 1.5 là xuất sắc. Ý nghĩa: mỗi đơn vị rủi ro chấp nhận mang lại bao nhiêu đơn vị lợi nhuận vượt trội so với gửi ngân hàng.',
  },
]

// ── backtest rows ─────────────────────────────────────────────
const BACKTEST_META = [
  {
    label: 'Tổng lợi nhuận (Cumulative Return)',
    ew: '65.90%', rt: '35.73%', pr: '16.44%', vni: '40.74%',
    tip: 'Lợi nhuận tích lũy toàn bộ giai đoạn kiểm thử. Tính bằng phương pháp nhân vốn từng ngày: ∏(1 + r_t) − 1. Đây là con số đã được kiểm chứng thực tế trên dữ liệu ngoài mẫu, không phải ước tính lý thuyết.',
  },
  {
    label: 'Lợi nhuận / năm (Annualized Return)',
    ew: '45.49%', rt: '29.88%', pr: '15.79%', vni: '31.03%',
    tip: 'Lợi nhuận quy về tốc độ 1 năm từ lợi nhuận tích lũy và số ngày kiểm thử. Cho phép so sánh công bằng giữa các chiến lược.',
  },
  {
    label: 'Biến động / năm (Volatility)',
    ew: '25.30%', rt: '29.52%', pr: '24.80%', vni: '22.69%',
    tip: 'Độ lệch chuẩn (Standard Deviation) của lợi nhuận hằng ngày, nhân √252 để quy về năm. Con số càng thấp thì rủi ro biến động càng ít.',
  },
  {
    label: 'Sharpe Ratio',
    ew: '1.6201', rt: '0.8596', pr: '0.4550', vni: '1.1694',
    tip: 'Sharpe = (Lợi nhuận năm − 4.5%) / Biến động năm. Risk-Free Rate = 4.5%/năm. Sharpe > 1.0 là tốt, > 1.5 là xuất sắc.',
  },
  {
    label: 'Max Drawdown',
    ew: '-19.87%', rt: '-23.20%', pr: '-21.97%', vni: 'N/A *',
    tip: 'Mức sụt giảm lớn nhất từ đỉnh xuống đáy trong toàn bộ giai đoạn. (*) VN-Index là chuẩn tham chiếu thụ động - không có tín hiệu giao dịch nên không tính Max Drawdown.',
  },
  {
    label: 'Win Rate',
    ew: '59.27%', rt: '55.63%', pr: '53.97%', vni: 'N/A *',
    tip: 'Tỷ lệ số phiên giao dịch có lợi nhuận dương. (*) VN-Index là chuẩn tham chiếu thụ động - không có tín hiệu giao dịch nên không tính Win Rate.',
  },
]

// ── profile modal content ─────────────────────────────────────
const PROFILE_DETAIL = {
  prudent: {
    title: 'Thận Trọng - Markowitz Tối Thiểu Rủi Ro',
    subtitle: 'Mục tiêu: Giảm thiểu biến động danh mục trong khi vẫn đảm bảo lợi nhuận tối thiểu',
    badge: 'Rủi ro thấp', badgeColor: '#1565C0',
    sections: [
      { title: 'Bài toán tối ưu hóa', body: 'Sử dụng thuật toán SLSQP (Sequential Least Squares Programming) từ thư viện scipy.optimize để giải bài toán tối thiểu hóa: tìm tỷ trọng w sao cho phương sai danh mục w^T Σ w nhỏ nhất, với ràng buộc: (1) tổng tỷ trọng = 1, (2) không được bán khống (w_i ≥ 0), (3) lợi nhuận kỳ vọng w^T μ đạt ít nhất một ngưỡng tối thiểu.' },
      { title: 'Dữ liệu đầu vào', body: 'Ma trận hiệp phương sai Σ và vector lợi nhuận kỳ vọng μ được ước lượng từ lợi nhuận hằng ngày giai đoạn 2020–2024, sau đó nhân 252 để quy về năm trước khi đưa vào SLSQP (pre-annualization - đảm bảo gradient tốt hơn).' },
      { title: 'Đặc điểm danh mục', body: 'Danh mục thiên về các cổ phiếu phòng thủ có biến động lịch sử thấp (ngân hàng lớn, tiêu dùng thiết yếu). Phân bổ không đều - cổ phiếu ổn định nhận tỷ trọng cao hơn.' },
      { title: 'Kết quả kiểm thử 2025–2026', body: 'Lợi nhuận thực nghiệm: 16.44% · Biến động: 24.80%/năm · Sharpe: 0.455 · Max Drawdown: -21.97% · Win Rate: 53.97%' },
    ],
  },
  equal_weight: {
    title: 'Cân Bằng - Phân Bổ Đều (Equal-Weight)',
    subtitle: 'Mục tiêu: Loại bỏ sai số ước lượng bằng cách phân bổ đều cho tất cả cổ phiếu được chọn',
    badge: 'Hiệu quả nhất', badgeColor: '#2E7D32',
    sections: [
      { title: 'Cơ chế đơn giản nhưng mạnh mẽ', body: 'Không có bài toán tối ưu hóa - mỗi cổ phiếu trong Top 10 do mô hình XGBoost T4 chọn lọc nhận đúng 10% danh mục. Không ưu tiên cổ phiếu nào hơn cổ phiếu nào dựa trên dữ liệu lịch sử.' },
      { title: 'Tại sao thắng Markowitz?', body: 'DeMiguel, Garlappi & Uppal (2009) kiểm chứng trên 14 bộ dữ liệu thực tế: phân bổ đều 1/N thường thắng MVO out-of-sample vì tránh được sai số tích lũy trong ước lượng ma trận hiệp phương sai Σ. Lỗi ước lượng Σ thường lớn hơn lợi ích của việc tối ưu hóa.' },
      { title: 'Vai trò của mô hình chọn cổ phiếu', body: 'Lợi thế thực sự đến từ bước chọn lọc: mô hình XGBoost T4 (Cascade: GRU → K-Means → XGBoost) xác định Top 10 cổ phiếu có tín hiệu tích cực nhất. Equal-Weight chỉ áp dụng sau khi đã có danh sách chất lượng cao.' },
      { title: 'Kết quả kiểm thử 2025–2026', body: 'Lợi nhuận thực nghiệm: 65.90% · Biến động: 25.30%/năm · Sharpe: 1.6201 · Max Drawdown: -19.87% · Win Rate: 59.27%' },
    ],
  },
  risk_taking: {
    title: 'Tích Cực - Markowitz Tối Đa Sharpe',
    subtitle: 'Mục tiêu: Tối đa hóa tỷ lệ lợi nhuận / rủi ro (Sharpe Ratio)',
    badge: 'Rủi ro cao', badgeColor: '#C0392B',
    sections: [
      { title: 'Bài toán tối ưu hóa', body: 'Sử dụng SLSQP để tối đa hóa Sharpe Ratio: tìm tỷ trọng w sao cho (w^T μ − RF) / √(w^T Σ w) lớn nhất, với ràng buộc: (1) tổng tỷ trọng = 1, (2) không được bán khống (w_i ≥ 0).' },
      { title: 'Dữ liệu đầu vào', body: 'Giống chiến lược Thận Trọng: Σ và μ ước lượng từ dữ liệu 2020–2024 và pre-annualize trước khi đưa vào SLSQP. Risk-Free Rate RF = 4.5%/năm.' },
      { title: 'Đặc điểm danh mục', body: 'Phân bổ tập trung vào các cổ phiếu có Sharpe cao trong giai đoạn huấn luyện. Dễ bị ảnh hưởng bởi outlier trong dữ liệu lịch sử - ví dụ: VIC có biến động cao trong 2020–2024 nên chỉ nhận 2% dù tăng +678% trong giai đoạn kiểm thử.' },
      { title: 'Kết quả kiểm thử 2025–2026', body: 'Lợi nhuận thực nghiệm: 35.73% · Biến động: 29.52%/năm · Sharpe: 0.8596 · Max Drawdown: -23.20% · Win Rate: 55.63%' },
    ],
  },
}

const SCORE_MODALS = {
  profit: {
    title: 'Điểm lợi nhuận (Profitability Score)',
    subtitle: 'Thang điểm 0–100 - thứ hạng phần trăm tổng hợp 5 yếu tố',
    sections: [
      { title: 'F1 - Tín hiệu mô hình MTL (Multi-Task Learning)', body: 'Xác suất tăng giá p_up từ mô hình deep learning GRU Encoder-Decoder, được huấn luyện đồng thời dự báo lợi nhuận 5 ngày và phân loại xu hướng.' },
      { title: 'F2 - Chỉ số kỹ thuật tổng hợp', body: 'Kết hợp RSI, MACD histogram, khoảng cách vùng hỗ trợ/kháng cự K-Means, và tín hiệu EMA crossover. Đánh giá động lực giá ngắn và trung hạn.' },
      { title: 'F3 - Tín hiệu XGBoost', body: 'Xác suất BUY từ mô hình XGBoost T4 huấn luyện trên 25 đặc trưng kỹ thuật kết hợp. Là lớp ra quyết định cuối cùng của Cascade Model.' },
      { title: 'F4 - Sharpe lịch sử', body: 'Sharpe Ratio của cổ phiếu trong giai đoạn huấn luyện (2020–2024), điều chỉnh theo Risk-Free Rate 4.5%/năm. Phản ánh lịch sử lợi nhuận / rủi ro.' },
      { title: 'F5 - Xu hướng giá (Trend Score)', body: 'Đánh giá xu hướng dựa trên chuỗi EMA 10/20/50 phiên. MA alignment = +1 nếu EMA10 > EMA20 > EMA50 (xu hướng tăng toàn diện), −1 nếu ngược lại.' },
    ],
  },
  risk: {
    title: 'Điểm rủi ro (Risk Score)',
    subtitle: 'Thang điểm 0–10 - điểm tổng hợp 5 yếu tố rủi ro (cao = rủi ro lớn)',
    sections: [
      { title: 'R1 - Biến động giá (Volatility Risk)', body: 'Độ lệch chuẩn lợi nhuận hằng ngày trong 60 phiên gần nhất, quy về năm. Biến động cao = rủi ro dao động giá lớn.' },
      { title: 'R2 - Áp lực bán (Sell Pressure Risk)', body: 'Xác suất SELL từ mô hình XGBoost T4 kết hợp với chiều MACD histogram. Đánh giá nguy cơ bán tháo ngắn hạn.' },
      { title: 'R3 - Mức sụt giảm lịch sử (Drawdown Risk)', body: 'Max Drawdown trong 252 phiên gần nhất (1 năm giao dịch). Phản ánh mức thua lỗ tệ nhất mà cổ phiếu đã trải qua.' },
      { title: 'R4 - Tương quan danh mục (Correlation Risk)', body: 'Hệ số tương quan trung bình với các cổ phiếu còn lại trong danh mục. Tương quan cao = ít tác dụng phân tán rủi ro.' },
      { title: 'R5 - Nguy cơ đảo chiều (Reversal Risk)', body: 'Kết hợp RSI (overbought/oversold) và khoảng cách đến vùng kháng cự K-Means. RSI > 70 gần kháng cự = rủi ro đảo chiều giảm cao.' },
    ],
  },
}

// ── helpers ───────────────────────────────────────────────────
function fmtPct(v, digits = 2) {
  return v === null || v === undefined ? '-' : (v * 100).toFixed(digits) + '%'
}

// ── Modal ─────────────────────────────────────────────────────
function InfoModal({ title, subtitle, sections, onClose }) {
  const [open, setOpen] = useState(null)
  const toggle = id => setOpen(prev => prev === id ? null : id)
  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)',
        zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16,
      }}
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      <div style={{
        background: '#FAF7F2', borderRadius: 16, width: '100%', maxWidth: 680,
        maxHeight: '90vh', overflowY: 'auto', boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          padding: '20px 24px 16px', borderBottom: '1px solid #EDE5D8',
          position: 'sticky', top: 0, background: '#FAF7F2', zIndex: 1,
        }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-primary)' }}>{title}</div>
            {subtitle && <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 3 }}>{subtitle}</div>}
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: '1px solid #EDE5D8', borderRadius: 8,
            width: 32, height: 32, cursor: 'pointer', fontSize: 16, color: 'var(--text-muted)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}>×</button>
        </div>
        <div style={{ padding: '16px 24px 24px' }}>
          {sections.map((s, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <button
                onClick={() => toggle(i)}
                style={{
                  width: '100%', textAlign: 'left', padding: '11px 16px',
                  background: open === i ? '#FAF3E0' : '#F9F6F1',
                  border: '1px solid #EDE5D8', borderRadius: open === i ? '10px 10px 0 0' : 10,
                  fontSize: 13, fontWeight: 600, color: 'var(--text-primary)',
                  cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                }}
              >
                <span>{s.title}</span>
                <span style={{ fontSize: 18, color: 'var(--text-muted)', lineHeight: 1 }}>{open === i ? '−' : '+'}</span>
              </button>
              {open === i && (
                <div style={{
                  padding: 16, border: '1px solid #EDE5D8', borderTop: 'none',
                  borderRadius: '0 0 10px 10px', background: '#fff',
                  fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7,
                }}>
                  {s.body}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── KPI grid with click-to-explain banner ─────────────────────
function KpiGrid({ cards }) {
  const [active, setActive] = useState(null)
  const card = active !== null ? cards[active] : null
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: card ? 0 : 0 }}>
        {cards.map((c, i) => (
          <div
            key={c.id}
            onClick={() => setActive(active === i ? null : i)}
            style={{
              background: c.bg, borderRadius: 12, padding: '14px 16px',
              border: `2px solid ${active === i ? c.border : 'transparent'}`,
              cursor: 'pointer', transition: 'border-color 0.15s, transform 0.15s',
              transform: active === i ? 'translateY(-2px)' : '',
              boxShadow: active === i ? `0 4px 14px ${c.border}55` : 'none',
            }}
          >
            <div style={{ fontSize: 12, color: '#888', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 4 }}>
              {c.icon}{c.label}
            </div>
            <div style={{ fontSize: 26, fontWeight: 800, color: c.valColor, marginBottom: 4, lineHeight: 1.1 }}>
              {c.value}
            </div>
            <div style={{ fontSize: 11, color: '#999', marginBottom: 2 }}>{c.sub}</div>
            <div style={{ fontSize: 10, color: '#BBB', marginTop: 6 }}>nhấp để xem giải thích</div>
          </div>
        ))}
      </div>
      {card && (
        <div style={{
          background: '#F5EFE6', border: `1px solid ${card.border}`,
          borderTop: 'none', borderRadius: '0 0 12px 12px',
          padding: '12px 16px', fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65,
        }}>
          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{card.explainTitle}: </span>
          {card.explain}
        </div>
      )}
    </div>
  )
}

// ── MetricRow (backtest table) ────────────────────────────────
function MetricRow({ row, i }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <tr
        style={{ background: i % 2 === 0 ? '#FDFAF6' : '#FAF7F2', cursor: 'pointer' }}
        onClick={() => setOpen(o => !o)}
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
          background: open ? '#F5EFE6' : '#FAF7F2', border: 'none', cursor: 'pointer',
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

// ── AlgoModal ─────────────────────────────────────────────────
function AlgoModal({ onClose }) {
  const [tab, setTab] = useState(0)
  const tabBtn = (active) => ({
    padding: '9px 18px', fontSize: 13, fontWeight: active ? 700 : 500,
    color: active ? 'var(--text-primary)' : 'var(--text-muted)',
    background: 'none', border: 'none',
    borderBottom: active ? '2px solid #C4A265' : '2px solid transparent',
    cursor: 'pointer',
  })
  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)',
        zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16,
      }}
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      <div style={{
        background: '#FAF7F2', borderRadius: 16, width: '100%', maxWidth: 700,
        maxHeight: '90vh', display: 'flex', flexDirection: 'column',
        boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          padding: '20px 24px 12px', borderBottom: '1px solid #EDE5D8', flexShrink: 0,
        }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-primary)' }}>Tại sao mô hình khuyến nghị như vậy?</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 3 }}>Quy trình 4 giai đoạn chọn lọc và phân bổ danh mục</div>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: '1px solid #EDE5D8', borderRadius: 8,
            width: 32, height: 32, cursor: 'pointer', fontSize: 16, color: 'var(--text-muted)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}>×</button>
        </div>

        <div style={{ display: 'flex', borderBottom: '1px solid #EDE5D8', padding: '0 16px', flexShrink: 0 }}>
          <button style={tabBtn(tab === 0)} onClick={() => setTab(0)}>Giai đoạn 1-3: Chon loc</button>
          <button style={tabBtn(tab === 1)} onClick={() => setTab(1)}>Giai đoạn 4: Phân bổ</button>
        </div>

        <div style={{ overflowY: 'auto', padding: '20px 24px 28px', flex: 1 }}>
          {tab === 0 && (
            <div>
              <div style={{ marginBottom: 22 }}>
                <div style={{
                  fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10,
                  padding: '8px 12px', background: '#F5EFE6', borderRadius: 6, borderLeft: '3px solid #C4A265',
                }}>Giai đoạn 1 - Điểm lợi nhuận (5-Factor Profitability Score)</div>
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7, margin: '0 0 10px' }}>
                  Toàn bộ cổ phiếu được chấm điểm theo 5 yếu tố, mỗi yếu tố chuyển thành thứ hạng phần trăm rồi tổng hợp theo trọng số:
                </p>
                {[
                  ['MTL Trajectory - GRU Encoder-Decoder', 'xác suất tăng giá p_up từ mô hình Multi-Task Learning, huấn luyện đồng thời dự báo lợi nhuận 5 ngày và phân loại xu hướng', '30%'],
                  ['Task 3 Signal - XGBoost T4', 'xác suất BUY từ mô hình Cascade (GRU -> K-Means -> XGBoost), là lớp ra quyết định cuối cùng', '25%'],
                  ['Technical Momentum', 'RSI, MACD histogram, tín hiệu EMA crossover 10/20/50, khoảng cách vùng hỗ trợ/kháng cự K-Means', '20%'],
                  ['Risk-Adjusted Efficiency', 'Sharpe/Sortino lịch sử giai đoạn 2020-2024, điều chỉnh theo RF = 4.5%/năm', '15%'],
                  ['Trend ADX', 'sức mạnh xu hướng đo bằng Average Directional Index', '10%'],
                ].map(([name, desc, w], i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
                    gap: 10, padding: '6px 10px', borderRadius: 4, marginBottom: 3,
                    background: i % 2 === 0 ? '#FDFAF6' : 'transparent',
                  }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                      <strong style={{ color: 'var(--text-primary)' }}>{name}</strong>: {desc}
                    </span>
                    <span style={{ fontWeight: 700, color: '#2E7D32', flexShrink: 0, fontSize: 13 }}>{w}</span>
                  </div>
                ))}
                <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '8px 0 0' }}>Top 10 cổ phiếu điểm cao nhất được chuyển sang giai đoạn 2.</p>
              </div>

              <div style={{ marginBottom: 22 }}>
                <div style={{
                  fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10,
                  padding: '8px 12px', background: '#FFF8E8', borderRadius: 6, borderLeft: '3px solid #E8C97A',
                }}>Giai đoạn 2 - Điểm rủi ro (5-Factor Risk Score) + Sharpe Stress Test</div>
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7, margin: '0 0 10px' }}>
                  Thang điểm 0-10 tổng hợp 5 yếu tố rủi ro (điểm càng cao, rủi ro càng lớn):
                </p>
                {[
                  ['Volatility', 'độ lệch chuẩn lợi nhuận 60 phiên gần nhất, quy về năm (×√252)', '30%'],
                  ['SELL Exposure', 'xác suất SELL từ XGBoost T4 kết hợp chiều MACD histogram', '25%'],
                  ['Max Drawdown', 'mức sụt giảm lớn nhất từ đỉnh xuống đáy trong 252 phiên gần nhất', '20%'],
                  ['Correlation/Sector Penalty', 'hệ số tương quan trung bình với các cổ phiếu còn lại trong danh mục', '15%'],
                  ['Reversal Risk', 'RSI overbought (> 70) kết hợp khoảng cách gần vùng kháng cự K-Means', '10%'],
                ].map(([name, desc, w], i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
                    gap: 10, padding: '6px 10px', borderRadius: 4, marginBottom: 3,
                    background: i % 2 === 0 ? '#FDFAF6' : 'transparent',
                  }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                      <strong style={{ color: 'var(--text-primary)' }}>{name}</strong>: {desc}
                    </span>
                    <span style={{ fontWeight: 700, color: '#C0392B', flexShrink: 0, fontSize: 13 }}>{w}</span>
                  </div>
                ))}
                <div style={{
                  marginTop: 10, padding: '9px 12px', background: '#FFF3E0',
                  borderRadius: 6, border: '1px solid #FFD180',
                  fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6,
                }}>
                  <strong>Sharpe Stress Test:</strong> Cổ phiếu được gán nhãn SPECULATIVE nếu có điểm lợi nhuận cao (Top 3) nhưng Sharpe lịch sử thấp. Phạt thêm <strong>+1.5 điểm rủi ro</strong> để cảnh báo nguy cơ lợi nhuận quá khứ không bền vững.
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10,
                  padding: '8px 12px', background: '#EFF4FF', borderRadius: 6, borderLeft: '3px solid #A8C0F0',
                }}>Giai đoạn 3 - Lọc rủi ro theo khẩu vị</div>
                {[
                  ['Thận trọng', '#1565C0', 'Chỉ giữ cổ phiếu có Điểm rủi ro ≤ 5.0. Loại bỏ hoàn toàn các cổ phiếu đầu cơ và biến động cao.'],
                  ['Cân bằng', '#2E7D32', 'Giữ cổ phiếu có Điểm rủi ro ≤ 7.0. Cân bằng giữa lợi nhuận và an toàn.'],
                  ['Tích cực', '#C0392B', 'Giữ cổ phiếu có Điểm rủi ro ≤ 7.0, chấp nhận biến động cao hơn để tối đa hóa lợi nhuận.'],
                ].map(([label, color, desc], i) => (
                  <div key={i} style={{
                    display: 'flex', alignItems: 'flex-start', gap: 8, padding: '7px 10px',
                    background: i % 2 === 0 ? '#FDFAF6' : 'transparent', borderRadius: 4, marginBottom: 3,
                  }}>
                    <span style={{ fontWeight: 700, color, whiteSpace: 'nowrap', fontSize: 13 }}>{label}:</span>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {tab === 1 && (
            <div>
              <div style={{ marginBottom: 24 }}>
                <div style={{
                  fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10,
                  padding: '8px 12px', background: '#EDFAF2', borderRadius: 6, borderLeft: '3px solid #A8D5B8',
                }}>Chiến lược Cân bằng: Equal-Weight 1/N</div>
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7, margin: '0 0 10px' }}>
                  Không có bài toán tối ưu hóa. Mỗi cổ phiếu trong Top 10 nhận đúng 10% danh mục, không ưu tiên cổ phiếu nào hơn dựa trên dữ liệu lịch sử.
                </p>
                <div style={{
                  padding: '10px 14px', background: '#F5EFE6', borderRadius: 6,
                  fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 10,
                }}>
                  <strong>Lý thuyết (DeMiguel et al., 2009):</strong> Phân bổ đều 1/N thường thắng MVO out-of-sample vì tránh được sai số tích lũy khi ước lượng ma trận hiệp phương sai Σ. Kiểm chứng trên 14 bộ dữ liệu thực tế: lỗi ước lượng Σ thường lớn hơn lợi ích của tối ưu hóa.
                </div>
                <div style={{
                  padding: '10px 14px', background: '#FFF8E8', borderRadius: 6,
                  fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 10,
                  border: '1px solid #F5DFA0',
                }}>
                  <strong>VIC outlier - ví dụ minh họa:</strong> VIC tăng <strong>+678%</strong> trong giai đoạn kiểm thử (24/01/2025 - 20/04/2026). Equal-Weight phân bổ đúng 10%. MVO Tích cực chỉ phân bổ 2% vì biến động lịch sử của VIC trong 2020-2024 quá cao - SLSQP đánh giá VIC là rủi ro. Một cổ phiếu này đóng góp phần lớn chênh lệch kết quả giữa hai chiến lược.
                </div>
                <div style={{
                  padding: '9px 14px', background: '#EDFAF2', border: '1px solid #A8D5B8',
                  borderRadius: 6, fontSize: 13, color: '#2E7D32', fontWeight: 600,
                }}>
                  Kết quả kiểm thử: Lợi nhuận tích lũy 65.90% · Sharpe Ratio 1.6201 · Biến động 25.30%/năm
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10,
                  padding: '8px 12px', background: '#F5EFE6', borderRadius: 6, borderLeft: '3px solid #C4A265',
                }}>Chiến lược Thận trọng và Tích cực: Markowitz MVO (SLSQP)</div>
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7, margin: '0 0 10px' }}>
                  Tìm tỷ trọng tối ưu bằng thuật toán SLSQP (Sequential Least Squares Programming):
                </p>
                <div style={{
                  background: '#F5EFE6', borderRadius: 6, padding: '10px 14px', marginBottom: 10,
                  fontFamily: 'monospace', fontSize: 13, color: 'var(--text-primary)', textAlign: 'center',
                }}>
                  Tối đa hóa: Sharpe = (w<sup>T</sup> μ<sub>annual</sub> − RF) / √(w<sup>T</sup> Σ<sub>annual</sub> w)
                </div>
                {[
                  ['RF', '4.5%/năm - lãi suất phi rủi ro tương đương trái phiếu Chính phủ VN'],
                  ['Ràng buộc', 'Tổng tỷ trọng = 100% · 2% ≤ w_i ≤ 30% mỗi cổ phiếu · Không bán khống'],
                  ['Pre-annualization', 'μ và Σ nhân 252 trước khi đưa vào SLSQP để gradient ổn định hơn'],
                  ['Thận trọng', 'Thay mục tiêu: tối thiểu hóa phương sai danh mục thay vì tối đa hóa Sharpe, phù hợp ưu tiên ổn định'],
                ].map(([name, desc], i) => (
                  <div key={i} style={{
                    padding: '6px 10px', borderRadius: 4, marginBottom: 3,
                    background: i % 2 === 0 ? '#FDFAF6' : 'transparent',
                    fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5,
                  }}>
                    <strong style={{ color: 'var(--text-primary)' }}>{name}:</strong> {desc}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
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
  const [stratModal,   setStratModal]   = useState(null)  // 'prudent' | 'equal_weight' | 'risk_taking'
  const [scoreModal,   setScoreModal]   = useState(null)  // 'profit' | 'risk'
  const [showAlgoModal,setShowAlgoModal]= useState(false)

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
      setEditWeights(initW); setScenario(null); setWeightError(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally { setLoading(false) }
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

  const retNote = cashPct > 0
    ? `Điều chỉnh ${cashPct}% tiền mặt`
    : scenario ? 'Kịch bản tùy chỉnh'
    : 'Lợi nhuận thực nghiệm kiểm thử'

  const kpiCards = [
    { ...KPI_CARDS[0], value: data ? fmtPct(dispRet) : '-', sub: retNote },
    { ...KPI_CARDS[1], value: data ? fmtPct(dispVol) : '-', sub: 'Độ lệch chuẩn (Standard Deviation)' },
    { ...KPI_CARDS[2], value: data ? dispSharpe.toFixed(4) : '-', sub: 'Risk-Free Rate = 4.5%/năm' },
  ]

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
    if (Math.abs(total - 100) > 1) { setWeightError(`Tổng tỷ trọng = ${total.toFixed(1)}% (cần = 100%)`); return }
    setScenarioLoad(true); setWeightError(null)
    try { setScenario(await computePortfolio(tickers, weights)) }
    catch (err) { setWeightError(err.response?.data?.detail || err.message) }
    finally { setScenarioLoad(false) }
  }

  const stratDetail = stratModal ? PROFILE_DETAIL[stratModal] : null
  const scoreDetail = scoreModal ? SCORE_MODALS[scoreModal]   : null

  return (
    <div>
      {/* ── Modals ── */}
      {stratDetail && (
        <InfoModal
          title={stratDetail.title}
          subtitle={stratDetail.subtitle}
          sections={stratDetail.sections}
          onClose={() => setStratModal(null)}
        />
      )}
      {scoreDetail && (
        <InfoModal
          title={scoreDetail.title}
          subtitle={scoreDetail.subtitle}
          sections={scoreDetail.sections}
          onClose={() => setScoreModal(null)}
        />
      )}
      {showAlgoModal && <AlgoModal onClose={() => setShowAlgoModal(false)} />}

      <h1 className="page-title"><PieIcon size={26} color="var(--gold)" /> Tối ưu hóa danh mục đầu tư</h1>
      <p className="page-subtitle">Phân bổ tài sản thông minh để đạt mục tiêu lợi nhuận và kiểm soát rủi ro</p>

      {/* ── KPI cards ── */}
      <KpiGrid cards={kpiCards} />

      {/* ── Backtest leaderboard ── */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Kết quả Kiểm Thử Thực Nghiệm - Walk-Forward Backtest</div>
        <div className="card-sub" style={{ marginBottom: 4 }}>
          Huấn luyện trên dữ liệu lịch sử <strong>2020–2024</strong> · Kiểm thử trên dữ liệu ngoài mẫu <strong>24/01/2025 – 20/04/2026</strong>.
          Quy tắc: không nhìn trước tương lai (look-ahead bias = 0), cửa sổ trượt 20 phiên, ngưỡng tin cậy ≥ 55%.
          Nhấn vào từng chỉ số để xem giải thích.
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
              {BACKTEST_META.map((row, i) => <MetricRow key={i} row={row} i={i} />)}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
          (*) VN-Index là chuẩn tham chiếu thụ động - không có tín hiệu giao dịch nên không tính Max Drawdown và Win Rate.
        </div>
      </div>

      <div className="portfolio-layout">
        {/* ── Left panel ── */}
        <div className="portfolio-left">

          {/* Strategy selector */}
          <div className="card">
            <div className="card-title">Chiến lược đầu tư</div>
            <div className="card-sub" style={{ marginBottom: 12 }}>Chọn chiến lược phù hợp với khẩu vị rủi ro</div>

            {/* Risk slider */}
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

            {/* Current strategy badge + detail button */}
            {(() => {
              const d = PROFILE_DETAIL[profile]
              return (
                <div style={{
                  background: '#F5EFE6', borderRadius: 8, padding: '9px 12px', marginBottom: 14,
                  borderLeft: `3px solid ${d.badgeColor}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10,
                }}>
                  <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.3 }}>{d.title}</span>
                  <button
                    onClick={() => setStratModal(profile)}
                    style={{
                      flexShrink: 0, fontSize: 11, fontWeight: 600, color: '#fff',
                      background: d.badgeColor, border: 'none', borderRadius: 4,
                      padding: '3px 10px', cursor: 'pointer', whiteSpace: 'nowrap',
                    }}
                  >{d.badge} ↗</button>
                </div>
              )
            })()}

            {/* Cash slider */}
            <div className="form-group">
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
            <button
              style={{
                width: '100%', marginTop: 10, padding: '8px 14px',
                background: 'none', border: '1px solid #D4C090', borderRadius: 8,
                fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer',
                fontWeight: 500,
              }}
              onClick={() => setShowAlgoModal(true)}
            >
              Tại sao mô hình khuyến nghị như vậy?
            </button>
          </div>

          {/* Asset list */}
          {data && (
            <div className="card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                <div className="card-title" style={{ margin: 0 }}>Quản lý tài sản</div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    onClick={() => setScoreModal('profit')}
                    style={{
                      fontSize: 11, fontWeight: 600, color: '#2E7D32',
                      background: '#E8F5E9', border: '1px solid #A8D5B8',
                      borderRadius: 6, padding: '3px 8px', cursor: 'pointer',
                    }}
                  >Điểm lợi nhuận ?</button>
                  <button
                    onClick={() => setScoreModal('risk')}
                    style={{
                      fontSize: 11, fontWeight: 600, color: '#9E7E45',
                      background: '#FFF8E1', border: '1px solid #E8C97A',
                      borderRadius: 6, padding: '3px 8px', cursor: 'pointer',
                    }}
                  >Điểm rủi ro ?</button>
                </div>
              </div>
              <div className="card-sub" style={{ marginBottom: 8 }}>Chỉnh tỷ trọng → Nhấn "Tính kịch bản"</div>

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
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 6 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Tỷ trọng</span>
                          <input
                            type="number" min={0} max={100} step={0.1}
                            value={editWeights[s.ticker] ?? (s.weight * 100).toFixed(1)}
                            onChange={e => handleWeightChange(s.ticker, e.target.value)}
                            style={{
                              width: 52, border: '1px solid #D4C090', borderRadius: 4,
                              padding: '2px 4px', fontSize: 12, textAlign: 'center',
                              background: '#FAF7F2', color: 'var(--text-primary)',
                            }}
                          />
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>%</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>LN</span>
                          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--green)' }}>
                            {prof ? (prof.composite_score * 100).toFixed(0) + '/100' : '-'}
                          </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>RR</span>
                          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--gold-dark)' }}>
                            {rsk ? rsk.composite_risk.toFixed(1) + '/10' : s.risk_score.toFixed(1) + '/10'}
                          </span>
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
              <div className="spinner-wrap"><div className="spinner" />
                <p className="spinner-label">Đang tải dữ liệu danh mục...</p>
              </div>
            </div>
          )}
          {error && <div className="error-box">Lỗi: {error}</div>}

          {data && !loading && (
            <>
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

              <div className="chart-block" style={{ marginBottom: 16 }}>
                <div className="chart-block-title">Điểm lợi nhuận vs điểm rủi ro</div>
                <div className="chart-block-sub">Điểm LN: thứ hạng % 0–100 · Điểm RR: 0–10 (×10 để so sánh)</div>
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

              <div className="card">
                <div className="card-title" style={{ marginBottom: 10 }}>Phương pháp học thuật</div>
                <AccSection id="walkforward" title="Walk-Forward Backtest - Cơ chế kiểm thử" open={!!accOpen.walkforward} onToggle={toggleAcc}>
                  <p>Walk-forward backtest mô phỏng giao dịch thực tế theo thời gian tuyến tính: tại mỗi phiên <em>t</em>, mô hình chỉ được phép sử dụng dữ liệu tính đến phiên <em>t</em> để phát tín hiệu, sau đó đánh giá kết quả tại phiên <em>t+1</em>. Không có look-ahead bias.</p>
                  <p style={{ marginTop: 8 }}>Thiết lập: cửa sổ nhìn lại = 20 phiên · Conviction ≥ 0.55 · Giai đoạn kiểm thử: 24/01/2025 – 20/04/2026 · Dữ liệu kiểm thử tách biệt hoàn toàn với dữ liệu huấn luyện (2020–2024).</p>
                </AccSection>
                <AccSection id="sharpe" title="Sharpe Ratio - Đo lường hiệu quả điều chỉnh rủi ro" open={!!accOpen.sharpe} onToggle={toggleAcc}>
                  <p>Công thức (đầu vào đã được quy về năm):</p>
                  <div style={{
                    background: '#F5EFE6', borderRadius: 6, padding: '8px 12px', margin: '8px 0',
                    fontFamily: 'monospace', fontSize: 13, color: 'var(--text-primary)', textAlign: 'center',
                  }}>
                    Sharpe = (μ<sub>năm</sub> − RF<sub>năm</sub>) / σ<sub>năm</sub>
                  </div>
                  <p>μ<sub>năm</sub> = lợi nhuận ngày × 252, σ<sub>năm</sub> = độ lệch chuẩn ngày × √252, RF<sub>năm</sub> = 4.5%/năm. Do μ và σ đã được quy về năm trước khi tính, công thức không cần thêm hệ số √252.</p>
                  <p style={{ marginTop: 8 }}>Equal-Weight Top 10 đạt Sharpe <strong>1.6201</strong> - vượt VN-Index (1.1694) và cả hai chiến lược Markowitz MVO.</p>
                </AccSection>
                <AccSection id="mvo" title="Tại sao Equal-Weight thắng Markowitz MVO?" open={!!accOpen.mvo} onToggle={toggleAcc}>
                  <p><strong>Lý thuyết (DeMiguel et al., 2009):</strong> Sai số ước lượng ma trận hiệp phương sai tích lũy và thường làm cho 1/N Equal-Weight thắng out-of-sample - kiểm chứng trên 14 bộ dữ liệu thực tế.</p>
                  <p style={{ marginTop: 8 }}><strong>Yếu tố cụ thể - VIC outlier:</strong> VIC tăng +678% trong giai đoạn kiểm thử. Equal-Weight phân bổ 10% cho VIC; MVO chỉ phân bổ 2% do biến động lịch sử cao. Một cổ phiếu này đóng góp phần lớn chênh lệch.</p>
                  <p style={{ marginTop: 8 }}><em>"No investment strategy can be expected to consistently outperform the 1/N naive diversification rule."</em> - DeMiguel, Garlappi &amp; Uppal (2009).</p>
                </AccSection>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
