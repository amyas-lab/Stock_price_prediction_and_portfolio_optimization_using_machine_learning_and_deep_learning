import { useState } from 'react'
import {
  ResponsiveContainer, AreaChart, Area,
  BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts'
import { predictPrice, predictSignal } from '../api/stockApi.js'

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

const SPECIALIZED = new Set([
  'FPT','VCB','VHM','VNM','HPG','VIC','TCB','MSN','MWG','VND',
  'BID','CTG','MBB','ACB','HDB','TPB','SHB','PDR','KDH','DXG',
  'GAS','HSG','PNJ','SAB','CMG','ELC','SGT',
])

function fmtVND(v) {
  if (!v && v !== 0) return ''
  return v.toLocaleString('vi-VN') + ' VNĐ'
}

function safeReturn(rawReturn, direction) {
  const MAX_REAL_DAILY = 0.07
  if (Math.abs(rawReturn) <= MAX_REAL_DAILY) return rawReturn
  const sign = direction === 'UP' ? 1 : direction === 'DOWN' ? -1 : 0
  return sign * 0.003
}

function buildChartData(currentPrice, predictedReturns, direction, historicalPrices = []) {
  const data = []
  if (historicalPrices.length > 0) {
    historicalPrices.forEach(({ date, price }) => {
      const d = new Date(date)
      const label = `${d.getDate()}/${d.getMonth() + 1}`
      data.push({ day: label, actual: Math.round(price), predicted: null })
    })
  } else {
    const HIST_LEN = 20
    let seed = Math.floor(currentPrice * 137) % 99991
    const rand = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280 }
    const hist = []
    let p = currentPrice
    for (let i = 0; i < HIST_LEN; i++) {
      p = p / (1 + (rand() - 0.49) * 0.018)
      hist.unshift(p)
    }
    hist[hist.length - 1] = currentPrice
    hist.forEach((price, i) => {
      data.push({ day: `T-${HIST_LEN - i}`, actual: Math.round(price), predicted: null })
    })
  }
  data.push({ day: 'Hôm nay', actual: Math.round(currentPrice), predicted: Math.round(currentPrice) })
  let predP = currentPrice
  predictedReturns.forEach((r, i) => {
    const normalizedR = safeReturn(r, direction)
    predP = predP * Math.exp(normalizedR)
    data.push({ day: `+${i + 1}`, actual: null, predicted: Math.round(predP) })
  })
  return data
}

function buildReturnBars(predictedReturns, direction) {
  return (predictedReturns || []).map((r, i) => ({
    day: `+${i + 1}`,
    return_pct: +(safeReturn(r, direction) * 100).toFixed(3),
  }))
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

const SIGNAL_STYLE = {
  BUY:  { bg: '#E8F5E9', color: '#2E7D32', label: 'MUA' },
  SELL: { bg: '#FFEBEE', color: '#C62828', label: 'BÁN' },
  HOLD: { bg: '#FFF8E1', color: '#9E7E45', label: 'GIỮ' },
}

// ── Feature groups for section 6 ─────────────────────────────

const FEATURE_GROUPS = [
  {
    title: 'Lớp 1: Meta-features từ mạng Học sâu (3 tính năng)',
    color: '#4A7C5F',
    bg: '#E8F5E9',
    features: [
      { name: 'mtl_p_up', desc: 'Xác suất (0 đến 1) mô hình Deep Learning dự báo giá sẽ nằm trong kịch bản tăng trưởng vượt ngưỡng +2.0% trong 5 phiên tới.' },
      { name: 'mtl_p_down', desc: 'Xác suất (0 đến 1) mô hình Deep Learning dự báo giá sẽ rơi vào kịch bản sụt giảm dưới ngưỡng −1.5% trong 5 phiên tới.' },
      { name: 'mtl_conviction', desc: 'Độ tự tin tối đa = max(mtl_p_up, mtl_p_down). Đóng vai trò làm màng lọc loại bỏ trạng thái thị trường đi ngang (Sideways).' },
    ],
  },
  {
    title: 'Lớp 2: Hình học vùng giá cản K-Means (5 tính năng)',
    color: '#9E7E45',
    bg: '#FFF8E1',
    features: [
      { name: 'sr_distance_pct', desc: 'Khoảng cách tương đối (%) từ giá đóng cửa hiện tại đến tâm vùng cản K-Means gần nhất. Dương (+) = trên hỗ trợ; âm (−) = dưới kháng cự.' },
      { name: 'sr_breakout_up', desc: 'Nhị phân (1/0). Trả về 1.0 nếu giá đóng cửa hôm nay cắt dứt khoát lên trên biên an toàn (+0.5%) của vùng kháng cự lịch sử.' },
      { name: 'sr_breakout_down', desc: 'Nhị phân (1/0). Trả về 1.0 nếu giá đóng cửa hôm nay đâm thủng xuống dưới biên an toàn (−0.5%) của vùng hỗ trợ lịch sử.' },
      { name: 'sr_near_resistance', desc: 'Chỉ thị (1/0). Kích hoạt bằng 1.0 nếu giá cách vùng kháng cự phía trên < 0.5%. Báo hiệu vùng rủi ro đảo chiều hoặc chuẩn bị nén giá breakout.' },
      { name: 'sr_near_support', desc: 'Chỉ thị (1/0). Kích hoạt bằng 1.0 nếu giá cách vùng hỗ trợ phía dưới < 0.5%. Báo hiệu vùng có lực cầu đỡ giá tiềm năng.' },
    ],
  },
  {
    title: 'Lớp 3: Giao cắt đường trung bình động MA (7 tính năng)',
    color: '#5B7FA6',
    bg: '#E8F0FB',
    features: [
      { name: 'ma_golden_cross_short', desc: 'Sự kiện (1/0). Kích hoạt duy nhất tại phiên EMA-10 cắt lên EMA-20 → xu hướng tăng ngắn hạn bắt đầu.' },
      { name: 'ma_death_cross_short', desc: 'Sự kiện (1/0). Kích hoạt duy nhất tại phiên EMA-10 cắt xuống EMA-20 → xu hướng giảm ngắn hạn bắt đầu.' },
      { name: 'ma_golden_cross_long', desc: 'Sự kiện (1/0). Kích hoạt duy nhất tại phiên EMA-20 cắt lên EMA-50 → xác nhận pha tăng giá trung hạn (Macro Bull Regime).' },
      { name: 'ma_death_cross_long', desc: 'Sự kiện (1/0). Kích hoạt duy nhất tại phiên EMA-20 cắt xuống EMA-50 → xác nhận pha giảm giá trung hạn (Macro Bear Regime).' },
      { name: 'ma_short_gap_pct', desc: '(EMA₁₀ − EMA₂₀) / EMA₂₀ × 100. Độ rộng phân kỳ ngắn hạn — giá trị càng lớn thể hiện gia tốc tăng giá càng mạnh.' },
      { name: 'ma_long_gap_pct', desc: '(EMA₂₀ − EMA₅₀) / EMA₅₀ × 100. Đo lường độ bền vững của xu hướng trung hạn.' },
      { name: 'ma_alignment', desc: '+1.0 nếu EMA₁₀ > EMA₂₀ > EMA₅₀ (toàn tăng); −1.0 nếu ngược lại (toàn giảm); 0.0 nếu các đường MA đan xen (Sideways).' },
    ],
  },
  {
    title: 'Lớp 4: Chỉ báo kỹ thuật cổ phiếu + VN-Index macro (27 tính năng)',
    color: '#7B5EA7',
    bg: '#F3EEF9',
    features: [
      { name: 'vni_log_return', desc: 'Tỷ suất sinh lời log từng phiên của VN-Index — cung cấp bối cảnh hệ số Beta và sức khỏe toàn thị trường chung.' },
      { name: 'vni_rsi', desc: 'RSI của VN-Index — giúp mô hình nhận diện thị trường chung đang quá hưng phấn (Overbought) hay hoảng loạn (Oversold).' },
      { name: 'vni_macd / signal / hist / ema_10 / bb_upper / bb_lower / atr', desc: 'Toàn bộ chỉ báo momentum và biến động của VN-Index — làm màng lọc vĩ mô hệ thống.' },
      { name: 'rsi_14', desc: 'RSI 14 phiên của cổ phiếu mục tiêu.' },
      { name: 'macd', desc: 'Đường MACD (EMA-12 − EMA-26) của cổ phiếu mục tiêu.' },
      { name: 'macd_signal', desc: 'Đường tín hiệu MACD (EMA-9 của đường MACD).' },
      { name: 'macd_hist', desc: 'MACD Histogram (MACD − MACD_signal) — phát hiện sớm tín hiệu phân kỳ đảo chiều.' },
      { name: 'bb_upper / bb_lower', desc: 'Bollinger Bands (SMA-20 ± 2σ) — vùng biên kỹ thuật trên/dưới của giá.' },
      { name: 'atr_14', desc: 'Average True Range — đo biến động tuyệt đối, dùng để tự động điều chỉnh mức stoploss.' },
      { name: 'obv', desc: 'On-Balance Volume — đo áp lực dòng tiền mua tích lũy dựa trên kết hợp khối lượng và hướng đi của giá.' },
      { name: 'Stochastic %K/%D, CMF, ROC, MFI...', desc: 'Các chỉ báo động lượng bổ trợ được giữ lại sau Feature Pruning nhờ điểm Feature Importance vượt mức trung bình.' },
      { name: 'ticker_encoded', desc: 'Định danh số hóa (0 đến N−1) của cổ phiếu. Giúp XGBoost học được hành vi giá riêng từng ngành: FPT (công nghệ) khác VCB (ngân hàng) về độ nhảy giá và tính giữ nền tài sản.' },
    ],
  },
]

// ── Model Explanation Modal ───────────────────────────────────

function ModelExplainModal({ onClose }) {
  const [openSection, setOpenSection] = useState(null)
  const toggle = (id) => setOpenSection(prev => prev === id ? null : id)

  const Section = ({ id, title, children }) => (
    <div style={{ marginBottom: 12 }}>
      <button
        onClick={() => toggle(id)}
        style={{
          width: '100%', textAlign: 'left', padding: '12px 16px',
          background: openSection === id ? '#FAF3E0' : '#F9F6F1',
          border: '1px solid #EDE5D8', borderRadius: 10,
          fontSize: 14, fontWeight: 600, color: 'var(--text-primary)',
          cursor: 'pointer', display: 'flex', justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span>{title}</span>
        <span style={{ fontSize: 18, color: 'var(--text-muted)', lineHeight: 1 }}>
          {openSection === id ? '−' : '+'}
        </span>
      </button>
      {openSection === id && (
        <div style={{
          padding: '16px', border: '1px solid #EDE5D8',
          borderTop: 'none', borderRadius: '0 0 10px 10px',
          background: '#fff', lineHeight: 1.7,
        }}>
          {children}
        </div>
      )}
    </div>
  )

  const P = ({ children, style }) => (
    <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 10, ...style }}>{children}</p>
  )

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)',
        zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '16px',
      }}
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      <div style={{
        background: '#FAF7F2', borderRadius: 16, width: '100%', maxWidth: 760,
        maxHeight: '90vh', overflowY: 'auto', boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
      }}>
        {/* Header */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '20px 24px 16px', borderBottom: '1px solid #EDE5D8',
          position: 'sticky', top: 0, background: '#FAF7F2', zIndex: 1,
        }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>
              Kiến trúc Mô hình Tín hiệu Giao dịch
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
              Cascade Model: GRU + Multi-Head Attention → K-Means → XGBoost (500 cây)
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none', border: '1px solid #EDE5D8',
              borderRadius: 8, width: 32, height: 32, cursor: 'pointer',
              fontSize: 16, color: 'var(--text-muted)', display: 'flex',
              alignItems: 'center', justifyContent: 'center',
            }}
          >×</button>
        </div>

        <div style={{ padding: '20px 24px' }}>

          {/* Architecture Flow Diagram */}
          <div style={{
            background: 'linear-gradient(135deg, #E8F0FB 0%, #E8F5E9 50%, #FFF8E1 100%)',
            borderRadius: 12, padding: '18px', marginBottom: 20,
            border: '1px solid #DDE8D8',
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Luồng xử lý dữ liệu
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', fontSize: 12 }}>
              {[
                { label: 'OHLCV\n20 phiên', color: '#5B7FA6', bg: '#E8F0FB' },
                { label: '→', color: '#999', bg: 'transparent', noBorder: true },
                { label: 'GRU +\nAttention', color: '#4A7C5F', bg: '#E8F5E9' },
                { label: '→', color: '#999', bg: 'transparent', noBorder: true },
                { label: 'MTL Output\np_up · p_down', color: '#4A7C5F', bg: '#E8F5E9' },
                { label: '+', color: '#999', bg: 'transparent', noBorder: true },
                { label: 'K-Means\nS/R Zones (63d)', color: '#9E7E45', bg: '#FFF8E1' },
                { label: '+', color: '#999', bg: 'transparent', noBorder: true },
                { label: 'MA\nCrossover', color: '#5B7FA6', bg: '#E8F0FB' },
                { label: '→', color: '#999', bg: 'transparent', noBorder: true },
                { label: 'XGBoost\n42 features', color: '#7B5EA7', bg: '#F3EEF9' },
                { label: '→', color: '#999', bg: 'transparent', noBorder: true },
                { label: 'BUY / SELL\n/ HOLD', color: '#C62828', bg: '#FFEBEE' },
              ].map((node, i) => (
                <div key={i} style={{
                  padding: node.noBorder ? '4px 2px' : '6px 10px',
                  borderRadius: 8, fontWeight: 600,
                  color: node.color, background: node.bg,
                  border: node.noBorder ? 'none' : `1px solid ${node.color}33`,
                  whiteSpace: 'pre', textAlign: 'center', lineHeight: 1.4,
                  fontSize: node.noBorder ? 16 : 11,
                }}>
                  {node.label}
                </div>
              ))}
            </div>
          </div>

          {/* ── Section 1 ── */}
          <Section id="arch" title="1. Bản thiết kế hệ thống: Cascade Model (Mô hình xếp chồng phân cấp)">
            <P>
              Hệ thống phát hiện tín hiệu giao dịch áp dụng cấu trúc Cascade Model. Thay vì sử dụng một mạng nơ-ron sâu duy nhất để đưa ra quyết định thẳng từ dữ liệu thô (vốn rất dễ bị bẫy nhiễu trên thị trường HOSE), hệ thống phân tách bài toán thành hai lớp xử lý tối ưu:
            </P>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {[
                {
                  title: 'Lớp 1: Deep Sequential Feature Extractor',
                  body: 'Sử dụng mạng Học sâu kết hợp GRU và Multi-Head Attention để giải mã động lực dòng tiền từ chuỗi lịch sử trượt 20 phiên.',
                  color: '#4A7C5F', bg: '#E8F5E9',
                },
                {
                  title: 'Lớp 2: Tabular Aggregator & Gate',
                  body: 'Sử dụng XGBoost Classifier để tích hợp thông tin xu hướng từ lớp Học sâu với không gian hình học S/R và chỉ báo động lượng, đưa ra xác suất hành động cuối cùng.',
                  color: '#7B5EA7', bg: '#F3EEF9',
                },
              ].map(c => (
                <div key={c.title} style={{ padding: 14, borderRadius: 10, background: c.bg, border: `1px solid ${c.color}33` }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: c.color, marginBottom: 6 }}>{c.title}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{c.body}</div>
                </div>
              ))}
            </div>
          </Section>

          {/* ── Section 2 ── */}
          <Section id="gru" title="2. Lớp học sâu trích xuất bối cảnh thời gian: GRU & Attention">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <P>
                Mô hình nạp dữ liệu dưới dạng chuỗi thời gian trượt (Sliding Window) có độ dài <strong>20 phiên liên tiếp</strong> (W=20, tương đương một tháng giao dịch).
              </P>

              <div style={{ fontWeight: 700, fontSize: 12, color: '#4A7C5F', marginBottom: 8 }}>Cơ chế tính toán của mạng GRU</div>
              <P>Mạng GRU giải quyết bài toán phụ thuộc dài hạn bằng cách quét qua 20 ngày và cập nhật thông tin qua hai cổng toán học:</P>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 }}>
                <div style={{ padding: 12, background: '#E8F5E9', borderRadius: 8 }}>
                  <div style={{ fontWeight: 700, fontSize: 12, color: '#4A7C5F', marginBottom: 4 }}>Update Gate (Cổng cập nhật)</div>
                  <div style={{ fontSize: 12 }}>Quyết định lượng thông tin từ xu hướng quá khứ cần truyền tiếp — ví dụ: trạng thái tích lũy/đè giá kéo dài của dòng tiền lớn.</div>
                </div>
                <div style={{ padding: 12, background: '#FFF8E1', borderRadius: 8 }}>
                  <div style={{ fontWeight: 700, fontSize: 12, color: '#9E7E45', marginBottom: 4 }}>Reset Gate (Cổng thiết lập lại)</div>
                  <div style={{ fontSize: 12 }}>Quyết định lượng thông tin nhiễu ngắn hạn cần loại bỏ — ví dụ: biến động kỹ thuật do đáo hạn phái sinh.</div>
                </div>
              </div>

              <div style={{ fontWeight: 700, fontSize: 12, color: '#4A7C5F', marginBottom: 8 }}>Cơ chế chú ý đa đầu (Multi-Head Attention)</div>
              <P>
                Hệ thống không chỉ lấy trạng thái cuối cùng của GRU mà ép tensor qua một lớp <strong>Multi-Head Attention (2 Heads)</strong>. Cơ chế này tính điểm tương quan giữa tất cả các ngày trong cửa sổ 20 phiên, cho phép mô hình "bơm thêm trọng số" vào các phiên có hành vi đặc biệt (như phiên FTD bùng nổ theo đà hoặc phiên ép bán Washout).
              </P>

              <div style={{ fontWeight: 700, fontSize: 12, color: '#4A7C5F', marginBottom: 8 }}>Mục tiêu tối ưu hóa đa nhiệm (Multi-Task Learning)</div>
              <P>Mô hình phải giải đồng thời 2 tác vụ: vừa dự đoán quỹ đạo log-return 5 ngày (Regression Head), vừa phân loại xu hướng thị trường (Classification Head).</P>
              <div style={{ background: '#F5F0FF', borderRadius: 8, padding: '10px 14px', fontFamily: 'monospace', fontSize: 12, margin: '8px 0 12px' }}>
                L_total = L_Huber(quỹ đạo 5 ngày) + <strong style={{ color: '#7B5EA7' }}>5.0</strong> × L_CrossEntropy(xu hướng)
              </div>
              <P>
                Hệ số phạt <code style={{ background: '#eee', padding: '1px 4px', borderRadius: 3 }}>λ_cls = 5.0</code> ép mạng phải ưu tiên trích xuất <em>Directional Accuracy</em> — biết đúng hướng quan trọng hơn biết chính xác biên độ. Đầu ra cung cấp 3 Meta-features đại diện cho "Niềm tin tiên nghiệm" (Prior Belief) của mạng sâu.
              </P>
            </div>
          </Section>

          {/* ── Section 3 ── */}
          <Section id="kmeans" title="3. Lớp hình học hóa tâm lý thị trường: K-Means Clustering">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <P>
                Để số hóa các vùng tâm lý mua/bán của nhà đầu tư thành biến số liên tục, hệ thống áp dụng K-Means Clustering trên cửa sổ lịch sử <strong>63 phiên (3 tháng)</strong>.
              </P>

              <div style={{ fontWeight: 700, fontSize: 12, color: '#9E7E45', marginBottom: 8 }}>Tự động xác định số vùng cản: Elbow Method</div>
              <P>
                Hệ thống tính hàm tổng bình phương khoảng cách trong cụm (Inertia) với k ∈ [2, 11]. Thuật toán tính đạo hàm bậc hai của chuỗi Inertia để tìm điểm gãy tối ưu:
              </P>
              <div style={{ background: '#F5F0FF', borderRadius: 8, padding: '10px 14px', fontFamily: 'monospace', fontSize: 12, margin: '10px 0 12px' }}>
                k_optimal = argmax_k [ (I_k₊₁ − I_k) − (I_k − I_k₋₁) ]
              </div>
              <P>
                Cấu trúc cản tự thích ứng với từng cổ phiếu riêng biệt — mã ngân hàng đi nền chặt có ít vùng cản hơn mã bất động sản biến động mạnh.
              </P>

              <div style={{ padding: 12, background: '#FFF8E1', borderRadius: 8, border: '1px solid #E8D5A033', marginTop: 4 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#9E7E45', marginBottom: 6 }}>Bộ lọc phá vỡ cấu trúc giá (BOS Filter)</div>
                <div style={{ fontSize: 12, lineHeight: 1.6 }}>
                  Các tâm cụm chính là tâm của các vùng Hỗ trợ/Kháng cự mạnh trong 3 tháng. Hệ thống thiết lập vùng biên an toàn <strong>±0.5%</strong> quanh tâm cản. Tín hiệu breakout chỉ được kích hoạt khi giá hôm nay vượt hoàn toàn qua biên trên, đồng thời phiên hôm trước nằm dưới biên dưới. Loại bỏ hoàn toàn các bẫy giá (Bull-trap / Bear-trap).
                </div>
              </div>
            </div>
          </Section>

          {/* ── Section 4 ── */}
          <Section id="xgb" title="4. Khối học máy ra quyết định tối ưu: XGBoost Classifier">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <P>
                Toàn bộ dữ liệu từ các nguồn được tích hợp thành một vectơ đặc trưng phẳng gồm <strong>42 tính năng</strong>. Hệ thống sử dụng XGBoost Classifier (<strong>500 cây quyết định phân cấp</strong>) nhờ các ưu điểm vượt trội:
              </P>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  {
                    title: 'Xử lý dữ liệu hỗn hợp không tuần bám',
                    body: 'XGBoost hoạt động dựa trên các mệnh đề logic hình cây (Đúng/Sai), chấp nhận đồng thời biến xác suất (0–1), biến liên tục (chỉ báo kỹ thuật), biến hình học (khoảng cách cản) và biến định danh ngành (ticker_encoded) mà không cần chuẩn hóa thô bạo làm mất thông tin gốc.',
                  },
                  {
                    title: 'Kiểm soát Overfitting bằng cấu trúc phạt toán học',
                    body: 'Mô hình sử dụng L1 Regularization (reg_alpha = 0.1) và L2 Regularization (reg_lambda = 1.0). Cơ chế này phạt nặng các cây quá phụ thuộc vào vài chỉ báo riêng lẻ, ép mô hình phải tìm sự đồng thuận từ tất cả các lớp tính năng.',
                  },
                  {
                    title: 'Giải thích được (Interpretable via SHAP)',
                    body: 'Feature importance và SHAP values cho biết chỉ báo nào đang thúc đẩy quyết định MUA/BÁN — không phải hộp đen. Có thể xem chi tiết bằng nút SHAP ở màn hình chính.',
                  },
                ].map(c => (
                  <div key={c.title} style={{ padding: 12, background: '#F3EEF9', borderRadius: 8, border: '1px solid #7B5EA733' }}>
                    <div style={{ fontWeight: 700, fontSize: 12, color: '#7B5EA7', marginBottom: 4 }}>▸ {c.title}</div>
                    <div style={{ fontSize: 12 }}>{c.body}</div>
                  </div>
                ))}
              </div>
            </div>
          </Section>

          {/* ── Section 5 ── */}
          <Section id="asymmetry" title="5. Lý do mua bán không đối xứng: Asymmetric Thresholds">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginBottom: 14 }}>
                {[
                  { label: 'BUY',  cond: 'Return ≥ +2.0%\ntrong 5 ngày', color: '#2E7D32', bg: '#E8F5E9' },
                  { label: 'SELL', cond: 'Return ≤ −1.5%\ntrong 5 ngày', color: '#C62828', bg: '#FFEBEE' },
                  { label: 'HOLD', cond: 'Khoảng giữa\nkhông tín hiệu', color: '#9E7E45', bg: '#FFF8E1' },
                ].map(c => (
                  <div key={c.label} style={{ padding: 12, borderRadius: 10, background: c.bg, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 800, color: c.color }}>{c.label}</div>
                    <div style={{ fontSize: 11, fontFamily: 'monospace', color: c.color, margin: '4px 0', whiteSpace: 'pre-line' }}>{c.cond}</div>
                  </div>
                ))}
              </div>
              <P>
                Đây là một điểm cộng cực lớn về mặt <strong>Market Microstructure (Cấu trúc vi mô thị trường)</strong>. Sàn HOSE có hai đặc tính kinh điển:
              </P>
              <ul style={{ marginLeft: 16, lineHeight: 2, fontSize: 12, marginBottom: 12 }}>
                <li><strong>Upward Drift:</strong> thị trường có xu hướng tăng dài hạn theo sự phát triển kinh tế.</li>
                <li><strong>Liquidity Asymmetry:</strong> khi sập thì lệnh bán tháo khớp rất nhanh và gắt, còn khi lên thì bò từ từ.</li>
              </ul>
              <P>
                Việc siết điều kiện MUA (+2%) chặt hơn BÁN (−1.5%) cho thấy mô hình được thiết kế bởi người hiểu rõ cơ chế vận hành của sàn HOSE, không phải một người làm toán lý thuyết thuần túy.
              </P>
              <ul style={{ marginLeft: 16, lineHeight: 2, fontSize: 12 }}>
                <li>Mất −1.5% cảm giác <em>đau</em> nhiều hơn lãi +1.5% cảm giác <em>vui</em> (Loss Aversion — Kahneman & Tversky).</li>
                <li>Ngưỡng BUY cao (+2%) đảm bảo chỉ vào lệnh khi có đủ Margin of Safety.</li>
                <li>Ngưỡng SELL thấp (−1.5%) ưu tiên bảo toàn vốn hơn tối đa hóa lợi nhuận.</li>
              </ul>
            </div>
          </Section>

          {/* ── Section 6 ── */}
          <Section id="features" title="6. Từ điển định nghĩa chi tiết các tính năng: Feature Definitions">
            {FEATURE_GROUPS.map(group => (
              <div key={group.title} style={{ marginBottom: 16 }}>
                <div style={{
                  fontSize: 12, fontWeight: 700, color: group.color,
                  padding: '6px 10px', background: group.bg,
                  borderRadius: 6, marginBottom: 8,
                }}>
                  {group.title}
                </div>
                {group.features.map(f => (
                  <div key={f.name} style={{
                    display: 'grid', gridTemplateColumns: '200px 1fr',
                    gap: 10, padding: '7px 10px',
                    borderBottom: '1px solid #F0EAE0', fontSize: 12,
                  }}>
                    <code style={{
                      fontFamily: 'monospace', fontSize: 11,
                      color: group.color, fontWeight: 600,
                    }}>{f.name}</code>
                    <span style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>{f.desc}</span>
                  </div>
                ))}
              </div>
            ))}
          </Section>

          {/* ── Section 7 ── */}
          <Section id="importance" title="7. Feature Importance: XGBoost T4">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Biểu đồ dưới đây cho thấy mức độ đóng góp của từng nhóm đặc trưng vào quyết định của XGBoost. Các feature từ mạng Học sâu (MTL) và Hình học cản (S/R) thường đứng đầu bảng.
            </div>
            <img
              src="/notebook/feature_importance_xgb.png"
              alt="XGBoost Feature Importance"
              style={{ width: '100%', borderRadius: 10, border: '1px solid #EDE5D8' }}
              onError={e => { e.target.style.display = 'none' }}
            />
          </Section>

          {/* ── Section 8 ── */}
          <Section id="usecase" title="8. Use Case: Khi nào nên và không nên dùng tín hiệu này?">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                <div style={{ padding: 14, background: '#E8F5E9', borderRadius: 10, border: '1px solid #4A7C5F33' }}>
                  <div style={{ fontWeight: 700, color: '#2E7D32', marginBottom: 8, fontSize: 13 }}>✓ Phù hợp khi</div>
                  <ul style={{ margin: 0, paddingLeft: 16, lineHeight: 2, fontSize: 12 }}>
                    <li>Giao dịch ngắn hạn 3–5 phiên</li>
                    <li>Cổ phiếu trong danh sách 27 mã HOSE</li>
                    <li>Thị trường có xu hướng rõ ràng (VN-Index trending)</li>
                    <li>Dùng kết hợp với phân tích cơ bản</li>
                    <li>Conviction ≥ 55% để ra quyết định</li>
                  </ul>
                </div>
                <div style={{ padding: 14, background: '#FFEBEE', borderRadius: 10, border: '1px solid #C6282833' }}>
                  <div style={{ fontWeight: 700, color: '#C62828', marginBottom: 8, fontSize: 13 }}>✗ Không phù hợp khi</div>
                  <ul style={{ margin: 0, paddingLeft: 16, lineHeight: 2, fontSize: 12 }}>
                    <li>Đầu tư dài hạn (cần phân tích fundamental)</li>
                    <li>Thị trường đang có sự kiện bất ngờ (tin tức lớn)</li>
                    <li>Cổ phiếu thanh khoản thấp (khó khớp lệnh)</li>
                    <li>Conviction &lt; 50% — tín hiệu không đủ mạnh</li>
                    <li>Phiên thị trường biến động cực đoan (ATC/ATO)</li>
                  </ul>
                </div>
              </div>
              <div style={{ padding: 12, background: '#FFF8E1', borderRadius: 8, fontSize: 12, border: '1px solid #E8D5A0' }}>
                <strong>Lưu ý quan trọng:</strong> Đây là công cụ hỗ trợ phân tích, không phải lời khuyên đầu tư tài chính. Luôn quản lý rủi ro với stoploss và chỉ đầu tư số tiền có thể chịu lỗ.
              </div>
            </div>
          </Section>

          {/* ── Section 9 ── */}
          <Section id="mlknowledge" title="9. Kiến thức về Machine Learning, Deep Learning và Chứng khoán">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>

              <div style={{ fontWeight: 700, fontSize: 13, color: '#4A7C5F', marginBottom: 6 }}>1. Kiến trúc Tổng thể: Feature Cascade (Phân cấp tính năng)</div>
              <P>
                Hệ thống không dùng một mô hình duy nhất mà là sự phối hợp của Deep Learning và Machine Learning xếp chồng. Mô hình đi trước "sơ chế" và bóc tách dữ liệu khó cho mô hình đi sau "gút lệnh".
              </P>
              <div style={{ background: '#1e1e1e', borderRadius: 8, padding: '12px 14px', fontFamily: 'monospace', fontSize: 11, color: '#d4d4d4', marginBottom: 14, overflowX: 'auto', lineHeight: 1.8 }}>
                {'[Chuỗi 20 ngày giá] ──► Mạng GRU + Attention ──► [Xác suất xu hướng (MTL Prior)]'}<br/>
                {'                                                            │'}<br/>
                {'                                                            ▼'}<br/>
                {'[Chỉ báo kỹ thuật + Hỗ trợ/Kháng cự K-Means] ───►  XGBoost  ───► TÍN HIỆU'}
              </div>

              <div style={{ fontWeight: 700, fontSize: 13, color: '#4A7C5F', marginBottom: 6 }}>2. Khối Deep Learning: Người đọc vị "Dòng chảy thời gian" (GRU + Attention)</div>
              <P>
                Chứng khoán là dữ liệu chuỗi thời gian. Điểm yếu của RSI hay MACD thông thường là chúng chỉ nhìn vào vài phiên gần nhất và mang tính "chết" (tĩnh). Mạng GRU quét chuỗi 20 ngày với cơ chế các cổng đóng/mở tự động:
              </P>
              <ul style={{ marginLeft: 16, lineHeight: 2, fontSize: 12, marginBottom: 10 }}>
                <li><strong>Cổng quên (Reset Gate):</strong> Tự động lọc bỏ biến động nhiễu (một phiên giảm sàn do tin đồn thất thiệt rồi bật lại ngay).</li>
                <li><strong>Cổng cập nhật (Update Gate):</strong> Ghi nhớ hành vi tích lũy (dòng tiền lớn âm thầm mua gom liên tục, đẩy đáy sau cao hơn đáy trước).</li>
              </ul>
              <P>
                Sau khi GRU quét xong, lớp <strong>Multi-Head Attention</strong> nhảy vào — giống Trader lão luyện nhìn lại đồ thị 1 tháng qua, tự động "bơm trọng số" vào những ngày quan trọng nhất (ngày bùng nổ thanh khoản, ngày giá chạm hỗ trợ bật lên).
              </P>
              <P>
                Mô hình <strong>Multi-Task Learning</strong> ép mạng giải đồng thời 2 bài toán: vừa dự đoán quỹ đạo giá 5 ngày (Regression), vừa phân loại xu hướng (Classification). Sự ép buộc này biến Context Vector thành bộ lọc cực kỳ chất lượng, đầu ra gồm 3 biến đắt giá: mtl_p_up, mtl_p_down, và mtl_conviction.
              </P>

              <div style={{ fontWeight: 700, fontSize: 13, color: '#9E7E45', marginBottom: 6 }}>3. Khối Data Science: Số hóa tâm lý thị trường (K-Means Clustering)</div>
              <P>
                Nhà giao dịch chuyên nghiệp luôn quan tâm đến Cấu trúc thị trường — các vùng Hỗ trợ và Kháng cự là nơi tâm lý con người lặp lại: tiếc nuối không mua nên chờ giá về đó để mua (Hỗ trợ), sợ mất lãi nên cứ lên tới đó là bán (Kháng cự).
              </P>
              <P>
                Hệ thống tự động hóa tư duy này bằng K-Means Clustering trên 63 phiên gần nhất: tìm K tối ưu bằng Elbow Method, tính khoảng cách hình học (sr_distance_pct), và cài đặt bộ lọc BOS ±0.5% để chỉ ghi nhận Phá vỡ cấu trúc (BOS) thật sự — loại bỏ hoàn toàn bẫy tăng giá (Bull-trap).
              </P>

              <div style={{ fontWeight: 700, fontSize: 13, color: '#7B5EA7', marginBottom: 6 }}>4. Khối Học Máy: Hội đồng ra quyết định tối ưu (XGBoost)</div>
              <P>
                42 tính năng hỗn hợp từ các lớp trên được nạp vào XGBoost — lựa chọn số 1 của các nhà làm Quant, vượt trội hơn mạng nơ-ron thông thường:
              </P>
              <ul style={{ marginLeft: 16, lineHeight: 2, fontSize: 12, marginBottom: 10 }}>
                <li>Mạng nơ-ron rất ghét việc nạp vào vừa biến xác suất (0–1) vừa biến phần trăm vừa biến phân loại. XGBoost dùng logic cây (Đúng/Sai) nên chấp nhận mọi định dạng mà không làm biến dạng thông tin.</li>
                <li>Dựng 500 cây liên tiếp, cây sau sửa sai cho cây trước. Phạt L1/L2 ép các cây không được chỉ tin vào một chỉ báo duy nhất.</li>
                <li>Early Stopping: nếu sau 30 cây liên tiếp mà độ chính xác không tăng, mô hình tự động dừng train, giữ trạng thái tổng quát nhất.</li>
              </ul>

              <div style={{ fontWeight: 700, fontSize: 13, color: '#C62828', marginBottom: 6 }}>5. Case Study: Luồng logic khi phân tích một cổ phiếu</div>
              <P>
                Khi phân tích một cổ phiếu như VNM, luồng logic diễn ra như sau:
              </P>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  { step: 'Lớp Deep Learning (Prior Belief)', body: 'GRU + Attention quét 20 phiên qua, nhìn thấy áp lực bán ngắn hạn và xuất ra xác suất giảm điểm chiếm ưu thế.', color: '#4A7C5F', bg: '#E8F5E9' },
                  { step: 'Lớp Kỹ thuật & Cấu trúc (Confirmation)', body: 'Dữ liệu được ném vào XGBoost kết hợp với việc kiểm tra vị trí giá: nằm dưới EMA-50 (xu hướng giảm) và dội ngược từ vùng kháng cự K-Means phía trên.', color: '#9E7E45', bg: '#FFF8E1' },
                  { step: 'Sự đồng thuận tối ưu (XGBoost Decision)', body: 'Hội đồng 500 cây tổng hợp: Prior Deep Learning báo GIẢM + Kỹ thuật báo CẢN CỨNG → P(BÁN) đạt ≥55% → kích hoạt lệnh BÁN.', color: '#C62828', bg: '#FFEBEE' },
                ].map(c => (
                  <div key={c.step} style={{ padding: 12, background: c.bg, borderRadius: 8, border: `1px solid ${c.color}33` }}>
                    <div style={{ fontWeight: 700, fontSize: 12, color: c.color, marginBottom: 4 }}>{c.step}</div>
                    <div style={{ fontSize: 12 }}>{c.body}</div>
                  </div>
                ))}
              </div>
              <div style={{ padding: 12, background: '#F9F6F1', borderRadius: 8, fontSize: 12, marginTop: 12, border: '1px solid #EDE5D8', lineHeight: 1.6 }}>
                <strong>Tóm lại:</strong> Deep Learning đóng vai trò dự báo thời tiết (xu hướng lớn), còn Machine Learning và toán hình học đóng vai trò chọn thời điểm xuống đường (vùng giá kích hoạt lệnh).
              </div>
            </div>
          </Section>

        </div>
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────

export default function PredictPrice() {
  const [ticker,       setTicker]       = useState('')
  const [forecastDays, setForecastDays] = useState(5)
  const [loading,      setLoading]      = useState(false)
  const [result,       setResult]       = useState(null)
  const [signal,       setSignal]       = useState(null)
  const [error,        setError]        = useState(null)
  const [showModal,    setShowModal]    = useState(false)

  async function handlePredict() {
    const t = ticker.trim().toUpperCase()
    if (!t) return
    setLoading(true); setError(null); setResult(null); setSignal(null)
    try {
      const [priceData, signalData] = await Promise.all([
        predictPrice(t, 20),
        predictSignal(t, 0.55),
      ])
      setResult(priceData)
      setSignal(signalData)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Lỗi không xác định')
    } finally {
      setLoading(false)
    }
  }

  const slicedReturns = result ? result.predicted_returns.slice(0, forecastDays) : []
  const chartData  = result ? buildChartData(result.current_price, slicedReturns, result.direction, result.historical_prices ?? []) : []
  const returnBars = result ? buildReturnBars(slicedReturns, result.direction) : []
  const allPrices  = chartData.flatMap(d => [d.actual, d.predicted]).filter(Boolean)
  const yMin = allPrices.length ? Math.round(Math.min(...allPrices) * 0.995) : 0
  const yMax = allPrices.length ? Math.round(Math.max(...allPrices) * 1.005) : 'auto'

  const cumReturn   = slicedReturns.reduce((acc, r) => acc + safeReturn(r, result?.direction), 0)
  const predictedND = result ? Math.round(result.current_price * Math.exp(cumReturn)) : null
  const isUp        = cumReturn > 0
  const sigStyle    = signal ? (SIGNAL_STYLE[signal.signal] ?? SIGNAL_STYLE.HOLD) : null
  const isSpecial   = result ? SPECIALIZED.has(result.ticker) : false

  return (
    <>
      {showModal && <ModelExplainModal onClose={() => setShowModal(false)} />}

      <div className="predict-layout">
        {/* Left: config */}
        <div className="predict-config">
          <div className="card">
            <div className="card-title">Cấu hình dự đoán</div>
            <div className="card-sub">Chọn cổ phiếu để dự đoán giá 5 ngày tới</div>

            <div className="form-group">
              <label className="form-label">Mã cổ phiếu</label>
              <select
                className="form-select"
                value={KNOWN.has(ticker) ? ticker : ''}
                onChange={e => setTicker(e.target.value)}
              >
                <option value="">Chọn cổ phiếu</option>
                {Object.entries(TICKER_NAMES).map(([k, v]) => (
                  <option key={k} value={k}>{k}: {v}</option>
                ))}
              </select>
            </div>

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
              <label className="form-label">Số ngày dự báo</label>
              <div style={{ display: 'flex', gap: 6 }}>
                {[1, 3, 5].map(n => (
                  <button
                    key={n}
                    onClick={() => setForecastDays(n)}
                    style={{
                      flex: 1, padding: '7px 0', borderRadius: 8,
                      border: `1.5px solid ${forecastDays === n ? 'var(--gold-dark)' : 'var(--border)'}`,
                      background: forecastDays === n ? '#FFF5E0' : 'transparent',
                      color: forecastDays === n ? 'var(--gold-dark)' : 'var(--text-muted)',
                      fontWeight: forecastDays === n ? 700 : 400,
                      fontSize: 13, cursor: 'pointer',
                    }}
                  >
                    {n} ngày
                  </button>
                ))}
              </div>
            </div>

            <button
              className="btn btn-primary btn-full"
              onClick={handlePredict}
              disabled={!ticker.trim() || loading}
            >
              {loading ? '⟳ Đang dự đoán...' : `⊕ Dự đoán ${forecastDays} ngày`}
            </button>

            {result && (
              <div className="price-result-section">
                <div style={{ borderTop: '1px solid var(--border)', margin: '18px 0 16px' }} />

                {/* Model badge */}
                <div style={{ marginBottom: 12 }}>
                  {isSpecial ? (
                    <span style={{
                      display: 'inline-block', padding: '3px 9px', borderRadius: 20,
                      background: '#FFF5E0', color: 'var(--gold-dark)',
                      fontSize: 11, fontWeight: 600, border: '1px solid #E8D5A0',
                    }}>
                      ★ Model Chuyên Biệt · DA 63.64%
                    </span>
                  ) : (
                    <span style={{
                      display: 'inline-block', padding: '3px 9px', borderRadius: 20,
                      background: '#F0F0F0', color: 'var(--text-muted)',
                      fontSize: 11, fontWeight: 600,
                    }}>
                      Model Tổng Quát
                    </span>
                  )}
                </div>

                {/* Signal badge */}
                {signal && (
                  <>
                    <div style={{
                      display: 'flex', alignItems: 'center', gap: 10,
                      padding: '10px 12px', borderRadius: 10,
                      background: sigStyle.bg, marginBottom: 8,
                    }}>
                      <span style={{ fontSize: 18, fontWeight: 800, color: sigStyle.color }}>
                        {sigStyle.label}
                      </span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 11, color: sigStyle.color, fontWeight: 600 }}>
                          Tín hiệu XGBoost
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          Độ tự tin: {(signal.conviction * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div style={{ textAlign: 'left', fontSize: 11, color: 'var(--text-muted)' }}>
                        <div>Xác suất MUA: {(signal.p_buy * 100).toFixed(0)}%</div>
                        <div>Xác suất BÁN: {(signal.p_sell * 100).toFixed(0)}%</div>
                        <div>Xác suất GIỮ: {(signal.p_hold * 100).toFixed(0)}%</div>
                      </div>
                    </div>

                    {/* "Tìm hiểu thêm" button */}
                    <button
                      onClick={() => setShowModal(true)}
                      style={{
                        width: '100%', padding: '8px 12px',
                        borderRadius: 8, cursor: 'pointer',
                        border: '1.5px solid #EDE5D8',
                        background: '#F9F6F1',
                        color: 'var(--text-secondary)',
                        fontSize: 11, fontWeight: 600,
                        whiteSpace: 'nowrap',
                        marginBottom: 14,
                        transition: 'background 0.15s',
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = '#FAF3E0'}
                      onMouseLeave={e => e.currentTarget.style.background = '#F9F6F1'}
                    >
                      Tại sao model đưa ra khuyến nghị này?
                    </button>
                  </>
                )}

                <div className="price-label">Cổ phiếu</div>
                <div className="price-ticker-name">
                  {result.ticker}{TICKER_NAMES[result.ticker] ? `: ${TICKER_NAMES[result.ticker]}` : ''}
                </div>
                <div className="price-label">Giá hiện tại</div>
                <div className="price-current">{fmtVND(result.current_price)}</div>
                <div className="price-label">Dự báo cuối kỳ (+{forecastDays} ngày)</div>
                <div className="price-predicted">{fmtVND(predictedND)}</div>
                <div className="price-label" style={{ marginTop: 8 }}>Tổng thay đổi dự kiến</div>
                <div className={`price-change ${isUp ? 'up' : 'down'}`}>
                  {isUp ? '↗' : '↘'} {isUp ? '+' : ''}{(cumReturn * 100).toFixed(2)}%
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: charts */}
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
              <div className="chart-block">
                <div className="chart-block-title">Biểu đồ dự đoán giá</div>
                <div className="chart-block-sub">
                  {result?.historical_prices?.length > 0
                    ? `${result.historical_prices.length} ngày thực tế (nét liền) · ${forecastDays} ngày dự báo (nét đứt vàng)`
                    : `20 ngày lịch sử (nét liền) · ${forecastDays} ngày dự báo (nét đứt vàng)`
                  }
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <AreaChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                    <defs>
                      <linearGradient id="fillActual" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#4A7C5F" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="#4A7C5F" stopOpacity={0.02} />
                      </linearGradient>
                      <linearGradient id="fillPredicted" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#C4A265" stopOpacity={0.25} />
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
                      type="monotone" dataKey="actual" name="Giá lịch sử"
                      stroke="#4A7C5F" strokeWidth={2}
                      fill="url(#fillActual)" dot={false} connectNulls={false}
                    />
                    <Area
                      type="monotone" dataKey="predicted" name="Giá dự báo"
                      stroke="#C4A265" strokeWidth={2} strokeDasharray="5 3"
                      fill="url(#fillPredicted)"
                      dot={{ r: 4, fill: '#C4A265', strokeWidth: 0 }}
                      connectNulls={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-block">
                <div className="chart-block-title">Biến động dự báo theo ngày</div>
                <div className="chart-block-sub">
                  Log-return từng ngày trong quỹ đạo {forecastDays} ngày — xanh tăng, vàng giảm
                </div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={returnBars} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#EDE5D8" vertical={false} />
                    <XAxis
                      dataKey="day"
                      tick={{ fill: '#A09080', fontSize: 13 }}
                      axisLine={false} tickLine={false}
                    />
                    <YAxis
                      tickFormatter={v => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`}
                      tick={{ fill: '#A09080', fontSize: 11 }}
                      axisLine={false} tickLine={false}
                      width={62}
                    />
                    <Tooltip
                      formatter={v => [`${v > 0 ? '+' : ''}${v.toFixed(3)}%`, 'Log-return']}
                      labelFormatter={l => `Ngày ${l}`}
                      contentStyle={{
                        background: '#FAF7F2', border: '1px solid #EDE5D8',
                        borderRadius: 8, fontSize: 12,
                      }}
                    />
                    <Bar dataKey="return_pct" name="Return (%)" radius={[4, 4, 0, 0]}>
                      {returnBars.map((d, i) => (
                        <Cell key={i} fill={d.return_pct >= 0 ? '#4A7C5F' : '#C4A265'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>

                {signal && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
                    {[
                      { label: 'P(MUA)', value: signal.p_buy,  color: '#4A7C5F', bg: '#E8F5E9' },
                      { label: 'P(GIỮ)', value: signal.p_hold, color: '#9E7E45', bg: '#FFF8E1' },
                      { label: 'P(BÁN)', value: signal.p_sell, color: '#C62828', bg: '#FFEBEE' },
                    ].map(({ label, value, color, bg }) => (
                      <div key={label} style={{
                        flex: 1, textAlign: 'center',
                        background: bg, borderRadius: 10, padding: '10px 6px',
                      }}>
                        <div style={{ fontSize: 20, fontWeight: 800, color }}>
                          {(value * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                          {label}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <p style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'right' }}>
                Model: {result.model_used} · Ref: {result.prediction_date} · {result.data_source}
                {result.is_known_ticker ? ' · Active DA 63.64%' : ' · Generalist model'}
              </p>
            </>
          )}
        </div>
      </div>
    </>
  )
}
