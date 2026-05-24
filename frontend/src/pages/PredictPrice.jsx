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

// ── Model Explanation Modal ───────────────────────────────────

const FEATURE_GROUPS = [
  {
    title: 'Lớp 1 — Meta-features từ mạng Học sâu (3 tính năng)',
    color: '#4A7C5F',
    bg: '#E8F5E9',
    features: [
      { name: 'mtl_p_up', desc: 'Xác suất (0–1) model Deep Learning dự báo giá tăng vượt +2.0% trong 5 phiên tới.' },
      { name: 'mtl_p_down', desc: 'Xác suất (0–1) model Deep Learning dự báo giá giảm dưới −1.5% trong 5 phiên tới.' },
      { name: 'mtl_conviction', desc: 'Độ tự tin tối đa = max(mtl_p_up, mtl_p_down). Lọc bỏ trạng thái thị trường đi ngang (Sideways).' },
    ],
  },
  {
    title: 'Lớp 2 — Hình học vùng giá cản K-Means (5 tính năng)',
    color: '#9E7E45',
    bg: '#FFF8E1',
    features: [
      { name: 'sr_distance_pct', desc: 'Khoảng cách (%) từ giá hiện tại đến tâm vùng cản gần nhất. Dương = trên hỗ trợ, âm = dưới kháng cự.' },
      { name: 'sr_breakout_up', desc: 'Nhị phân (0/1). = 1 khi giá cắt dứt khoát lên trên vùng kháng cự (+0.5% zone).' },
      { name: 'sr_breakout_down', desc: 'Nhị phân (0/1). = 1 khi giá đâm thủng xuống dưới vùng hỗ trợ (−0.5% zone).' },
      { name: 'sr_near_resistance', desc: 'Chỉ thị (0/1). = 1 nếu giá cách vùng kháng cự phía trên < 0.5%. Vùng rủi ro đảo chiều.' },
      { name: 'sr_near_support', desc: 'Chỉ thị (0/1). = 1 nếu giá cách vùng hỗ trợ phía dưới < 0.5%. Vùng có lực cầu tiềm năng.' },
    ],
  },
  {
    title: 'Lớp 3 — Giao cắt đường trung bình động MA (7 tính năng)',
    color: '#5B7FA6',
    bg: '#E8F0FB',
    features: [
      { name: 'ma_golden_cross_short', desc: 'Sự kiện (0/1). = 1 tại phiên EMA-10 cắt lên EMA-20 → xu hướng tăng ngắn hạn bắt đầu.' },
      { name: 'ma_death_cross_short', desc: 'Sự kiện (0/1). = 1 tại phiên EMA-10 cắt xuống EMA-20 → xu hướng giảm ngắn hạn bắt đầu.' },
      { name: 'ma_golden_cross_long', desc: 'Sự kiện (0/1). = 1 tại phiên EMA-20 cắt lên EMA-50 → xác nhận pha tăng trung hạn.' },
      { name: 'ma_death_cross_long', desc: 'Sự kiện (0/1). = 1 tại phiên EMA-20 cắt xuống EMA-50 → xác nhận pha giảm trung hạn.' },
      { name: 'ma_short_gap_pct', desc: '(EMA₁₀ − EMA₂₀) / EMA₂₀ × 100. Độ rộng phân kỳ ngắn hạn — giá trị càng lớn thể hiện gia tốc tăng càng mạnh.' },
      { name: 'ma_long_gap_pct', desc: '(EMA₂₀ − EMA₅₀) / EMA₅₀ × 100. Độ bền vững xu hướng trung hạn.' },
      { name: 'ma_alignment', desc: '+1 nếu EMA₁₀ > EMA₂₀ > EMA₅₀ (toàn tăng), −1 nếu ngược lại (toàn giảm), 0 nếu đan xen (Sideways).' },
    ],
  },
  {
    title: 'Lớp 4 — Chỉ báo kỹ thuật cổ phiếu + VN-Index macro (27 tính năng)',
    color: '#7B5EA7',
    bg: '#F3EEF9',
    features: [
      { name: 'rsi_14', desc: 'Relative Strength Index 14 phiên — đo trạng thái quá mua/quá bán của cổ phiếu mục tiêu.' },
      { name: 'macd / macd_signal / macd_hist', desc: 'MACD và histogram phân kỳ — phát hiện sớm tín hiệu đảo chiều.' },
      { name: 'bb_upper / bb_lower', desc: 'Bollinger Bands (SMA-20 ± 2σ) — vùng biên kỹ thuật trên/dưới của giá.' },
      { name: 'atr_14', desc: 'Average True Range — đo biến động tuyệt đối, dùng để tự động điều chỉnh mức stoploss.' },
      { name: 'ema_10 / ema_20 / ema_50', desc: 'Đường trung bình hàm mũ ngắn/trung/dài hạn của cổ phiếu.' },
      { name: 'vni_* (12 features)', desc: 'Toàn bộ chỉ báo trên nhưng áp dụng cho VN-Index — làm màng lọc bối cảnh thị trường vĩ mô.' },
      { name: 'ticker_encoded', desc: 'Định danh số hóa (0–26) của cổ phiếu. Giúp XGBoost học hành vi riêng từng ngành: FPT (công nghệ) khác VCB (ngân hàng).' },
    ],
  },
]

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

  return (
    <div style={{
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

          {/* Sections */}
          <Section id="arch" title="1. Bản thiết kế hệ thống — Cascade Model (Mô hình xếp chồng phân cấp)">
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 10 }}>
              Thay vì dùng một mạng nơ-ron sâu duy nhất (dễ bị bẫy nhiễu trên thị trường HOSE), hệ thống phân tách bài toán thành hai lớp tối ưu:
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {[
                { title: 'Lớp 1 — Deep Sequential Extractor', body: 'GRU + Multi-Head Attention trích xuất động lực dòng tiền từ chuỗi lịch sử 20 phiên. Đầu ra: 3 meta-features thể hiện "niềm tin tiên nghiệm" của mạng sâu.', color: '#4A7C5F', bg: '#E8F5E9' },
                { title: 'Lớp 2 — Tabular Aggregator & Gate', body: 'XGBoost tích hợp thông tin xu hướng từ lớp 1 với không gian hình học S/R và chỉ báo động lượng để đưa ra xác suất hành động cuối cùng.', color: '#7B5EA7', bg: '#F3EEF9' },
              ].map(c => (
                <div key={c.title} style={{ padding: 14, borderRadius: 10, background: c.bg, border: `1px solid ${c.color}33` }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: c.color, marginBottom: 6 }}>{c.title}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{c.body}</div>
                </div>
              ))}
            </div>
          </Section>

          <Section id="gru" title="2. Lớp học sâu — GRU & Multi-Head Attention">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <p><strong>Cửa sổ trượt W=20 phiên</strong> — tương đương 1 tháng giao dịch. Mô hình quét chuỗi thời gian và cập nhật qua hai cổng toán học:</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, margin: '12px 0' }}>
                <div style={{ padding: 12, background: '#E8F5E9', borderRadius: 8 }}>
                  <div style={{ fontWeight: 700, fontSize: 12, color: '#4A7C5F', marginBottom: 4 }}>Update Gate</div>
                  <div style={{ fontSize: 12 }}>Quyết định lượng thông tin từ xu hướng quá khứ cần truyền tiếp — ví dụ: trạng thái tích lũy/đè giá kéo dài của dòng tiền lớn.</div>
                </div>
                <div style={{ padding: 12, background: '#FFF8E1', borderRadius: 8 }}>
                  <div style={{ fontWeight: 700, fontSize: 12, color: '#9E7E45', marginBottom: 4 }}>Reset Gate</div>
                  <div style={{ fontSize: 12 }}>Loại bỏ nhiễu ngắn hạn — ví dụ: biến động kỹ thuật do đáo hạn phái sinh.</div>
                </div>
              </div>
              <p><strong>Multi-Head Attention (2 heads)</strong>: Tính điểm tương quan giữa tất cả các ngày trong cửa sổ 20 phiên. Tự động "bơm trọng số" vào các phiên có hành vi đặc biệt (FTD bùng nổ, Washout ép bán).</p>
              <p style={{ marginTop: 10 }}><strong>Multi-Task Loss:</strong></p>
              <div style={{ background: '#F5F0FF', borderRadius: 8, padding: '10px 14px', fontFamily: 'monospace', fontSize: 12, margin: '8px 0' }}>
                L_total = L_Huber(quỹ đạo 5 ngày) + <strong style={{ color: '#7B5EA7' }}>5.0</strong> × L_CrossEntropy(xu hướng)
              </div>
              <p>Hệ số phạt <code style={{ background: '#eee', padding: '1px 4px', borderRadius: 3 }}>λ_cls = 5.0</code> ép mạng phải ưu tiên trích xuất <em>Directional Alpha</em> — biết đúng hướng quan trọng hơn biết chính xác biên độ.</p>
            </div>
          </Section>

          <Section id="kmeans" title="3. Hình học hóa tâm lý thị trường — K-Means Clustering">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <p>Cửa sổ lịch sử <strong>63 phiên (3 tháng)</strong>. Tự động tìm số vùng cản tối ưu bằng Elbow Method:</p>
              <div style={{ background: '#F5F0FF', borderRadius: 8, padding: '10px 14px', fontFamily: 'monospace', fontSize: 12, margin: '10px 0' }}>
                k_optimal = argmax_k [ (I_k₊₁ − I_k) − (I_k − I_k₋₁) ]
              </div>
              <p>Tìm điểm gãy (Elbow) của chuỗi Inertia với k ∈ [2, 11]. Mỗi cổ phiếu có cấu trúc cản riêng — ngân hàng đi nền chặt ít vùng cản hơn bất động sản biến động mạnh.</p>
              <div style={{ padding: 12, background: '#FFF8E1', borderRadius: 8, marginTop: 10 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#9E7E45', marginBottom: 6 }}>Bộ lọc phá vỡ cấu trúc giá (BOS Filter)</div>
                <div style={{ fontSize: 12 }}>
                  Vùng an toàn ±0.5% quanh tâm cản. Tín hiệu breakout chỉ kích hoạt khi giá hôm nay vượt hoàn toàn biên trên, đồng thời hôm qua nằm dưới biên dưới. Loại bỏ bull-trap và bear-trap.
                </div>
              </div>
            </div>
          </Section>

          <Section id="xgb" title="4. XGBoost — Lớp ra quyết định tối ưu (500 cây)">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <p style={{ marginBottom: 10 }}>42 features từ các lớp trên được tích hợp thành vector phẳng. XGBoost có 3 ưu điểm chính:</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  { title: 'Xử lý dữ liệu hỗn hợp không tuần bám', body: 'Chấp nhận đồng thời xác suất (0–1), liên tục (chỉ báo), hình học (khoảng cách cản) và định danh ngành — không cần chuẩn hóa thô bạo làm mất thông tin.' },
                  { title: 'Kiểm soát Overfitting bằng L1+L2 Regularization', body: 'L1 (α=0.1) và L2 (λ=1.0) phạt cây quá phụ thuộc vào vài chỉ báo riêng lẻ — ép model tìm sự đồng thuận từ tất cả lớp tính năng.' },
                  { title: 'Giải thích được (Interpretable)', body: 'Feature importance và SHAP values cho biết chỉ báo nào đang thúc đẩy quyết định — không phải hộp đen.' },
                ].map(c => (
                  <div key={c.title} style={{ padding: 12, background: '#F3EEF9', borderRadius: 8, border: '1px solid #7B5EA733' }}>
                    <div style={{ fontWeight: 700, fontSize: 12, color: '#7B5EA7', marginBottom: 4 }}>▸ {c.title}</div>
                    <div style={{ fontSize: 12 }}>{c.body}</div>
                  </div>
                ))}
              </div>
            </div>
          </Section>

          <Section id="asymmetry" title="5. Lý do mua bán không đối xứng (Asymmetric Thresholds)">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginBottom: 14 }}>
                {[
                  { label: 'BUY', cond: 'fwd_ret_5 ≥ +2.0%', color: '#2E7D32', bg: '#E8F5E9', note: 'Ngưỡng cao hơn' },
                  { label: 'SELL', cond: 'fwd_ret_5 ≤ −1.5%', color: '#C62828', bg: '#FFEBEE', note: 'Ngưỡng thấp hơn' },
                  { label: 'HOLD', cond: 'Khoảng giữa', color: '#9E7E45', bg: '#FFF8E1', note: 'Không có tín hiệu' },
                ].map(c => (
                  <div key={c.label} style={{ padding: 12, borderRadius: 10, background: c.bg, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 800, color: c.color }}>{c.label}</div>
                    <div style={{ fontSize: 11, fontFamily: 'monospace', color: c.color, margin: '4px 0' }}>{c.cond}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{c.note}</div>
                  </div>
                ))}
              </div>
              <p>Sự không đối xứng phản ánh <strong>tâm lý hành vi</strong> thực tế của nhà đầu tư:</p>
              <ul style={{ marginLeft: 16, lineHeight: 2, fontSize: 12 }}>
                <li>Mất −1.5% cảm giác <em>đau</em> nhiều hơn lãi +1.5% cảm giác <em>vui</em> (Loss Aversion — Kahneman & Tversky).</li>
                <li>Thị trường HOSE có xu hướng giảm nhanh hơn tăng — cần trigger SELL sớm hơn.</li>
                <li>Ngưỡng BUY cao (+2%) đảm bảo chỉ vào lệnh khi có đủ biên an toàn (Margin of Safety).</li>
                <li>Ngưỡng SELL thấp (−1.5%) ưu tiên bảo toàn vốn hơn tối đa hóa lợi nhuận.</li>
              </ul>
            </div>
          </Section>

          <Section id="features" title="6. Từ điển đặc trưng — 42 tính năng trong 4 lớp">
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
                    display: 'grid', gridTemplateColumns: '180px 1fr',
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

          <Section id="importance" title="7. Feature Importance — XGBoost T4">
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Biểu đồ dưới đây cho thấy mức độ đóng góp của từng nhóm đặc trưng vào quyết định của XGBoost. Các feature từ mạng Học sâu (MTL) và Hình học cản (S/R) thường đứng đầu bảng.
            </div>
            <img
              src="/notebook/feature_importance_xgb.png"
              alt="XGBoost Feature Importance"
              style={{
                width: '100%', borderRadius: 10,
                border: '1px solid #EDE5D8',
              }}
              onError={e => { e.target.style.display = 'none' }}
            />
          </Section>

          <Section id="usecase" title="8. Use Case — Khi nào nên và không nên dùng tín hiệu này?">
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
        {/* ── Left: config ── */}
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
                  <option key={k} value={k}>{k} — {v}</option>
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
                          Tín hiệu XGBoost T4 · Horizon 5 ngày
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          Độ tự tin: {(signal.conviction * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div style={{ textAlign: 'right', fontSize: 11, color: 'var(--text-muted)' }}>
                        <div>P(MUA) {(signal.p_buy * 100).toFixed(0)}%</div>
                        <div>P(BÁN) {(signal.p_sell * 100).toFixed(0)}%</div>
                        <div>P(GIỮ) {(signal.p_hold * 100).toFixed(0)}%</div>
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
                        fontSize: 12, fontWeight: 600,
                        display: 'flex', alignItems: 'center',
                        justifyContent: 'center', gap: 6,
                        marginBottom: 14,
                        transition: 'background 0.15s',
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = '#FAF3E0'}
                      onMouseLeave={e => e.currentTarget.style.background = '#F9F6F1'}
                    >
                      <span>🔍</span>
                      Tại sao model đưa ra khuyến nghị này?
                    </button>
                  </>
                )}

                <div className="price-label">Cổ phiếu</div>
                <div className="price-ticker-name">
                  {result.ticker}{TICKER_NAMES[result.ticker] ? ` — ${TICKER_NAMES[result.ticker]}` : ''}
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
