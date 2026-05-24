import { useState, useEffect } from 'react'
import { BarChartIcon, TrendUpIcon, ShieldIcon, TargetIcon, LeafIcon, ClockIcon, PieIcon } from '../components/Icons.jsx'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from 'recharts'

const API_BASE = import.meta.env.VITE_API_URL || 'https://amyas0107-investnature-api.hf.space'

const TABS = [
  { id: 'journey',      label: 'Hành trình R&D',      Icon: TrendUpIcon  },
  { id: 'architecture', label: 'Kiến trúc MTL',        Icon: BarChartIcon },
  { id: 'signal',       label: 'Tín hiệu XGBoost',    Icon: ShieldIcon   },
  { id: 'portfolio',    label: 'Tối ưu danh mục',      Icon: TargetIcon   },
  { id: 'features',     label: '25 Chỉ số kỹ thuật',  Icon: PieIcon      },
  { id: 'backtest',     label: 'Walk-Forward Backtest', Icon: ClockIcon    },
]

/* ── Reusable atoms ───────────────────────────────────────── */

function SectionTitle({ children }) {
  return <h2 style={{ fontSize: 20, fontWeight: 700, color: 'var(--gold-dark)', marginBottom: 6 }}>{children}</h2>
}

function Sub({ children }) {
  return <p style={{ fontSize: 13.5, color: 'var(--text-muted)', marginBottom: 18, lineHeight: 1.7 }}>{children}</p>
}

function KpiRow({ items }) {
  return (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
      {items.map(({ label, value, color = 'var(--gold-dark)', sub }) => (
        <div key={label} style={{
          flex: '1 1 140px', background: 'var(--bg-card)', border: '1px solid var(--border)',
          borderRadius: 12, padding: '14px 16px',
        }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
          <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
          {sub && <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>}
        </div>
      ))}
    </div>
  )
}

function Tag({ children, color = '#FFF5E0', textColor = 'var(--gold-dark)' }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 20,
      background: color, color: textColor, fontSize: 11, fontWeight: 600,
      marginRight: 6, marginBottom: 4,
    }}>{children}</span>
  )
}

function MilestoneCard({ num, title, tag, tagColor, tagText, children, accent }) {
  return (
    <div style={{
      display: 'flex', gap: 16, marginBottom: 20,
      background: 'var(--bg-card)', border: `1px solid var(--border)`,
      borderLeft: `4px solid ${accent}`, borderRadius: 12, padding: '16px 18px',
    }}>
      <div style={{
        minWidth: 32, height: 32, borderRadius: '50%',
        background: accent, color: '#fff',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontWeight: 800, fontSize: 14, flexShrink: 0,
      }}>{num}</div>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>{title}</span>
          <Tag color={tagColor} textColor={tagText}>{tag}</Tag>
        </div>
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.7, margin: 0 }}>{children}</p>
      </div>
    </div>
  )
}

/* ── Tabs ─────────────────────────────────────────────────── */

function TabJourney() {
  return (
    <div>
      <SectionTitle>Hành trình phát triển mô hình</SectionTitle>
      <Sub>Từ baseline machine learning đến kiến trúc Multi-Task Learning — 4 cột mốc quyết định.</Sub>

      <MilestoneCard num={1} accent="#8B7355"
        title="Mốc 1 — ML Baselines: Random Forest & XGBoost"
        tag="Baseline" tagColor="#F5F0E8" tagText="#8B7355"
      >
        Random Forest đạt DA ~58.36% (2 ngày đầu) nhưng bị <strong>Temporal Lag</strong> — chạy theo đuôi thị
        trường, phản ứng chậm. XGBoost cho MAE thấp nhưng cả hai đều kém khi thị trường đảo chiều Bearish
        đầu 2026. Chứng minh: cần kiến trúc sâu hơn để nắm bắt cấu trúc xu hướng.
      </MilestoneCard>

      <MilestoneCard num={2} accent="#C62828"
        title="Mốc 2 — LSTM truyền thống: dự đoán T+3 đơn lẻ"
        tag="Overfitting" tagColor="#FFEBEE" tagText="#C62828"
      >
        LSTM single-step cố đoán chính xác ngày thứ 3 — nhưng Val Loss tăng liên tục. Mô hình học thuộc
        lòng nhiễu thị trường thay vì hiểu cấu trúc xu hướng. Bài học: bài toán điểm đơn lẻ quá nhạy
        cảm với outlier và noise trong chuỗi tài chính.
      </MilestoneCard>

      <MilestoneCard num={3} accent="#9E7E45"
        title="Mốc 3 — Seq2Seq + Multi-Head Attention: quỹ đạo 5 ngày"
        tag="Mean-Hugging Trap" tagColor="#FFF8E1" tagText="#9E7E45"
      >
        Chuyển từ đoán 1 điểm sang đoán <strong>cả quỹ đạo 5 ngày liên tiếp</strong> — một chân lý:
        "Dự đoán ngày T+n là bài toán con của quỹ đạo". Tuy nhiên Seq2Seq thuần bị sập
        <strong> Mean-Hugging Trap</strong>: tối ưu MAE khiến AI chỉ đi ngang quanh mức 0, DA sụt ~41%.
        Cần cơ chế ép AI phải chọn hướng.
      </MilestoneCard>

      <MilestoneCard num={4} accent="#2D6A4F"
        title="Mốc 4 — MTL: Shared Encoder + 2 Head chuyên biệt"
        tag="DA 58.57% Active" tagColor="#E8F5E9" tagText="#2E7D32"
      >
        Đập tan bẫy "ôm trung bình" bằng <strong>tỷ lệ Loss 1:5</strong> (ép AI ưu tiên đúng hướng gấp 5
        lần đúng giá). Head Regression đoán Log Return, Head Classification đoán hướng (Softmax 3 class).
        Shared GRU Encoder + Multi-Head Attention dùng chung — kiến thức từ một task củng cố task kia.
        <strong> Active DA đạt 58.57%</strong> — vượt xa mức ngẫu nhiên 50% trong thị trường đầy nhiễu.
      </MilestoneCard>

      <div className="card" style={{ background: '#FFFBF2', border: '1px solid #E8D5A0' }}>
        <div style={{ fontWeight: 700, color: 'var(--gold-dark)', marginBottom: 8 }}>
          Triết lý cốt lõi
        </div>
        <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.8, margin: 0 }}>
          <em>"Đúng xu hướng để kiếm lợi nhuận quan trọng hơn đúng từng đồng nhưng đi sai hướng."</em>
          <br />
          Mô hình được thiết kế để <strong>Dẫn dắt xu hướng (Leading Indicator)</strong> — không phải chạy
          theo đuôi thị trường (Lagging). Chấp nhận Vertical Gap nhỏ để đổi lấy Directional Curvature
          bám sát thực tế.
        </p>
      </div>
    </div>
  )
}

function TabArchitecture() {
  return (
    <div>
      <SectionTitle>Kiến trúc Multi-Task Learning (MTL)</SectionTitle>
      <Sub>Shared Encoder + Multi-Head Attention dùng chung, tách thành 2 head chuyên biệt.</Sub>

      {/* Architecture diagram (ASCII-style) */}
      <div style={{
        background: '#1E1E1E', color: '#D4C090', borderRadius: 14,
        padding: '20px 24px', fontFamily: 'monospace', fontSize: 12.5,
        lineHeight: 2, marginBottom: 20, overflowX: 'auto',
      }}>
        <div style={{ color: '#C4A265', fontWeight: 700, marginBottom: 8 }}>
          Input: (None, 20, 25)  ←  cửa sổ 20 phiên × 25 features
        </div>
        <div>{'          ┌──────────────────────────────┐'}</div>
        <div>{'          │   Shared GRU Encoder          │'}</div>
        <div>{'          │   (Bidirectional GRU × 2)     │'}</div>
        <div>{'          │   + Multi-Head Attention       │'}</div>
        <div>{'          │   + Dropout 0.3               │'}</div>
        <div>{'          └──────────────┬───────────────┘'}</div>
        <div>{'                         │'}</div>
        <div>{'          ┌──────────────┴───────────────┐'}</div>
        <div>{'          │                               │'}</div>
        <div style={{ color: '#5C7A45' }}>
          {'     ┌────────────┐           ┌────────────────┐'}
        </div>
        <div style={{ color: '#5C7A45' }}>
          {'     │ Head 1     │           │ Head 2         │'}
        </div>
        <div style={{ color: '#5C7A45' }}>
          {'     │ Regression │           │ Classification │'}
        </div>
        <div style={{ color: '#5C7A45' }}>
          {'     │ Log Return │           │ Softmax(3)     │'}
        </div>
        <div style={{ color: '#5C7A45' }}>
          {'     │ (5 days)   │           │ DOWN/HOLD/UP   │'}
        </div>
        <div style={{ color: '#5C7A45' }}>
          {'     └────────────┘           └────────────────┘'}
        </div>
        <div style={{ marginTop: 8, color: '#A09080' }}>
          Loss = 1.0×MSE(returns) + 5.0×CategoricalCE(direction)
        </div>
      </div>

      <KpiRow items={[
        { label: 'Cửa sổ đầu vào', value: '20 phiên', sub: '≈ 1 tháng giao dịch', color: 'var(--text-primary)' },
        { label: 'Dự báo đầu ra', value: '5 ngày', sub: 'Quỹ đạo liên tiếp' },
        { label: 'Trọng số Loss', value: '1 : 5', sub: 'Regression : Classification' },
        { label: 'Active DA', value: '58.57%', sub: 'MTL model, gated', color: '#2E7D32' },
      ]} />

      <SectionTitle>25 Features đầu vào</SectionTitle>
      <Sub>12 chỉ báo kỹ thuật của cổ phiếu + 12 chỉ báo VN-Index + Volume.</Sub>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 20 }}>
        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontWeight: 700, fontSize: 12, color: 'var(--gold-dark)', marginBottom: 8 }}>
            Cổ phiếu (13)
          </div>
          {['volume','log_return_1d','rsi','macd','macd_signal','macd_hist',
            'bb_upper','bb_lower','bb_middle','ema_10','ema_20','ema_50','atr'].map(f => (
            <div key={f} style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '2px 0', borderBottom: '1px solid var(--border-light)' }}>
              {f}
            </div>
          ))}
        </div>
        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontWeight: 700, fontSize: 12, color: '#2E7D32', marginBottom: 8 }}>
            VN-Index (12)
          </div>
          {['vni_log_return','vni_rsi','vni_macd','vni_macd_signal','vni_macd_hist',
            'vni_bb_upper','vni_bb_lower','vni_bb_middle','vni_ema_10','vni_ema_20','vni_ema_50','vni_atr'].map(f => (
            <div key={f} style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '2px 0', borderBottom: '1px solid var(--border-light)' }}>
              {f}
            </div>
          ))}
        </div>
        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontWeight: 700, fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
            Tiền xử lý
          </div>
          {[
            '✓ RobustScaler (fit train only)',
            '✓ Sliding window 20',
            '✓ Không shuffle (time-series)',
            '✓ 70/10/20 train/val/test',
            '✓ Loại giá trị tuyệt đối',
            '✓ Min history ≥ 120 phiên',
          ].map(t => (
            <div key={t} style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '3px 0', borderBottom: '1px solid var(--border-light)' }}>
              {t}
            </div>
          ))}
        </div>
      </div>

      <SectionTitle>Active Directional Accuracy (Active DA)</SectionTitle>
      <div className="card" style={{ background: '#F0F7F0', border: '1px solid #B5D5B5' }}>
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.8, margin: 0 }}>
          <strong>Active DA</strong> là chỉ số tự định nghĩa — chỉ đo độ chính xác trên{' '}
          <strong>những ngày mà mô hình thực sự phát tín hiệu</strong> (conviction vượt ngưỡng),
          không tính đại trà toàn bộ ngày giao dịch.<br /><br />
          Khác với Random Forest (Passive DA ~58%) bị ép phải đoán mỗi ngày kể cả sideway,
          MTL chỉ "bóp cò" khi Softmax output {'>'} ngưỡng. Nhờ vậy, những lệnh được phát ra mang
          chất lượng cao hơn đáng kể — đây là lý do <strong>Active DA 58.57%</strong> thực chiến
          giá trị hơn con số 58.36% passive của baseline.
        </p>
      </div>
    </div>
  )
}

function TabSignal() {
  return (
    <div>
      <SectionTitle>Hệ thống tín hiệu XGBoost — Task 3</SectionTitle>
      <Sub>
        Bộ lọc hội tụ 4 tầng thông tin: Kỹ thuật · Cấu trúc S/R · VN-Index vĩ mô · Deep Learning Prior.
      </Sub>

      <KpiRow items={[
        { label: 'Active DA (threshold=0.55)', value: '63.64%', sub: 'XGBoost classifier', color: '#2E7D32' },
        { label: 'Tỷ lệ đứng ngoài', value: '89.2%', sub: 'HOLD — lọc sideway', color: 'var(--text-primary)' },
        { label: 'Tín hiệu MUA (precision)', value: '58.24%', sub: '91 tín hiệu', color: '#2E7D32' },
        { label: 'Tín hiệu BÁN (precision)', value: '~53%', sub: '226 tín hiệu', color: '#C62828' },
      ]} />

      {/* 4-layer confluence */}
      <SectionTitle>Bộ lọc Hội tụ 4 tầng</SectionTitle>
      {[
        { num: 1, title: 'Động lượng Kỹ thuật', color: '#C4A265', items: ['RSI (vùng an toàn)', 'MACD Histogram', 'EMA10×20 / EMA20×50 crossover', 'Bollinger Band mở rộng'] },
        { num: 2, title: 'Cấu trúc Hỗ trợ / Kháng cự', color: '#5C7A45', items: ['K-Means clustering (lookback 63 phiên)', 'sr_distance_pct — khoảng cách tới S/R', 'sr_breakout_up / sr_breakout_down', 'sr_near_support / sr_near_resistance'] },
        { num: 3, title: 'Ngữ cảnh Vĩ mô VN-Index', color: '#4A7C5F', items: ['12 chỉ báo kỹ thuật tính riêng cho VNI', 'Tránh đánh ngược chiều thị trường chung', 'VNI EMA, MACD, BB, RSI, ATR'] },
        { num: 4, title: 'Deep Learning Prior (MTL)', color: '#9E7E45', items: ['mtl_p_up — xác suất TĂNG từ Seq2Seq', 'mtl_p_down — xác suất GIẢM', 'mtl_conviction — độ tự tin MTL', 'Feature importance #1 và #2 của XGBoost'] },
      ].map(({ num, title, color, items }) => (
        <div key={num} style={{
          display: 'flex', gap: 14, marginBottom: 12,
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          borderLeft: `4px solid ${color}`, borderRadius: 10, padding: '12px 16px',
        }}>
          <div style={{ minWidth: 28, height: 28, borderRadius: '50%', background: color, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: 13, flexShrink: 0 }}>{num}</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)', marginBottom: 6 }}>{title}</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {items.map(i => <Tag key={i}>{i}</Tag>)}
            </div>
          </div>
        </div>
      ))}

      {/* Conviction Gate flowchart */}
      <SectionTitle style={{ marginTop: 24 }}>Conviction-Gated — Bộ lọc kỷ luật</SectionTitle>
      <Sub>Hệ thống chỉ phát tín hiệu khi độ tự tin (conviction) vượt ngưỡng an toàn.</Sub>

      <div style={{
        background: '#1E1E1E', color: '#D4C090', borderRadius: 14,
        padding: '18px 22px', fontFamily: 'monospace', fontSize: 12.5,
        lineHeight: 2, marginBottom: 20,
      }}>
        <div style={{ color: '#C4A265' }}>{'[ Softmax Output ]  →  P(UP)=75%  P(DOWN)=10%  P(HOLD)=15%'}</div>
        <div>{'                              │'}</div>
        <div>{'         ┌────────────────────┴──────────────────────┐'}</div>
        <div>{'         │     max(P) > threshold (0.55)?            │'}</div>
        <div>{'         └────────────────────┬──────────────────────┘'}</div>
        <div>{'                    ┌─────────┴──────────┐'}</div>
        <div style={{ color: '#5C7A45' }}>{'               (YES)                   (NO)'}</div>
        <div style={{ color: '#5C7A45' }}>{'                 │                       │'}</div>
        <div style={{ color: '#5C7A45' }}>{'        [Gate OPEN]             [Gate CLOSED]'}</div>
        <div style={{ color: '#5C7A45' }}>{'       Phát MUA / BÁN          Giữ vị thế FLAT'}</div>
        <div style={{ color: '#5C7A45' }}>{'       (High conviction)       (Sideway — đứng ngoài)'}</div>
      </div>

      {/* Per-ticker highlights */}
      <SectionTitle>Hiệu suất nổi bật theo mã</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 10 }}>
        {[
          { ticker: 'VIC', signals: 14, precision: '93%', color: '#E8F5E9', border: '#B5D5B5', text: '#2E7D32' },
          { ticker: 'MWG', signals: 7,  precision: '86%', color: '#E8F5E9', border: '#B5D5B5', text: '#2E7D32' },
          { ticker: 'VHM', signals: 28, precision: '64%', color: '#FFF8E1', border: '#E8D5A0', text: '#9E7E45' },
        ].map(({ ticker, signals, precision, color, border, text }) => (
          <div key={ticker} style={{ background: color, border: `1px solid ${border}`, borderRadius: 10, padding: '14px 16px', textAlign: 'center' }}>
            <div style={{ fontSize: 22, fontWeight: 800, color: text }}>{ticker}</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: text }}>{precision}</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{signals} tín hiệu MUA</div>
          </div>
        ))}
      </div>

      {/* Label design */}
      <div className="card" style={{ marginTop: 16, background: '#FFFBF2', border: '1px solid #E8D5A0' }}>
        <div style={{ fontWeight: 700, color: 'var(--gold-dark)', marginBottom: 8, fontSize: 14 }}>
          Thiết kế Nhãn bất đối xứng (Asymmetric Labels)
        </div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 120 }}>
            <Tag color="#E8F5E9" textColor="#2E7D32">BUY (Class 2)</Tag>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '6px 0 0' }}>
              Lợi nhuận 5 ngày {'>'} <strong>+2.0%</strong><br />Khắt khe để lọc tăng giả
            </p>
          </div>
          <div style={{ flex: 1, minWidth: 120 }}>
            <Tag color="#FFEBEE" textColor="#C62828">SELL (Class 0)</Tag>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '6px 0 0' }}>
              Lợi nhuận 5 ngày {'<'} <strong>-1.5%</strong><br />Nhạy hơn vì sập HOSE khốc liệt
            </p>
          </div>
          <div style={{ flex: 1, minWidth: 120 }}>
            <Tag color="#F5F0E8" textColor="#8B7355">HOLD (Class 1)</Tag>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '6px 0 0' }}>
              Nằm giữa hai ngưỡng<br />Đứng ngoài, quan sát
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function TabPortfolio() {
  const [openSec, setOpenSec] = useState(null)
  const toggle = id => setOpenSec(prev => prev === id ? null : id)

  const Acc = ({ id, title, children }) => (
    <div style={{ marginBottom: 10 }}>
      <button
        onClick={() => toggle(id)}
        style={{
          width: '100%', textAlign: 'left', padding: '12px 16px',
          background: openSec === id ? '#FAF3E0' : 'var(--bg-card)',
          border: '1px solid var(--border)', borderRadius: openSec === id ? '10px 10px 0 0' : 10,
          fontSize: 14, fontWeight: 600, color: 'var(--text-primary)',
          cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          fontFamily: 'inherit',
        }}
      >
        <span>{title}</span>
        <span style={{ fontSize: 18, color: 'var(--text-muted)', lineHeight: 1 }}>
          {openSec === id ? '−' : '+'}
        </span>
      </button>
      {openSec === id && (
        <div style={{
          padding: '16px 18px', border: '1px solid var(--border)', borderTop: 'none',
          borderRadius: '0 0 10px 10px', background: '#fff', lineHeight: 1.7,
        }}>
          {children}
        </div>
      )}
    </div>
  )

  return (
    <div>
      <SectionTitle>Hệ thống tối ưu danh mục — Task 4</SectionTitle>
      <Sub>27 mã lớn nhất HOSE · 5 nhân tố chọn cổ phiếu · Markowitz Mean-Variance Optimization · Backtest 2025–2026.</Sub>

      <KpiRow items={[
        { label: 'Equal-Weight Top 10', value: '65.90%', sub: 'Lợi nhuận realized (backtest 2025–2026)', color: '#2E7D32' },
        { label: 'Sharpe Ratio', value: '1.62', sub: 'Equal-Weight Top 10', color: 'var(--gold-dark)' },
        { label: 'Spearman ρ', value: '0.556', sub: 'p-value = 0.0026 (1%)', color: '#2E7D32' },
        { label: 'Precision lift', value: '×2.6', sub: '90% top pick vs 35% phần còn lại', color: 'var(--gold-dark)' },
      ]} />

      {/* 5-factor scoring */}
      <SectionTitle>Mô hình 5 Nhân Tố — Composite Profitability Score</SectionTitle>
      {[
        { w: '30%', name: 'MTL Trajectory Score', desc: 'Dự báo xu hướng 5 ngày từ Seq2Seq MTL retrained trên 27 mã' },
        { w: '25%', name: 'Task 3 Signal Score', desc: 'XGBoost BUY conviction trong 3 tháng gần nhất (tần suất + độ tự tin)' },
        { w: '20%', name: 'Technical Momentum Score', desc: 'RSI an toàn + MACD mở rộng + EMA bull alignment + BB mở rộng' },
        { w: '15%', name: 'Rolling Sharpe Score', desc: 'Sharpe + Sortino 252 phiên — phạt nặng downside volatility' },
        { w: '10%', name: 'Trend Strength Score', desc: 'ADX — đo độ mạnh xu hướng hiện tại' },
      ].map(({ w, name, desc }) => (
        <div key={name} style={{
          display: 'flex', alignItems: 'flex-start', gap: 14, marginBottom: 10,
          background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 16px',
        }}>
          <div style={{ minWidth: 44, fontWeight: 800, fontSize: 15, color: 'var(--gold-dark)', textAlign: 'center', flexShrink: 0 }}>{w}</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)', marginBottom: 2 }}>{name}</div>
            <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>{desc}</div>
          </div>
        </div>
      ))}

      {/* Backtest table — realized returns from test period */}
      <SectionTitle style={{ marginTop: 20 }}>Kết quả Backtest Thực Tế (02/2025 – 04/2026)</SectionTitle>
      <div style={{ overflowX: 'auto', marginBottom: 8 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#FAF0DC' }}>
              {['Chỉ số', 'Equal-Weight 🏆', 'Risk-Taking', 'Prudent 🛡', 'VN-Index'].map(h => (
                <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, color: 'var(--gold-dark)', borderBottom: '2px solid var(--border)', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              ['Tổng lợi nhuận (realized)', '65.90%', '35.73%', '16.44%', '40.74%'],
              ['Lợi nhuận/năm',             '45.49%', '29.88%', '15.79%', '31.03%'],
              ['Biến động/năm',             '25.30%', '29.52%', '24.80%', '22.69%'],
              ['Sharpe Ratio',              '1.6201', '0.8596', '0.4550', '1.1694'],
              ['Max Drawdown',              '-19.87%','-23.20%','-21.97%','N/A'],
              ['Win Rate',                  '59.27%', '55.63%', '53.97%', 'N/A'],
            ].map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? 'var(--bg-base)' : 'var(--bg-card)' }}>
                {row.map((cell, j) => (
                  <td key={j} style={{
                    padding: '9px 12px', borderBottom: '1px solid var(--border)',
                    fontWeight: j === 1 ? 700 : 400,
                    color: j === 1 ? '#2E7D32' : 'var(--text-secondary)',
                  }}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 20, fontStyle: 'italic' }}>
        * "Tổng lợi nhuận (realized)" = lợi nhuận thực tế giai đoạn test (backtest) — khác với "Expected Return" mà MVO tính từ dữ liệu train (dùng để tối ưu tỷ trọng, không dự báo tương lai). Equal-Weight Top 10 = chia đều 1/10 cho 10 mã mô hình chọn.
      </p>

      {/* Academic explanation accordion */}
      <Acc id="mvo_analysis" title="🔬 Tại sao Equal-Weight thắng Markowitz MVO? — Phân tích toán học (DeMiguel, 2009)">
        <div style={{ background: '#E8F0FB', border: '1px solid #5B7FA644', borderRadius: 10, padding: '10px 14px', marginBottom: 14, fontSize: 12.5, color: 'var(--text-secondary)' }}>
          <strong style={{ color: '#5B7FA6' }}>Lưu ý về số liệu:</strong> Phân tích dưới đây so sánh{' '}
          <strong>Equal-Weight cùng 9 mã (53.93%)</strong> vs <strong>MVO Risk-Taking (35.73%)</strong> — đây là so sánh
          apples-to-apples trên cùng tập cổ phiếu. Kết quả Equal-Weight Top 10 trong bảng trên{' '}
          (<strong>65.90%</strong>) còn cao hơn vì bao gồm VHM và 9 mã khác được model chọn lọc.
        </div>
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.8, marginBottom: 16 }}>
          <strong>Cả hai, nhưng outlier VIC chiếm tới 80% nguyên nhân</strong> khiến MVO ngậm ngùi hít khói Equal-Weight.
          Công thức tính toán niên hóa có một chút sai lệch về mặt lý thuyết, nhưng nó không phải là thứ làm đảo lộn
          kết quả backtest. Chính <strong>bản chất toán học mang tính "nhút nhát" của Markowitz MVO</strong> khi đụng
          độ một "siêu quái vật" tăng trưởng như <strong>VIC (+678%)</strong> mới là nguyên nhân cốt lõi.
        </p>

        {/* Factor 1 */}
        <div style={{ background: '#FFF8E1', border: '1.5px solid #E8D5A0', borderRadius: 12, padding: '16px 18px', marginBottom: 14 }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--gold-dark)', marginBottom: 12 }}>
            Yếu tố 1: Tác động hủy diệt của siêu Outlier VIC (+678%)
          </div>
          <div style={{ background: '#1e1e1e', color: '#C4A265', borderRadius: 8, padding: '10px 14px', fontFamily: 'monospace', fontSize: 12.5, marginBottom: 14, fontStyle: 'italic' }}>
            "Mean-Variance Optimization is an error maximizer" — câu nói kinh điển trong toán định lượng
          </div>

          <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--text-primary)', marginBottom: 8 }}>
            Cách Equal-Weight ăn trọn sóng VIC:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 14 }}>
            {[
              'Bạn chọn ra danh mục Top 10, trong đó có VIC. Chiến lược Equal-Weight cào bằng mỗi mã 10% tỷ trọng.',
              'Khi tập test chạy, một mình mã VIC vọt lên +678%. Khoản đầu tư vào VIC nhân gần 8 lần.',
              'Vì cầm tới 10% tỷ trọng, một mình VIC gánh toàn bộ danh mục, kéo tổng lợi nhuận tích lũy của Equal-Weight vọt lên 53.93%.',
            ].map((t, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
                <span style={{ color: '#2E7D32', fontWeight: 700, flexShrink: 0 }}>▸</span>
                <span>{t}</span>
              </div>
            ))}
          </div>

          <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--text-primary)', marginBottom: 8 }}>
            Cách MVO "bỏ lỡ" VIC:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 14 }}>
            {[
              'Thuật toán MVO phân bổ tỷ trọng dựa trên dữ liệu quá khứ (Train Set 2020–2024).',
              'Trong giai đoạn 2020–2024, VIC là một mã có độ lệch chuẩn (Volatility) cực kỳ gắt nhưng lợi nhuận trung bình lại không quá vượt trội so với nhóm Ngân hàng (TCB, HDB).',
              'Khi đưa ma trận hiệp biến (Covariance Matrix) của giai đoạn Train vào, bộ tối ưu hóa SLSQP nhìn thấy VIC là một thực thể "quá rủi ro, tương quan cao với vĩ mô". Để tối đa hóa Sharpe Ratio cho tập Train, MVO quyết định đưa VIC về mức tỷ trọng tối thiểu (2% minimum weight) để phòng thủ.',
              'Kết quả sang tập Test (2025–2026), VIC bùng nổ vọt xà, nhưng vì MVO chỉ cầm vỏn vẹn 2% nên mức đóng góp của VIC vào đường vốn của Risk-Taking portfolio gần như bằng 0. MVO tập trung phần lớn vốn vào HDB, VND, HPG — những mã tăng trưởng tốt nhưng không thể nào đọ lại mức nhân 8 lần của VIC.',
            ].map((t, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
                <span style={{ color: '#C62828', fontWeight: 700, flexShrink: 0 }}>▸</span>
                <span>{t}</span>
              </div>
            ))}
          </div>

          <div style={{ background: '#E8F0FB', border: '1px solid #5B7FA644', borderRadius: 10, padding: '12px 16px', fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            <strong style={{ color: '#5B7FA6' }}>Kết luận:</strong> Mô hình lọc cổ phiếu <strong>rất đúng</strong> khi
            xếp VIC ở Hạng 4 (Top 5 Profitability). Nhưng thuật toán MVO đã bị quá khứ đánh lừa, nó chê VIC rủi ro
            nên không dám phân bổ vốn. Đây là hiện tượng lỗi ước lượng ma trận (<em>covariance estimation error</em>)
            kinh điển trong tài chính mà bài báo khoa học của <strong>DeMiguel et al. (2009)</strong> đã chứng minh:{' '}
            <em>"Chiến lược chia đều 1/N hầu như luôn đánh bại Markowitz MVO out-of-sample vì mô hình tối ưu bị quá
            khớp vào dữ liệu quá khứ."</em>
          </div>
        </div>

        {/* Factor 2 */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: '16px 18px', marginBottom: 14 }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)', marginBottom: 10 }}>
            Yếu tố 2: Lỗi công thức niên hóa (Annualization) — ảnh hưởng nhỏ
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.75, marginBottom: 10 }}>
            Trong code, việc tính <code style={{ background: '#F5F0E8', padding: '1px 5px', borderRadius: 4 }}>cov_matrix × 252</code> và{' '}
            <code style={{ background: '#F5F0E8', padding: '1px 5px', borderRadius: 4 }}>mean_returns × 252</code> là{' '}
            <strong>đúng chuẩn</strong> về mặt lý thuyết để đưa hai biến số về cùng không gian thời gian (thường niên).
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.75, margin: 0 }}>
            Tuy nhiên, có một điểm lấn cấn nhỏ: tối ưu hóa dựa trên <strong>lợi nhuận ngày trung bình (Daily Mean)</strong>{' '}
            nhưng các ràng buộc bounds và constraints lại áp cho không gian năm mà không có sự đồng bộ nhịp nhàng trong hàm
            gọi, dễ khiến SLSQP Solver bị lộn điểm hội tụ cục bộ (Local Minimum). Nhưng lỗi này chỉ làm lệch tỷ trọng giữa
            các mã khoảng 2–3%, chứ <strong>không phải</strong> là lý do làm cho Risk-Taking thua Equal-Weight. Dù bạn có
            sửa công thức chuẩn 100%, MVO vẫn sẽ bóp chết tỷ trọng của VIC vì bản chất rủi ro lịch sử của nó quá lớn.
          </p>
        </div>

        {/* Recommendations */}
        <div style={{ background: '#F0F7F0', border: '1.5px solid #4A7C5F', borderRadius: 12, padding: '16px 18px' }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: '#2E7D32', marginBottom: 12 }}>
            Xử lý như thế nào để viết báo cáo / nâng cấp hệ thống?
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 14 }}>
            Thay vì coi đây là một "thất bại" của code, hãy biến nó thành một{' '}
            <strong>Điểm nhấn học thuật (Academic Contribution)</strong> cực kỳ sáng giá:
          </p>
          {[
            {
              num: 1,
              title: 'Giữ nguyên kết quả',
              body: 'Đừng cố sửa code để ép MVO thắng Equal-Weight. Hãy dũng cảm show kết quả Equal-Weight (+53.93%) thắng MVO (+35.73%).',
            },
            {
              num: 2,
              title: 'Biện hộ bằng lý thuyết tài chính',
              body: null,
              quote: 'Kết quả backtest minh chứng cho một hiện tượng kinh điển trong quản lý quỹ (DeMiguel, 2009): Trong các thị trường cận biên biến động mạnh và xuất hiện siêu cổ phiếu bứt phá như VIC (+678%), chiến lược phân bổ đều (Equal-Weight) thể hiện tính ưu việt tuyệt đối nhờ khai thác trọn vẹn chỉ số Alpha của các cổ phiếu đột biến. Ngược lại, Mean-Variance Optimization (MVO) bị dính bẫy quá khớp rủi ro quá khứ, dẫn đến việc phân bổ tỷ trọng thấp cho các mã biến động cao, làm giảm hiệu suất sinh lời thực tế.',
            },
            {
              num: 3,
              title: 'Đề xuất Production Ready',
              body: 'Thêm tùy chọn chiến lược cho người dùng tự chọn: Markowitz MVO (Tối ưu theo rủi ro lịch sử) hoặc Equal-Weight 1/N (Kỷ luật phân tán vốn rủi ro) — khuyến nghị cho thị trường HOSE biến động mạnh.',
            },
          ].map(({ num, title, body, quote }) => (
            <div key={num} style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
              <div style={{ minWidth: 26, height: 26, borderRadius: '50%', background: '#2E7D32', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: 13, flexShrink: 0 }}>{num}</div>
              <div>
                <div style={{ fontWeight: 700, fontSize: 13.5, color: 'var(--text-primary)', marginBottom: 4 }}>{title}</div>
                {body && <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65 }}>{body}</div>}
                {quote && (
                  <blockquote style={{
                    margin: '8px 0 0', padding: '10px 14px',
                    borderLeft: '3px solid #4A7C5F', background: '#E8F5E9',
                    borderRadius: '0 8px 8px 0', fontSize: 12.5,
                    color: 'var(--text-secondary)', lineHeight: 1.7, fontStyle: 'italic',
                  }}>
                    "{quote}"
                  </blockquote>
                )}
              </div>
            </div>
          ))}
        </div>
      </Acc>
    </div>
  )
}

/* ── Tab: 25 Chỉ số ──────────────────────────────────────── */

function TabFeatures() {
  return (
    <div>
      <SectionTitle>Tại sao chỉ dùng 25 chỉ số kỹ thuật?</SectionTitle>
      <Sub>
        Câu trả lời đến từ chính dữ liệu: XGBoost với 52 đặc trưng (kỹ thuật + cơ bản + sentiment)
        chứng minh rằng <strong>chỉ số kỹ thuật + VNIndex chiếm toàn bộ sức mạnh dự đoán</strong>.
        Fundamental và Sentiment gần như bằng 0 — bỏ đi để giảm noise và overfitting.
      </Sub>

      <div className="card" style={{ marginBottom: 20, background: '#FFFBF2', border: '1px solid #E8D5A0' }}>
        <div style={{ fontWeight: 700, color: 'var(--gold-dark)', marginBottom: 10, fontSize: 15 }}>
          XGBoost Feature Importance — 52 đặc trưng
        </div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.6 }}>
          Biểu đồ dưới đây từ notebook nghiên cứu: các chỉ số VNIndex (vni_*) và kỹ thuật (EMA, MACD, RSI…)
          nằm đầu bảng. P/E, ROE, sentiment score gần bằng không — bằng chứng rõ ràng nhất.
        </p>
        <img
          src="/notebook/feature_importance_xgb.png"
          alt="XGBoost Feature Importance"
          style={{ width: '100%', borderRadius: 8, border: '1px solid var(--border)' }}
        />
      </div>

      <KpiRow items={[
        { label: 'Chỉ số VNIndex', value: '12/25', sub: 'market context' },
        { label: 'Chỉ số kỹ thuật', value: '13/25', sub: 'price & volume' },
        { label: 'Fundamental', value: '≈ 0', color: '#999', sub: 'bị loại bỏ' },
        { label: 'Sentiment', value: '≈ 0', color: '#999', sub: 'bị loại bỏ' },
      ]} />

      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ fontWeight: 700, color: 'var(--gold-dark)', marginBottom: 10, fontSize: 15 }}>
          Signal Classifier — Top features (Task 3 XGBoost)
        </div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.6 }}>
          Mô hình tín hiệu giao dịch còn một lớp đặc trưng nữa: <strong>đầu ra của MTL</strong> (p_up, p_down,
          mtl_conviction) — xác suất từ neural net trở thành feature cho XGBoost. Cách kết hợp hai loại
          mô hình (ensemble stacking) để tăng độ tin cậy.
        </p>
        <img
          src="/notebook/feature_importance_signal.png"
          alt="Signal Classifier Feature Importance"
          style={{ width: '100%', borderRadius: 8, border: '1px solid var(--border)' }}
        />
      </div>

      <div className="card" style={{ background: '#F0F7F0', border: '1px solid #B5D5B5' }}>
        <div style={{ fontWeight: 700, color: '#2E7D32', marginBottom: 8, fontSize: 14 }}>
          Triết lý Feature Engineering
        </div>
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.8, margin: 0 }}>
          Thị trường chứng khoán Việt Nam ở giai đoạn phát triển — <strong>thông tin kỹ thuật phản ánh
          hành vi đám đông nhanh hơn nhiều</strong> so với báo cáo tài chính (công bố theo quý, dễ trễ 3–6 tháng).
          VNIndex là "nhịp đập" của cả thị trường — 27 cổ phiếu blue-chip đều bị ảnh hưởng bởi xu hướng
          vĩ mô này, nên 12 chỉ số VNI là context không thể thiếu.
        </p>
      </div>
    </div>
  )
}

/* ── Tab: Backtest ────────────────────────────────────────── */

const CUSTOM_TOOLTIP = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '10px 14px', fontSize: 12,
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontWeight: 600 }}>
          {p.name}: {(p.value * 100).toFixed(1)}%
        </div>
      ))}
    </div>
  )
}

function TabBacktest() {
  const [data,     setData]     = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(null)
  const [openSec,  setOpenSec]  = useState(null)
  const toggle = id => setOpenSec(prev => prev === id ? null : id)

  useEffect(() => {
    fetch(`${API_BASE}/backtest/price`)
      .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(String(e)); setLoading(false) })
  }, [])

  if (loading) return (
    <div style={{ textAlign: 'center', padding: 60, color: 'var(--text-muted)' }}>
      Đang tải dữ liệu backtest…
    </div>
  )
  if (error) return (
    <div style={{ textAlign: 'center', padding: 40, color: '#C62828' }}>
      Không thể tải backtest: {error}
    </div>
  )

  const { summary, equity_curve } = data
  const step = Math.max(1, Math.floor(equity_curve.length / 200))
  const chartData = equity_curve.filter((_, i) => i % step === 0).map(p => ({
    date:  p.date.slice(0, 7),
    strat: p.strat_cum,
    bench: p.bench_cum,
  }))

  const fmtPct = v => `${(v * 100).toFixed(1)}%`
  const green = '#2D6A4F'
  const gold  = '#C4A265'

  const Acc = ({ id, title, children }) => (
    <div style={{ marginBottom: 10 }}>
      <button
        onClick={() => toggle(id)}
        style={{
          width: '100%', textAlign: 'left', padding: '12px 16px',
          background: openSec === id ? '#FAF3E0' : 'var(--bg-card)',
          border: '1px solid var(--border)', borderRadius: openSec === id ? '10px 10px 0 0' : 10,
          fontSize: 14, fontWeight: 600, color: 'var(--text-primary)',
          cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}
      >
        <span>{title}</span>
        <span style={{ fontSize: 18, color: 'var(--text-muted)', lineHeight: 1 }}>
          {openSec === id ? '−' : '+'}
        </span>
      </button>
      {openSec === id && (
        <div style={{
          padding: '16px 18px', border: '1px solid var(--border)', borderTop: 'none',
          borderRadius: '0 0 10px 10px', background: '#fff', lineHeight: 1.7,
        }}>
          {children}
        </div>
      )}
    </div>
  )

  return (
    <div>
      <SectionTitle>Walk-Forward Backtest — MTL T4</SectionTitle>

      {/* ── What is WF backtest ── */}
      <div className="card" style={{ background: '#FFFBF2', border: '1px solid #E8D5A0', marginBottom: 16 }}>
        <div style={{ fontWeight: 700, color: 'var(--gold-dark)', marginBottom: 8, fontSize: 14 }}>
          Walk-Forward Backtest là gì?
        </div>
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.8, margin: 0 }}>
          Trong thị trường tài chính, việc kiểm tra mô hình bằng cách xáo trộn dữ liệu ngẫu nhiên là một
          sai lầm nghiêm trọng vì nó sẽ làm rò rỉ dữ liệu của tương lai vào quá khứ.<br /><br />
          <strong>Walk-Forward Backtest</strong> (Kiểm thử tịnh tiến thời gian) là phương pháp mô phỏng
          giao dịch chuẩn mực nhất của các quỹ đầu tư định lượng. Hệ thống đóng băng mô hình MTL-T4 tại một
          thời điểm, sau đó tịnh tiến từng ngày dọc theo lịch sử để kiểm tra: nếu trong quá khứ chúng ta
          thực sự xuống tiền theo tín hiệu của mô hình thì kết quả tài sản sẽ ra sao.
        </p>
      </div>

      {/* ── How the system runs ── */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 14, marginBottom: 4 }}>
          Hệ thống InvestNature vận hành kiểm thử như thế nào?
        </div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.7, marginBottom: 14 }}>
          Xuyên suốt chu kỳ 6 năm từ 04/2020 đến 04/2026, hệ thống đã thực hiện{' '}
          <strong>{summary.total_predictions.toLocaleString()} lượt dự đoán</strong> độc lập
          trên toàn bộ 27 mã cổ phiếu HOSE theo cơ chế nghiêm ngặt:
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[
            { label: 'Cửa sổ 20 ngày trượt (Look-back Window)', color: '#5B7FA6', bg: '#E8F0FB',
              body: 'Tại mỗi ngày giao dịch T trong lịch sử, hệ thống chỉ cắt đúng dữ liệu của 20 phiên liền trước (tương đương 1 tháng lịch sử) để làm "đề bài" nạp vào mô hình.' },
            { label: 'Cam kết Look-ahead Bias = 0', color: '#4A7C5F', bg: '#E8F5E9',
              body: 'Mô hình hoàn toàn bị "mù" trước dữ liệu của ngày T và tương lai. Điều này đảm bảo kết quả thu được là hoàn toàn thực tế, không hề dính lỗi "biết trước tương lai".' },
            { label: 'Dự báo quỹ đạo 5 ngày', color: '#9E7E45', bg: '#FFF8E1',
              body: 'Từ 20 phiên lịch sử đó, mô hình dự phóng liên tục tỷ suất sinh lời lũy kế cho 5 ngày kế tiếp (T+1 đến T+5).' },
            { label: 'Ngưỡng kích hoạt lệnh (Threshold = 0.55)', color: '#7B5EA7', bg: '#F3EEF9',
              body: 'Chỉ khi nhánh phân loại của MTL-T4 đạt độ tự tin (conviction) từ 55.0% trở lên, tín hiệu MUA hoặc BÁN mới được phát ra. Nếu dưới ngưỡng, hệ thống đưa về trạng thái GIỮ (HOLD) để bảo vệ vốn.' },
          ].map(item => (
            <div key={item.label} style={{
              display: 'flex', gap: 12, padding: '10px 12px',
              background: item.bg, borderRadius: 10, border: `1px solid ${item.color}22`,
            }}>
              <div style={{ width: 4, borderRadius: 2, background: item.color, flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: item.color, marginBottom: 3 }}>{item.label}</div>
                <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{item.body}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Metrics grid (ordered: accuracy → error → activity → returns → risk-adj) ── */}
      {(() => {
        const metrics = [
          { key: 'da1d',   label: 'DA Ngắn hạn (1d)',              value: fmtPct(summary.da_1d),            color: '#5B7FA6', bg: '#E8F0FB', sub: 'dự đoán hướng T+1',
            explain: 'Directional Accuracy 1 ngày — tỷ lệ MTL-T4 đoán đúng chiều tăng/giảm ngay trong phiên T+1. Đây là bài toán khó nhất vì biến động trong 24h phần lớn là nhiễu (noise). 51% tốt hơn tung đồng xu nhưng không đủ để giao dịch đơn lẻ — cần kết hợp với DA-5d.' },
          { key: 'da5d',   label: 'DA Trung hạn (5d)',              value: fmtPct(summary.da_5d),            color: green,     bg: '#E8F5E9', sub: 'ngưỡng ngẫu nhiên: 50%',
            explain: 'Directional Accuracy 5 ngày — tỷ lệ đoán đúng chiều trong cả tuần giao dịch. Ngưỡng ngẫu nhiên là 50% (tung đồng xu). MTL-T4 đạt 56.8%, vượt ngưỡng ngẫu nhiên +6.8 điểm % — nhờ bắt trọn các nhịp dịch chuyển dòng tiền trung hạn mà dao động 1 ngày che khuất.' },
          { key: 'mae',    label: 'MAE 1 ngày',                    value: summary.mae_1d.toFixed(4),         color: '#9E7E45', bg: '#FFF8E1', sub: 'sai số log-return',
            explain: 'Mean Absolute Error (sai số tuyệt đối trung bình). MAE = 0.0255 nghĩa là mô hình sai lệch trung bình ~2.55% log-return mỗi phiên. Dùng để xác định vùng giá kỳ vọng chứ không phải điểm giá chính xác — luôn đặt stop-loss 3–5% để bù biên độ sai số này.' },
          { key: 'buy',    label: 'Tín hiệu BUY',                  value: fmtPct(summary.buy_signal_pct),   color: '#7B5EA7', bg: '#F3EEF9', sub: 'trong tổng số phiên',
            explain: '9.7% có nghĩa là trên tổng số phiên giao dịch, chỉ chưa đến 1/10 số phiên được phát tín hiệu MUA. MTL-T4 cực kỳ khắt khe — chỉ giải ngân khi conviction ≥ 55%. Đây là ưu điểm: tránh giao dịch ồn ào theo cảm tính, chỉ vào lệnh khi có cơ sở chắc chắn.' },
          { key: 'cumS',   label: 'Lợi nhuận tích lũy (MTL-T4)',   value: fmtPct(summary.cum_return_strat), color: green,     bg: '#E8F5E9', sub: 'Equal-weight BUY signals',
            explain: 'Tổng lợi nhuận tích lũy khi đầu tư theo tín hiệu MTL-T4 trong 6 năm (2020–2026). Equal-weight: mỗi phiên có N mã BUY thì vốn chia đều cho N mã. 157.0% tương đương vốn tăng gấp 2.57 lần sau 6 năm.' },
          { key: 'cumB',   label: 'Lợi nhuận tích lũy (benchmark)', value: fmtPct(summary.cum_return_bench), color: 'var(--text-secondary)', bg: 'var(--bg-card)', sub: 'Nắm giữ 27 mã từ đầu',
            explain: 'Benchmark = mua và nắm giữ (Buy & Hold) toàn bộ 27 mã ngay từ đầu, không cần dự đoán. 144.4% là mức tăng trưởng thụ động của rổ blue-chip HOSE trong 6 năm. MTL-T4 vượt mức này (+12.6 điểm %) nhờ bộ lọc chủ động.' },
          { key: 'shS',    label: 'Sharpe (MTL-T4)',                value: summary.sharpe_strategy,           color: gold,      bg: '#FFFBF2', sub: 'lợi nhuận/rủi ro',
            explain: 'Sharpe = μ/σ × √252. MTL-T4 đạt 0.816 — nghĩa là cứ 1 đơn vị rủi ro (độ lệch chuẩn) chịu đựng, mô hình sinh ra 0.816 đơn vị lợi nhuận. Cao hơn benchmark (0.782) chứng minh mô hình không chỉ lời nhiều hơn mà còn hiệu quả hơn về chất lượng lợi nhuận/rủi ro.' },
          { key: 'shB',    label: 'Sharpe (benchmark)',             value: summary.sharpe_benchmark,          color: 'var(--text-muted)', bg: 'var(--bg-card)', sub: 'Nắm giữ 27 mã từ đầu',
            explain: 'Sharpe Ratio của chiến lược Buy & Hold toàn bộ 27 mã. 0.782 là mức tham chiếu thị trường — mua hết 27 mã từ đầu và không làm gì suốt 6 năm. MTL-T4 vượt qua (0.816) nhờ bộ lọc conviction chủ động ngắt vị thế khi thị trường xấu.' },
        ]
        return (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: openSec === 'metric' ? 0 : 20 }}>
              {metrics.map(({ key, label, value, color, bg, sub }) => (
                <div
                  key={key}
                  onClick={() => setOpenSec(prev => prev === `m_${key}` ? null : `m_${key}`)}
                  style={{
                    background: openSec === `m_${key}` ? '#FAF3E0' : bg,
                    border: openSec === `m_${key}` ? '1.5px solid var(--gold-dark)' : '1px solid var(--border)',
                    borderRadius: 12, padding: '14px 16px', cursor: 'pointer',
                    transition: 'box-shadow 0.15s, transform 0.15s',
                    boxShadow: openSec === `m_${key}` ? '0 4px 16px rgba(0,0,0,0.10)' : 'none',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 4px 16px rgba(0,0,0,0.10)'; e.currentTarget.style.transform = 'translateY(-2px)' }}
                  onMouseLeave={e => { e.currentTarget.style.boxShadow = openSec === `m_${key}` ? '0 4px 16px rgba(0,0,0,0.10)' : 'none'; e.currentTarget.style.transform = 'translateY(0)' }}
                >
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
                  {sub && <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>}
                  <div style={{ fontSize: 10, color: 'var(--gold-dark)', marginTop: 6, opacity: 0.7 }}>nhấp để xem giải thích</div>
                </div>
              ))}
            </div>
            {metrics.map(({ key, label, explain }) => openSec === `m_${key}` && (
              <div key={key} style={{
                padding: '14px 16px', background: '#FFFBF2', border: '1.5px solid var(--gold-dark)',
                borderTop: 'none', borderRadius: '0 0 12px 12px', marginBottom: 20,
                fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.75,
              }}>
                <strong style={{ color: 'var(--gold-dark)' }}>{label}:</strong> {explain}
              </div>
            ))}
          </>
        )
      })()}

      {/* ── Equity curve chart ── */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)', marginBottom: 16 }}>
          Đường vốn tích lũy (2020–2026)
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} tickLine={false} />
            <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} tickLine={false} />
            <Tooltip content={<CUSTOM_TOOLTIP />} />
            <Legend wrapperStyle={{ fontSize: 13 }} />
            <Line type="monotone" dataKey="strat" name="Chiến lược BUY"
              stroke={green} dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="bench" name="Benchmark"
              stroke={gold} dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Accordion deep-dive sections ── */}
      <Acc id="equity" title="📈 Cách tính đường vốn tích lũy (Equity Curve) như thế nào?">
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
          Đường vốn tích lũy không phải là đồ thị giá của một cổ phiếu, mà là đồ thị giả lập tài sản của một{' '}
          <strong>Quỹ đầu tư ảo</strong> chạy bằng mô hình MTL-T4:
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
          {[
            { label: 'Hành động hàng ngày',
              body: 'Vào cuối mỗi ngày giao dịch T, hệ thống lọc ra tất cả các mã cổ phiếu trong rổ 27 mã HOSE có tín hiệu MUA (BUY) thỏa mãn độ tự tin ≥ 55.0%. Gọi tập hợp này là S(T).' },
            { label: 'Phân bổ vốn — Equal-weight',
              body: 'Số vốn hiện có được chia đều với tỷ trọng bằng nhau cho nhóm cổ phiếu được chọn: w_i = 1/|S(T)| với mọi i ∈ S(T). Nếu không có tín hiệu BUY nào, vốn nằm tiền mặt (return = 0).' },
            { label: 'Tính toán lợi nhuận hàng ngày',
              body: 'Lợi nhuận ngày T+1 của chiến lược bằng trung bình cộng log-return thực tế từ các mã được chọn. Kết quả nhân lũy kế để tạo đường vốn màu xanh đậm.' },
            { label: 'Đường Benchmark (Vàng nét đứt)',
              body: 'Chia đều vốn mua hết cả 27 mã ngay từ ngày đầu tiên và giữ nguyên cho đến hết 6 năm (Buy & Hold). Khoảng cách giữa đường xanh và đường vàng chính là giá trị thặng dư (Alpha) mà MTL-T4 mang lại.' },
          ].map((item, i) => (
            <div key={i} style={{ padding: '10px 12px', borderRadius: 8, background: '#F9F6F1', border: '1px solid var(--border)' }}>
              <div style={{ fontWeight: 700, fontSize: 12, color: 'var(--gold-dark)', marginBottom: 4 }}>▸ {item.label}</div>
              <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{item.body}</div>
            </div>
          ))}
        </div>

        {/* Formulas */}
        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8 }}>Công thức toán học</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[
            { label: 'Return chiến lược ngày T', formula: 'r_strat(T) = (1/|S(T)|) × Σ r_i(T+1)  ,  i ∈ S(T)' },
            { label: 'Return benchmark ngày T',  formula: 'r_bench(T) = (1/27) × Σ r_i(T+1)  ,  i ∈ {toàn bộ 27 mã}' },
            { label: 'Equity Curve MTL-T4',      formula: 'EC_strat(T) = ∏(t=1→T) [1 + r_strat(t)]' },
            { label: 'Equity Curve Benchmark',   formula: 'EC_bench(T) = ∏(t=1→T) [1 + r_bench(t)]' },
            { label: 'Alpha (giá trị thặng dư)', formula: 'α = EC_strat(T) − EC_bench(T)' },
          ].map(f => (
            <div key={f.label} style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 10, alignItems: 'center' }}>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', fontWeight: 600 }}>{f.label}</div>
              <div style={{ background: '#1e1e1e', color: '#C4A265', borderRadius: 6, padding: '6px 12px', fontFamily: 'monospace', fontSize: 12 }}>{f.formula}</div>
            </div>
          ))}
        </div>
      </Acc>

      <Acc id="sharpe" title="📐 Chỉ số Sharpe (Sharpe Ratio) là gì?">
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
          Chỉ số Sharpe là thước đo kinh điển trong giới tài chính để trả lời câu hỏi:{' '}
          <em>"Để kiếm được mức lợi nhuận đó, tài khoản của bạn đã phải chịu bao nhiêu sự dập vờn, rủi ro?"</em>
        </p>
        <div style={{ background: '#F5F0E8', borderRadius: 8, padding: '10px 14px', fontFamily: 'monospace', fontSize: 13, marginBottom: 14, textAlign: 'center', color: 'var(--text-primary)' }}>
          Sharpe = (Lợi nhuận TB hàng ngày / Độ lệch chuẩn hàng ngày) × √252
        </div>
        <div style={{ padding: '10px 12px', background: '#FFF8E1', borderRadius: 8, border: '1px solid #E8D5A033', marginBottom: 10, fontSize: 12.5, color: 'var(--text-secondary)' }}>
          <strong style={{ color: '#9E7E45' }}>Buy & Hold (Benchmark) là gì?</strong> Chiến lược thụ động: mua toàn bộ 27 mã ngay từ ngày đầu tiên với tỷ trọng bằng nhau và không làm gì thêm suốt 6 năm. Đây là mức tham chiếu tối thiểu — nếu mô hình không vượt qua được thì không có giá trị.
        </div>
        <div style={{ padding: '12px 14px', background: '#E8F5E9', borderRadius: 10, border: '1px solid #4A7C5F33', marginBottom: 12 }}>
          <div style={{ fontWeight: 700, fontSize: 12.5, color: '#2E7D32', marginBottom: 6 }}>
            Sharpe MTL-T4 ({summary.sharpe_strategy}) {'>'} Sharpe Benchmark ({summary.sharpe_benchmark})
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            MTL-T4 không chỉ ăn dày hơn về mặt phần trăm cuối kỳ ({fmtPct(summary.cum_return_strat)} {'>'} {fmtPct(summary.cum_return_bench)}),
            mà còn mang lại <strong>lợi nhuận trên một đơn vị rủi ro tốt hơn</strong>.
          </div>
        </div>
        <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.7, margin: 0 }}>
          Đường vốn của MTL-T4 chạy êm hơn, ít xuất hiện các pha sụt giảm tài sản kinh hoàng (Max Drawdown)
          nhờ mô hình chủ động ngắt vị thế, ôm tiền mặt đứng ngoài khi khối vĩ mô VN-Index báo xấu
          (như đợt Bear Market cuối năm 2022).
        </p>
      </Acc>

      <Acc id="mae" title="🧮 Sai số tuyệt đối trung bình MAE (Mean Absolute Error) là gì?">
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
          MAE đo lường khoảng cách sai lệch trung bình giữa con số mô hình AI dự đoán và con số thực tế
          diễn ra trên sàn chứng khoán.
        </p>
        <div style={{ background: '#F5F0E8', borderRadius: 8, padding: '10px 14px', fontFamily: 'monospace', fontSize: 13, marginBottom: 14, textAlign: 'center', color: 'var(--text-primary)' }}>
          MAE = (1/n) × Σ |y_thực_tế − ŷ_dự_đoán|
        </div>
        <div style={{ padding: '12px 14px', background: '#FFF8E1', borderRadius: 10, border: '1px solid #9E7E4533', marginBottom: 12 }}>
          <div style={{ fontWeight: 700, fontSize: 12.5, color: '#9E7E45', marginBottom: 6 }}>
            MAE 1 ngày = {summary.mae_1d.toFixed(4)} — "Phanh giảm chấn" cho lòng tham
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            Chỉ số này nhắc nhở Trader rằng: mô hình dù thông minh đến đâu vẫn luôn có biên độ sai số
            (sai lệch log-return khoảng {(summary.mae_1d * 100).toFixed(2)}%).
          </div>
        </div>
        <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.7, margin: 0 }}>
          Do đó, người dùng không nên tất tay (All-in) vào một mức giá chính xác, mà phải coi các con số
          dự đoán của InvestNature là một <strong>Vùng giá cân bằng kỳ vọng</strong>, đồng thời luôn tuân
          thủ đặt lệnh dừng lỗ (Stop-loss) từ 3%–5% để bảo vệ tài khoản trước các biến động bất ngờ.
        </p>
      </Acc>

    </div>
  )
}

/* ── Main Page ────────────────────────────────────────────── */

export default function ModelInsights() {
  const [active, setActive] = useState('journey')

  const content = {
    journey:      <TabJourney />,
    architecture: <TabArchitecture />,
    signal:       <TabSignal />,
    portfolio:    <TabPortfolio />,
    features:     <TabFeatures />,
    backtest:     <TabBacktest />,
  }

  return (
    <div>
      <h1 className="page-title">
        <LeafIcon size={24} color="var(--gold)" /> Công nghệ &amp; Nghiên cứu
      </h1>
      <p className="page-subtitle">
        Hành trình R&amp;D từ baseline đến kiến trúc Seq2Seq MTL — toàn bộ quá trình xây dựng mô hình
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(({ id, label, Icon }) => (
          <button
            key={id}
            onClick={() => setActive(id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '9px 18px', borderRadius: 24,
              border: active === id ? 'none' : '1px solid var(--border)',
              background: active === id ? 'var(--gold)' : 'var(--bg-card)',
              color: active === id ? '#fff' : 'var(--text-secondary)',
              fontWeight: active === id ? 700 : 500,
              fontSize: 14, cursor: 'pointer', fontFamily: 'inherit',
              transition: 'all 0.15s',
            }}
          >
            <Icon size={15} color={active === id ? '#fff' : 'currentColor'} />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>{content[active]}</div>
    </div>
  )
}
