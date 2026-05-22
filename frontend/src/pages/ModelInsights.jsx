import { useState } from 'react'
import { BarChartIcon, TrendUpIcon, ShieldIcon, TargetIcon, LeafIcon } from '../components/Icons.jsx'

const TABS = [
  { id: 'journey',      label: 'Hành trình R&D',      Icon: TrendUpIcon  },
  { id: 'architecture', label: 'Kiến trúc MTL',        Icon: BarChartIcon },
  { id: 'signal',       label: 'Tín hiệu XGBoost',    Icon: ShieldIcon   },
  { id: 'portfolio',    label: 'Tối ưu danh mục',      Icon: TargetIcon   },
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
  return (
    <div>
      <SectionTitle>Hệ thống tối ưu danh mục — Task 4</SectionTitle>
      <Sub>27 mã lớn nhất HOSE · 5 nhân tố chọn cổ phiếu · Markowitz Mean-Variance Optimization · Backtest 2025–2026.</Sub>

      <KpiRow items={[
        { label: 'Equal-Weight Return', value: '143.5%', sub: 'Tổng lợi nhuận backtest', color: '#2E7D32' },
        { label: 'Sharpe Ratio', value: '3.99', sub: 'Equal-Weight portfolio', color: 'var(--gold-dark)' },
        { label: 'Spearman ρ', value: '0.556', sub: 'p-value = 0.0026 (1%)', color: '#2E7D32' },
        { label: 'Precision lift', value: '×2.6', sub: '90% AI pick vs 35% rest', color: 'var(--gold-dark)' },
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

      {/* Backtest table */}
      <SectionTitle style={{ marginTop: 20 }}>Kết quả Backtest (02/2025 – 04/2026)</SectionTitle>
      <div style={{ overflowX: 'auto', marginBottom: 20 }}>
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
              ['Tổng lợi nhuận', '143.51%', '68.61%', '17.93%', '40.74%'],
              ['Lợi nhuận/năm', '118.61%', '56.70%', '14.82%', '33.67%'],
              ['Biến động/năm', '28.60%', '31.33%', '19.86%', '17.89%'],
              ['Sharpe Ratio', '3.99', '1.67', '0.52', '1.63'],
              ['Max Drawdown', '-18.06%', '-26.63%', '-12.72%', 'N/A'],
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

      <div className="card" style={{ background: '#F0F7F0', border: '1px solid #B5D5B5' }}>
        <div style={{ fontWeight: 700, color: '#2E7D32', marginBottom: 8, fontSize: 14 }}>
          Tại sao Equal-Weight thắng Markowitz?
        </div>
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.8, margin: 0 }}>
          Mô hình xếp hạng <strong>VIC ở vị trí Top 4</strong> — thực tế VIC tăng <strong>+678%</strong> trong giai
          đoạn test. Tuy nhiên bộ tối ưu Markowitz của Risk-Taking, vì e ngại biến động lịch sử của VIC,
          chỉ phân bổ <strong>2% tối thiểu</strong>. Equal-Weight phân bổ đều 10% nên ăn trọn sóng tăng
          lịch sử này — minh chứng cho sức mạnh của bộ lọc chọn cổ phiếu AI.
        </p>
      </div>
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
