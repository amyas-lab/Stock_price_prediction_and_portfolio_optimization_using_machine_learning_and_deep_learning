import {
  ResponsiveContainer, ComposedChart,
  Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts'

function fmtVND(v) {
  if (v >= 1000) return `${(v / 1000).toFixed(1)}k`
  return v.toLocaleString('vi-VN')
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#1e293b', border: '1px solid #334155',
      borderRadius: 6, padding: '10px 14px', fontSize: 13,
    }}>
      <p style={{ color: '#94a3b8', marginBottom: 6 }}>{label}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color, fontWeight: 600 }}>
          {p.name}: {p.value?.toLocaleString('vi-VN')} VND
        </p>
      ))}
    </div>
  )
}

export default function PriceChart({ currentPrice, predictedPrices, predictionDate, direction }) {
  const accentColor = direction === 'UP' ? '#10b981' : direction === 'DOWN' ? '#ef4444' : '#3b82f6'

  const data = [
    { day: `${predictionDate} (Hiện tại)`, actual: currentPrice },
    ...predictedPrices.map((p, i) => ({
      day: `Ngày +${i + 1}`,
      predicted: p,
    })),
  ]

  // Connect actual to first predicted
  if (data.length > 1) {
    data[0].predicted = currentPrice
  }

  const allPrices = [currentPrice, ...predictedPrices].filter(Boolean)
  const minP = Math.min(...allPrices) * 0.997
  const maxP = Math.max(...allPrices) * 1.003

  return (
    <div className="chart-wrap">
      <p className="chart-title">Dự báo giá (VND)</p>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={data} margin={{ top: 8, right: 20, bottom: 0, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="day"
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[minP, maxP]}
            tickFormatter={fmtVND}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={60}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, color: '#94a3b8', paddingTop: 8 }}
          />
          <ReferenceLine x={`${predictionDate} (Hiện tại)`} stroke="#475569" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="actual"
            name="Giá hiện tại"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={{ r: 5, fill: '#3b82f6' }}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="predicted"
            name="Dự báo"
            stroke={accentColor}
            strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={{ r: 5, fill: accentColor }}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
