export function SunPlantIcon({ size = 32, className = '' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
      <circle cx="16" cy="13" r="5" fill="#C4A265" />
      <line x1="16" y1="3"  x2="16" y2="6.5" stroke="#C4A265" strokeWidth="2"   strokeLinecap="round"/>
      <line x1="16" y1="19.5" x2="16" y2="23" stroke="#C4A265" strokeWidth="2"   strokeLinecap="round"/>
      <line x1="6"  y1="13" x2="9.5"  y2="13" stroke="#C4A265" strokeWidth="2"   strokeLinecap="round"/>
      <line x1="22.5" y1="13" x2="26" y2="13" stroke="#C4A265" strokeWidth="2"   strokeLinecap="round"/>
      <line x1="8.5"  y1="5.5"  x2="11" y2="8"  stroke="#C4A265" strokeWidth="1.5" strokeLinecap="round"/>
      <line x1="21"   y1="18"   x2="23.5" y2="20.5" stroke="#C4A265" strokeWidth="1.5" strokeLinecap="round"/>
      <line x1="23.5" y1="5.5"  x2="21" y2="8"  stroke="#C4A265" strokeWidth="1.5" strokeLinecap="round"/>
      <line x1="8.5"  y1="20.5" x2="11" y2="18" stroke="#C4A265" strokeWidth="1.5" strokeLinecap="round"/>
      <line x1="16" y1="23" x2="16" y2="30" stroke="#5C7A45" strokeWidth="1.8" strokeLinecap="round"/>
      <path d="M16 27 Q13 24 11 22.5" stroke="#5C7A45" strokeWidth="1.5" strokeLinecap="round" fill="none"/>
      <path d="M16 27 Q19 24 21 22.5" stroke="#5C7A45" strokeWidth="1.5" strokeLinecap="round" fill="none"/>
    </svg>
  )
}

export function SunIcon({ size = 48, color = '#C4A265' }) {
  const s = size
  const cx = s / 2, cy = s * 0.42, r = s * 0.16
  const rayLen = s * 0.11, gap = s * 0.09
  const rays = [0,45,90,135,180,225,270,315]
  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} fill="none">
      <circle cx={cx} cy={cy} r={r} fill={color}/>
      {rays.map(deg => {
        const rad = (deg * Math.PI) / 180
        const x1 = cx + Math.cos(rad) * (r + gap)
        const y1 = cy + Math.sin(rad) * (r + gap)
        const x2 = cx + Math.cos(rad) * (r + gap + rayLen)
        const y2 = cy + Math.sin(rad) * (r + gap + rayLen)
        const sw = deg % 90 === 0 ? s * 0.055 : s * 0.038
        return <line key={deg} x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={sw} strokeLinecap="round"/>
      })}
      <line x1={cx} y1={cy + r + gap + rayLen * 0.5} x2={cx} y2={s * 0.92} stroke="#5C7A45" strokeWidth={s * 0.05} strokeLinecap="round"/>
      <path d={`M${cx} ${s*0.76} Q${cx-s*0.12} ${s*0.65} ${cx-s*0.18} ${s*0.6}`} stroke="#5C7A45" strokeWidth={s*0.042} strokeLinecap="round" fill="none"/>
      <path d={`M${cx} ${s*0.76} Q${cx+s*0.12} ${s*0.65} ${cx+s*0.18} ${s*0.6}`} stroke="#5C7A45" strokeWidth={s*0.042} strokeLinecap="round" fill="none"/>
    </svg>
  )
}

export function TrendUpIcon({ size = 20, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
      <polyline points="17 6 23 6 23 12"/>
    </svg>
  )
}

export function PieIcon({ size = 20, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21.21 15.89A10 10 0 1 1 8 2.83"/>
      <path d="M22 12A10 10 0 0 0 12 2v10z"/>
    </svg>
  )
}

export function BarChartIcon({ size = 20, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10"/>
      <line x1="12" y1="20" x2="12" y2="4"/>
      <line x1="6"  y1="20" x2="6"  y2="14"/>
    </svg>
  )
}

export function LeafIcon({ size = 20, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* organic filled leaf — like Figma eco/sprout icon */}
      <path
        d="M21 3C21 3 14 3 9.5 7.5C5.5 11.5 5 17 5 17C5 17 10.5 16.5 14.5 12.5C19 8 21 3 21 3Z"
        fill={color}
      />
      {/* stem */}
      <path
        d="M3 21C3 21 5 18 8 15.5"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  )
}

export function TargetIcon({ size = 20, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <circle cx="12" cy="12" r="6"/>
      <circle cx="12" cy="12" r="2"/>
    </svg>
  )
}

export function ShieldIcon({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  )
}

export function ClockIcon({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
    </svg>
  )
}
