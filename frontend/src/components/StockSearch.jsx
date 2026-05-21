import { useState, useRef, useEffect } from 'react'

const KNOWN_TICKERS = [
  'FPT','VCB','VHM','VNM','HPG','VIC','TCB',
  'MSN','MWG','VND','BID','CTG','MBB','ACB',
  'HDB','TPB','SHB','PDR','KDH','DXG','GAS',
  'HSG','PNJ','SAB','CMG','ELC','SGT',
]

export default function StockSearch({ value, onChange, placeholder = 'Nhập mã CK (VD: FPT, SSI, VPB...)' }) {
  const [open, setOpen]   = useState(false)
  const [query, setQuery] = useState(value || '')
  const ref               = useRef(null)

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => { if (!ref.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const filtered = query.trim()
    ? KNOWN_TICKERS.filter(t => t.startsWith(query.toUpperCase()))
    : KNOWN_TICKERS

  const isKnown    = KNOWN_TICKERS.includes(query.toUpperCase())
  const showCustom = query.trim() && !isKnown

  function handleSelect(ticker) {
    setQuery(ticker)
    onChange(ticker)
    setOpen(false)
  }

  function handleInput(e) {
    const v = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '')
    setQuery(v)
    onChange(v)
    setOpen(true)
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') {
      const q = query.trim().toUpperCase()
      if (q) { onChange(q); setOpen(false) }
    }
    if (e.key === 'Escape') setOpen(false)
  }

  return (
    <div className="search-wrap" ref={ref}>
      <input
        className="form-input"
        value={query}
        onChange={handleInput}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        autoComplete="off"
        spellCheck={false}
      />

      {open && (
        <div className="search-dropdown">
          {filtered.length > 0 && (
            <>
              <div className="dropdown-section-label">
                Mã tối ưu (đã huấn luyện)
              </div>
              {filtered.map(ticker => (
                <div
                  key={ticker}
                  className="dropdown-item"
                  onMouseDown={() => handleSelect(ticker)}
                >
                  <span className="ticker-text">{ticker}</span>
                  <span className="badge badge-known" style={{ fontSize: '11px', padding: '1px 7px' }}>
                    Chuyên biệt
                  </span>
                </div>
              ))}
            </>
          )}

          {showCustom && (
            <div
              className="dropdown-item"
              onMouseDown={() => handleSelect(query.toUpperCase())}
            >
              <span className="ticker-text">{query.toUpperCase()}</span>
              <span className="badge badge-unknown" style={{ fontSize: '11px', padding: '1px 7px' }}>
                Tổng quát
              </span>
            </div>
          )}

          <div className="dropdown-hint">
            Gõ bất kỳ mã HOSE — hệ thống sẽ tự tải dữ liệu trực tuyến
          </div>
        </div>
      )}
    </div>
  )
}
