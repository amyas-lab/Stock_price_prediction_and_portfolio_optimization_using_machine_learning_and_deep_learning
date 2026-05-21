export default function LoadingSpinner({ ticker, isLive }) {
  const msg = isLive
    ? `Đang thu thập dữ liệu trực tuyến và tính toán 60+ chỉ báo kỹ thuật cho mã ${ticker}...`
    : `Đang tải dữ liệu cho ${ticker}...`

  return (
    <div className="spinner-wrap">
      <div className="spinner" />
      <p className="spinner-label">{msg}</p>
    </div>
  )
}
