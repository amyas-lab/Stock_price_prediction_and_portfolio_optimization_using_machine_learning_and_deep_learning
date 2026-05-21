export default function TickerBadge({ isKnown, dataSource }) {
  if (isKnown) {
    return (
      <span className="badge badge-known" title="Dùng model chuyên biệt, features tính sẵn">
        ✓ Model chuyên biệt
      </span>
    )
  }
  return (
    <span className="badge badge-unknown" title="Dùng model tổng quát, dữ liệu tải trực tuyến">
      ⚠ Dự báo tổng quát
    </span>
  )
}
