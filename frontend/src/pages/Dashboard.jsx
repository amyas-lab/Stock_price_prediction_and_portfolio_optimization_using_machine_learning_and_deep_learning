import { useNavigate } from 'react-router-dom'
import { SunIcon, TrendUpIcon, PieIcon, BarChartIcon, LeafIcon, TargetIcon, SunPlantIcon } from '../components/Icons.jsx'

const features = [
  {
    Icon: TrendUpIcon,
    name: 'Dự đoán giá cổ phiếu',
    desc: 'Sử dụng AI và machine learning để dự đoán xu hướng giá cổ phiếu dựa trên dữ liệu lịch sử.',
    path: '/predict',
  },
  {
    Icon: PieIcon,
    name: 'Tối ưu danh mục',
    desc: 'Tối ưu hóa phân bổ tài sản để cân bằng lợi nhuận và rủi ro theo mục tiêu của bạn.',
    path: '/portfolio',
  },
  {
    Icon: BarChartIcon,
    name: 'Phân tích chuyên sâu',
    desc: 'Biểu đồ và báo cáo chi tiết giúp bạn hiểu rõ xu hướng thị trường và hiệu suất đầu tư.',
    path: '/predict',
  },
  {
    Icon: LeafIcon,
    name: 'Đầu tư bền vững',
    desc: 'Tích hợp yếu tố ESG và đầu tư có trách nhiệm vào quyết định của bạn.',
    path: '/portfolio',
  },
  {
    Icon: TargetIcon,
    name: 'Mục tiêu cá nhân',
    desc: 'Đặt và theo dõi các mục tiêu đầu tư phù hợp với kế hoạch tài chính của bạn.',
    path: '/portfolio',
  },
  {
    Icon: SunPlantIcon,
    name: 'Giao diện thân thiện',
    desc: 'Thiết kế lấy cảm hứng từ thiên nhiên, dễ sử dụng và mang lại trải nghiệm tốt nhất.',
    path: '/',
  },
]

export default function Dashboard() {
  const navigate = useNavigate()

  return (
    <div>
      <div className="hero">
        <div className="hero-icon">
          <SunIcon size={64} />
        </div>
        <h1 className="hero-title">
          Đầu tư thông minh với<br />InvestNature
        </h1>
        <p className="hero-sub">
          Kết hợp sức mạnh của AI và trực giác tự nhiên để dự đoán giá cổ phiếu
          và tối ưu hóa danh mục đầu tư của bạn
        </p>
        <div className="hero-actions">
          <button className="btn btn-primary" onClick={() => navigate('/predict')}>
            <TrendUpIcon size={16} color="#fff" /> Bắt đầu dự đoán
          </button>
          <button className="btn btn-outline" onClick={() => navigate('/portfolio')}>
            <PieIcon size={16} /> Tối ưu danh mục
          </button>
        </div>
      </div>

      <h2 className="section-heading">Tính năng nổi bật</h2>
      <p className="section-sub">Công cụ đầu tư toàn diện cho nhà đầu tư thông minh</p>

      <div className="feature-grid">
        {features.map((f, i) => (
          <div
            key={i}
            className="feature-card"
            onClick={() => navigate(f.path)}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && navigate(f.path)}
          >
            <div className="feature-card-icon">
              <f.Icon size={22} color="var(--text-secondary)" />
            </div>
            <div className="feature-card-name">{f.name}</div>
            <div className="feature-card-desc">{f.desc}</div>
          </div>
        ))}
      </div>

      <div className="cta-section">
        <div className="cta-icon"><LeafIcon size={36} color="#5C7A45" /></div>
        <h2 className="cta-title">Sẵn sàng phát triển danh mục của bạn?</h2>
        <p className="cta-sub">Bắt đầu hành trình đầu tư thông minh và bền vững ngay hôm nay</p>
        <div className="cta-actions">
          <button className="btn btn-primary" onClick={() => navigate('/predict')}>
            Dự đoán ngay
          </button>
          <button className="btn btn-dark" onClick={() => navigate('/portfolio')}>
            Tối ưu danh mục
          </button>
        </div>
      </div>
    </div>
  )
}
