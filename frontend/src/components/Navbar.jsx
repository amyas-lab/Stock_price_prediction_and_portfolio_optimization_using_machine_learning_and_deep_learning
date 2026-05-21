import { NavLink } from 'react-router-dom'
import { SunPlantIcon, TrendUpIcon, PieIcon } from './Icons.jsx'

const links = [
  { to: '/',          label: 'Trang chủ',       Icon: SunPlantIcon },
  { to: '/predict',   label: 'Dự đoán giá',     Icon: TrendUpIcon  },
  { to: '/portfolio', label: 'Tối ưu danh mục', Icon: PieIcon      },
]

export default function Navbar() {
  return (
    <nav className="navbar">
      <NavLink to="/" className="navbar-brand">
        <SunPlantIcon size={30} />
        <span>InvestNature</span>
      </NavLink>
      <ul className="navbar-links">
        {links.map(({ to, label, Icon }) => (
          <li key={to}>
            <NavLink to={to} className={({ isActive }) => isActive ? 'active' : ''} end={to === '/'}>
              <Icon size={15} /> {label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  )
}
