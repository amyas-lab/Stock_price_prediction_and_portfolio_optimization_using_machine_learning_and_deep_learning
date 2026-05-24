import axios from 'axios'

// In dev: requests go to /api/* which Vite proxies to localhost:8000
// In prod: set VITE_API_URL to your HF Spaces URL
const BASE = import.meta.env.VITE_API_URL || '/api'

const http = axios.create({
  baseURL: BASE,
  timeout: 30_000,
})

export async function fetchTickers() {
  const { data } = await http.get('/tickers')
  return data
}

export async function predictPrice(ticker, nDaysBack = 20) {
  const { data } = await http.post('/predict/price', {
    ticker,
    n_days_back: nDaysBack,
  })
  return data
}

export async function predictSignal(ticker, threshold = 0.55) {
  const { data } = await http.post('/predict/signal', {
    ticker,
    threshold,
  })
  return data
}

export async function fetchPortfolio(profile) {
  const { data } = await http.get(`/portfolio/${profile}`)
  return data
}

export async function fetchProfitabilityScores() {
  const { data } = await http.get('/portfolio/scores/profitability')
  return data
}

export async function fetchRiskScores() {
  const { data } = await http.get('/portfolio/scores/risk')
  return data
}

export async function computePortfolio(tickers, weights) {
  const { data } = await http.post('/portfolio/compute', { tickers, weights })
  return data
}

export async function checkHealth() {
  const { data } = await http.get('/health')
  return data
}
