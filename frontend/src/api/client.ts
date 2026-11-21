import axios from 'axios'

const client = axios.create({
  baseURL: '',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('personal_kb_token')
  const tenant = localStorage.getItem('personal_kb_selected_tenant_id')
  if (token) config.headers.Authorization = `Bearer ${token}`
  if (tenant) config.headers['X-Tenant-ID'] = tenant
  config.headers['X-Request-ID'] = Math.random().toString(36).slice(2)
  return config
})

function persistAuth(data: any) {
  if (!data?.token) return
  localStorage.setItem('personal_kb_token', data.token)
  if (data.refresh_token) localStorage.setItem('personal_kb_refresh_token', data.refresh_token)
  if (data.user) localStorage.setItem('personal_kb_user', JSON.stringify(data.user))
  if (data.tenant) {
    localStorage.setItem('personal_kb_tenant', JSON.stringify(data.tenant))
    localStorage.setItem('personal_kb_selected_tenant_id', String(data.tenant.id || ''))
  }
}

function clearAuth() {
  ;[
    'personal_kb_user',
    'personal_kb_tenant',
    'personal_kb_token',
    'personal_kb_selected_tenant_id',
    'personal_kb_refresh_token',
  ].forEach((key) => localStorage.removeItem(key))
}

let refreshing: Promise<any> | null = null

// 并发的 401 共用同一次刷新：refresh token 用一次就作废
function refreshTokens() {
  if (!refreshing) {
    const refreshToken = localStorage.getItem('personal_kb_refresh_token')
    const request = refreshToken
      ? axios.post('/api/v1/auth/refresh', { refresh_token: refreshToken }, { headers: { 'Content-Type': 'application/json' } })
      : Promise.reject(new Error('no refresh token'))
    refreshing = request
      .then((response) => {
        const data = response.data?.data || response.data
        persistAuth(data)
        return data
      })
      .finally(() => {
        refreshing = null
      })
  }
  return refreshing
}

client.interceptors.response.use(
  (response) => response.data,
  async (error) => {
    const original = error.config as any
    const status = error.response?.status
    const url = String(original?.url || '')
    if (status === 401 && original && !original._retry && !url.includes('/api/v1/auth/')) {
      original._retry = true
      try {
        const data = await refreshTokens()
        original.headers = original.headers || {}
        original.headers.Authorization = `Bearer ${data.token}`
        if (data.tenant?.id) original.headers['X-Tenant-ID'] = String(data.tenant.id)
        return client(original)
      } catch {
        // 重新加载后由路由守卫决定：本地开发免密进入，否则跳到登录页
        clearAuth()
        if (location.pathname !== '/login') location.reload()
      }
    }
    return Promise.reject(error.response?.data || { message: error.message })
  },
)

export default client
