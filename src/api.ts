const defaultApiBaseUrl = import.meta.env.DEV
  ? 'http://localhost:8080'
  : 'https://trinetra-kuvl.onrender.com';

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL?.trim() || defaultApiBaseUrl
).replace(/\/+$/, '');

export const WS_BASE_URL = API_BASE_URL
  .replace(/^https:/, 'wss:')
  .replace(/^http:/, 'ws:');
