// ============================================================
// config.js — API URL config
// Automatically uses the right URL for local vs production
// ============================================================

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default API;