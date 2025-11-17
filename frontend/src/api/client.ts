export const API_URL = import.meta.env.VITE_BACKEND_URL;
export const ML_URL = import.meta.env.VITE_ML_URL;

export async function apiFetch(path: string, options: RequestInit = {}) {
  const base = API_URL.endsWith("/") ? API_URL : `${API_URL}/`;
  const cleanPath = path.startsWith("/") ? path.slice(1) : path;

  const res = await fetch(`${base}${cleanPath}`, {
    credentials: "include",
    ...options,
  });

  return res;
}

export async function mlFetch(path: string, options: RequestInit = {}) {
  const base = ML_URL.endsWith("/") ? ML_URL : `${ML_URL}/`;
  const cleanPath = path.startsWith("/") ? path.slice(1) : path;

  const res = await fetch(`${base}${cleanPath}`, {
    credentials: "include",
    ...options,
  });

  return res;
}