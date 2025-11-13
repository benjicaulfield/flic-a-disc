export const API_URL = import.meta.env.VITE_BACKEND_URL;
export const ML_URL = import.meta.env.VITE_ML_URL;

export async function apiFetch(path: string, options: RequestInit = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    credentials: "include",
    ...options,
  });
  return res;
}

export async function mlFetch(path: string, options: RequestInit = {}) {
  const res = await fetch(`${ML_URL}${path}`, {
    credentials: "include",
    ...options,
  });
  return res;
}
