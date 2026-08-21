export const joinList = (v: unknown) => (Array.isArray(v) ? v.join(', ') : '') || 'N/A';
export const money = (v: number | string | null | undefined) => {
  if (v == null || v === '') return 'N/A';
  const n = typeof v === 'string' ? parseFloat(v) : v;
  return isNaN(n) ? 'N/A' : `$${n.toFixed(2)}`;
}