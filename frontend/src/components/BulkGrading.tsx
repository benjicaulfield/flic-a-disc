import { useState } from 'react'
import { apiFetch } from '../api/client'

type BulkGradingForm = {
  artist: string
  label: string
  genre: string
  style: string
  wanted: boolean
}

type PreviewResult = {
  conditions: string[]
  matched: number
} | null

export const BulkGrading = () => {
  const [form, setForm] = useState<BulkGradingForm>({
    artist: '',
    label: '',
    genre: '',
    style: '',
    wanted: false,
  })
  const [preview, setPreview] = useState<PreviewResult>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<number | null>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setForm(prev => ({ ...prev, [name]: value }))
  }

  const handlePreview = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setPreview(null)
    setResult(null)
    setLoading(true)
    try {
      const res = await apiFetch('api/catalog/bulk-grading', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, dry_run: true }),
      })
      if (!res.ok) throw new Error('Preview failed')
      const data = await res.json()
      setPreview(data)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  const handleApply = async () => {
    setLoading(true)
    try {
      const res = await apiFetch('api/catalog/bulk-grading', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, dry_run: false }),
      })
      if (!res.ok) throw new Error('Apply failed')
      const data = await res.json()
      setResult(data.affected)
      setPreview(null)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handlePreview} className="flex flex-col gap-3 p-4 max-w-md">
      {(['artist', 'label', 'genre', 'style'] as const).map(field => (
        <input
          key={field}
          name={field}
          value={form[field] as string}
          onChange={handleChange}
          placeholder={field.charAt(0).toUpperCase() + field.slice(1)}
          className="border border-gray-300 rounded px-3 py-2 text-sm"
        />
      ))}
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={form.wanted}
          onChange={e => setForm(prev => ({ ...prev, wanted: e.target.checked }))}
        />
        Wanted
      </label>
      <button type="submit" disabled={loading} className="px-4 py-2 bg-slate-700 text-white text-sm font-semibold rounded">
        Preview
      </button>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      {preview && (
        <div className="text-sm">
          <p>{preview.matched} records will be marked <strong>{form.wanted ? 'wanted' : 'unwanted'}</strong>.</p>
          <button
            type="button"
            onClick={handleApply}
            disabled={loading}
            className="mt-2 px-4 py-2 bg-black text-white text-sm font-semibold rounded"
          >
            Confirm
          </button>
        </div>
      )}
      {result !== null && (
        <p className="text-sm text-green-700 font-medium">{result} records updated.</p>
      )}
    </form>
  )
}
