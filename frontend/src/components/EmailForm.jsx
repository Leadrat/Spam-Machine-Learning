import React, { useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export default function EmailForm({ onResult }) {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    onResult(null)
    if (!text.trim()) {
      setError('Please enter email text')
      return
    }
    setLoading(true)
    try {
      const { data } = await axios.post(`${API}/predict`, { text })
      onResult(data)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <textarea
        className="w-full h-40 p-3 border rounded focus:outline-none focus:ring"
        placeholder="Paste email content here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex items-center gap-3">
        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          disabled={loading}
        >
          {loading ? 'Detecting...' : 'Detect Spam'}
        </button>
        {error && <span className="text-red-600 text-sm">{error}</span>}
      </div>
    </form>
  )
}
