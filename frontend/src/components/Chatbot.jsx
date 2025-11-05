import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { role: 'bot', content: 'Hi! Paste email text to analyze, or ask "How does this work?"' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const bottomRef = useRef(null)
  const sidRef = useRef(`${Date.now()}-${Math.random().toString(36).slice(2,8)}`)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async (text) => {
    if (!text?.trim()) return
    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    try {
      const { data } = await axios.post(`${API}/chat`, { message: text, sessionId: sidRef.current })
      const botMsg = formatBotReply(data)
      const msgs = [botMsg]
      if (data?.llmMessage) {
        msgs.push({ role: 'bot', content: data.llmMessage })
      }
      setMessages(prev => [...prev, ...msgs])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', content: err?.response?.data?.detail || err.message }])
    } finally {
      setLoading(false)
    }
  }

  const onSubmit = (e) => {
    e.preventDefault()
    sendMessage(input)
  }

  const clearChat = () => {
    setMessages([{ role: 'bot', content: 'Chat cleared. How can I help you analyze emails?' }])
  }

  const onUploadTxt = async (file) => {
    if (!file) return
    const allowed = ['text/plain', 'message/rfc822']
    const isAllowedExt = /\.(txt|eml)$/i.test(file.name)
    const isAllowedType = allowed.includes(file.type) || isAllowedExt
    if (!isAllowedType) {
      setMessages(prev => [...prev, { role: 'bot', content: 'Please upload a .txt or .eml file.' }])
      return
    }
    const maxBytes = 1.5 * 1024 * 1024
    if (file.size > maxBytes) {
      setMessages(prev => [...prev, { role: 'bot', content: 'File too large. Please keep under 1.5 MB.' }])
      return
    }
    setUploading(true)
    try {
      const text = await file.text()
      const trimmed = text.slice(0, 12000)
      await sendMessage(trimmed)
    } catch (e) {
      setMessages(prev => [...prev, { role: 'bot', content: e.message }])
    } finally {
      setUploading(false)
    }
  }

  const formatBotReply = (res) => {
    if (res?.type === 'prediction') {
      const prob = (res.probability * 100).toFixed(2)
      const head = res.isSpam ? `⚠️ This email looks like spam (${prob}%).` : `✅ This email looks safe (${prob}%).`
      const reasons = res.reason ? `\nReasons: ${res.reason}` : ''
      const advice = res.advice ? `\nAdvice: ${res.advice}` : ''
      const tokens = Array.isArray(res.topTokens) && res.topTokens.length
        ? `\nTop tokens: ${res.topTokens.map(t => t.token).slice(0,5).join(', ')}`
        : ''
      return { role: 'bot', content: head + reasons + tokens + advice }
    }
    if (res?.type === 'examples') {
      const items = Array.isArray(res.items) ? res.items : []
      const lines = items.map(it => `- [${it.label}] ${it.text}`).join('\n')
      const head = res.message ? `${res.message}\n` : ''
      return { role: 'bot', content: head + lines }
    }
    if (res?.type === 'gmail_summary') {
      const lines = (res.items || []).map(it => `- "${it.subject}" → ${it.isSpam ? 'Spam' : 'Not Spam'} (${(it.probability*100).toFixed(0)}%)`).join('\n')
      return { role: 'bot', content: `${res.message || 'Summary:'}\n${lines}` }
    }
    if (res?.type === 'info' || res?.type === 'error') {
      return { role: 'bot', content: res.message || '' }
    }
    return { role: 'bot', content: 'Sorry, I could not understand the response.' }
  }

  return (
    <div>
      <button
        onClick={() => setOpen(v => !v)}
        className="fixed bottom-6 right-6 z-40 rounded-full bg-white border shadow-lg hover:shadow-xl"
        aria-label={open ? 'Close chat' : 'Open chat'}
      >
        {open ? (
          <div className="h-12 w-12 flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-6 w-6 text-gray-700">
              <path fillRule="evenodd" d="M6.225 4.811a1 1 0 011.414 0L12 9.172l4.361-4.361a1 1 0 111.414 1.414L13.414 10.586l4.361 4.361a1 1 0 01-1.414 1.414L12 12l-4.361 4.361a1 1 0 01-1.414-1.414l4.361-4.361-4.361-4.361a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </div>
        ) : (
          <img src="/logo.svg" alt="Open chat" className="h-12 w-12" />
        )}
      </button>

      {open && (
        <div className="fixed bottom-24 right-6 z-30 w-96 max-w-[95vw] bg-white rounded-xl shadow-2xl border flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b bg-gradient-to-r from-blue-50 to-white flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <img src="/logo.svg" alt="Logo" className="h-7 w-7 rounded" />
              <div>
                <h3 className="font-semibold leading-tight">Spam Assistant</h3>
                <p className="text-xs text-gray-500 leading-tight">Analyze text, explanations, Gmail summary.</p>
              </div>
            </div>
            <button
              onClick={clearChat}
              title="Clear chat"
              className="text-xs px-2 py-1 border rounded hover:bg-gray-100"
            >
              Clear
            </button>
          </div>

          <div className="p-3 h-80 overflow-y-auto space-y-3">
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'} px-3 py-2 rounded-lg max-w-[80%] whitespace-pre-wrap`}>{m.content}</div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-600 px-3 py-2 rounded-lg inline-flex items-center gap-2">
                  <span className="animate-pulse inline-block h-2 w-2 bg-gray-400 rounded-full" />
                  <span className="animate-pulse inline-block h-2 w-2 bg-gray-400 rounded-full" />
                  <span className="animate-pulse inline-block h-2 w-2 bg-gray-400 rounded-full" />
                </div>
              </div>
            )}
          </div>

          <div className="px-3 pb-2 flex flex-wrap gap-2 border-t bg-white">
            <QuickButton onClick={() => sendMessage('How does this work?')}>How does this work?</QuickButton>
            <QuickButton onClick={() => sendMessage('Show example spam.')}>Show example spam.</QuickButton>
            <QuickButton onClick={() => sendMessage('Show ham examples.')}>Show ham examples.</QuickButton>
            <QuickButton onClick={() => sendMessage('Give me anti-spam tips.')}>Anti-spam tips</QuickButton>
            <QuickButton onClick={() => sendMessage('Analyze my last 5 Gmail emails')}>Analyze Gmail</QuickButton>
          </div>

          <form
            onSubmit={onSubmit}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files?.[0]; if (f) onUploadTxt(f) }}
            className={`p-3 flex items-center gap-2 ${dragging ? 'ring-2 ring-blue-400 bg-blue-50' : ''}`}
          >
            <input
              className="flex-1 border rounded px-3 py-2 focus:outline-none focus:ring"
              placeholder="Type a message or paste email text..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading || uploading}
            />
            <label className={`inline-flex items-center gap-2 cursor-pointer text-sm px-3 py-2 border rounded hover:bg-gray-50 ${uploading ? 'opacity-60 pointer-events-none' : ''}`}>
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4 text-blue-600">
                <path d="M12 3a1 1 0 0 1 1 1v7.586l2.293-2.293a1 1 0 1 1 1.414 1.414l-4 4a1 1 0 0 1-1.414 0l-4-4A1 1 0 1 1 8.707 9.293L11 11.586V4a1 1 0 0 1 1-1z"/>
                <path d="M5 15a1 1 0 0 1 1 1v2h12v-2a1 1 0 1 1 2 0v3a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-3a1 1 0 0 1 2 0v2h2v-2a1 1 0 0 1 1-1h8z"/>
              </svg>
              <span className="text-blue-700">Send file</span>
              <input type="file" accept=".txt,.eml" className="hidden" onChange={(e) => e.target.files?.[0] && onUploadTxt(e.target.files[0])} />
            </label>
            <button
              type="submit"
              className="px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              disabled={loading || uploading || !input.trim()}
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </form>
        </div>
      )}
    </div>
  )
}

function QuickButton({ onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="text-xs px-2 py-1 bg-gray-100 rounded hover:bg-gray-200 border"
    >
      {children}
    </button>
  )
}
