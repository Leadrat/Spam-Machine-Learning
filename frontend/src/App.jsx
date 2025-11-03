import React, { useState } from 'react'
import EmailForm from './components/EmailForm.jsx'
import ResultCard from './components/ResultCard.jsx'

export default function App() {
  const [result, setResult] = useState(null)

  return (
    <div className="container mx-auto p-4 max-w-3xl">
      <header className="py-6 text-center">
        <h1 className="text-3xl font-bold">Spam Email Detector</h1>
        <p className="text-gray-600 mt-2">Classify email text as Spam or Ham using an LSTM model</p>
      </header>

      <div className="bg-white rounded-lg shadow p-6">
        <EmailForm onResult={setResult} />
      </div>

      {result && (
        <div className="mt-6">
          <ResultCard prediction={result.prediction} probability={result.probability} />
        </div>
      )}

      <footer className="text-center text-sm text-gray-500 mt-10">
        FastAPI backend at <code>http://localhost:8000</code>
      </footer>
    </div>
  )
}
