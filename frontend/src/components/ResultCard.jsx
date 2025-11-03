import React from 'react'

export default function ResultCard({ prediction, probability }) {
  const isSpam = prediction?.toLowerCase() === 'spam'
  return (
    <div className={`p-5 rounded-lg shadow border ${isSpam ? 'bg-red-50 border-red-300' : 'bg-green-50 border-green-300'}`}>
      <h2 className="text-xl font-semibold">
        Result: <span className={isSpam ? 'text-red-700' : 'text-green-700'}>{prediction}</span>
      </h2>
      <p className="text-gray-700 mt-2">Probability: {(probability * 100).toFixed(2)}%</p>
      <p className="text-sm text-gray-500 mt-1">(Threshold = 0.5)</p>
    </div>
  )
}
