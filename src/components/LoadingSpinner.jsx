import React from 'react'

const LoadingSpinner = () => {
  return (
    <div className="text-center">
      <div className="loading loading-spinner loading-lg text-primary mb-4"></div>
      <div className="text-xl font-semibold mb-2">Loading Embeddings</div>
      <div className="text-base-content/60">
        This may take a moment...
      </div>
      <div className="text-sm text-base-content/40 mt-2">
        Downloading and parsing Mistral-7B word embeddings
      </div>
    </div>
  )
}

export default LoadingSpinner
