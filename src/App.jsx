import { useState, useEffect } from 'react'
import { embeddingEngine } from './embeddings'
import AnalogyForm from './components/AnalogyForm'
import ResultsDisplay from './components/ResultsDisplay'

function App() {
  const [embeddingsLoading, setEmbeddingsLoading] = useState(true)
  const [loadError, setLoadError] = useState(null)
  const [analogyResult, setAnalogyResult] = useState(null)

  useEffect(() => {
    const loadMetadata = async () => {
      try {
        await embeddingEngine.loadMetadata()
      } catch (error) {
        setLoadError(error.message)
      }
    }

    const loadEmbeddings = async () => {
      try {
        await embeddingEngine.loadEmbeddings()
        setEmbeddingsLoading(false)
      } catch (error) {
        setLoadError(error.message)
        setEmbeddingsLoading(false)
      }
    }

    // Load metadata first for immediate autocompletion
    loadMetadata()
    // Load embeddings in background for analogy computation
    loadEmbeddings()
  }, [])

  const handleAnalogySubmit = async ({ tokenA, tokenB, tokenC }) => {
    if (!tokenA || !tokenB || !tokenC) return

    // Check if embeddings are loaded for analogy computation
    if (embeddingsLoading) {
      setAnalogyResult({
        input: { tokenA, tokenB, tokenC },
        error: "Embeddings are still loading. Please wait a moment and try again."
      })
      return
    }

    try {
      const result = embeddingEngine.performAnalogy(tokenA, tokenB, tokenC)
      setAnalogyResult({
        input: { tokenA, tokenB, tokenC },
        ...result
      })
    } catch (error) {
      setAnalogyResult({
        input: { tokenA, tokenB, tokenC },
        error: error.message
      })
    }
  }

  // Show error state if there's a critical loading error
  if (loadError) {
    return (
      <div className="min-h-screen bg-base-100 flex items-center justify-center">
        <div className="text-center">
          <div className="text-error text-xl mb-4">Failed to load embeddings</div>
          <div className="text-base-content/70">{loadError}</div>
          <div className="mt-4 text-sm text-base-content/50">
            Make sure you&apos;ve run the Python script to generate the embedding files.
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-base-100">
      {/* Header */}
      <div className="navbar bg-primary text-primary-content">
        <div className="flex-1">
          <h1 className="text-xl font-bold">AInalogy</h1>
        </div>
        <div className="flex-none">
          <div className="text-sm opacity-70">
            Word Embedding Analogies
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Description */}
        <div className="mb-8">
          <div className="bg-base-200 rounded-lg p-6">
            <h2 className="text-2xl font-semibold mb-4">How it works</h2>
            <p className="text-base-content/80 mb-4">
              This demo shows how word embeddings capture semantic relationships. 
              Fill in the analogy form below with the pattern &quot;A is to B as C is to ?&quot;
            </p>
            <p className="text-base-content/80 mb-4">
              For example: <span className="font-mono bg-base-300 px-2 py-1 rounded">man : woman :: king : ?</span>
            </p>
            <p className="text-base-content/60 text-sm">
              The algorithm computes: embedding(king) + (embedding(woman) - embedding(man)) 
              and finds the closest word in the vocabulary.
            </p>
          </div>
        </div>

        {/* Main Form */}
        <div className="grid gap-8 lg:grid-cols-2">
          <div>
            <AnalogyForm onSubmit={handleAnalogySubmit} />
          </div>
          
          <div>
            <ResultsDisplay 
              result={analogyResult} 
              isEmbeddingsLoading={embeddingsLoading}
            />
          </div>
        </div>

        {/* Footer */}
        <div className="mt-16 text-center text-base-content/50 text-sm">
          <p>
            Using embeddings from Mistral-7B • Built with React and Tailwind CSS
          </p>
        </div>
      </div>
    </div>
  )
}

export default App
