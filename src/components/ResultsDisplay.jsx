import { embeddingEngine } from '../embeddings'
import LoadingSpinner from './LoadingSpinner'
import PropTypes from 'prop-types'

const ResultsDisplay = ({ result, isEmbeddingsLoading = false }) => {
  if (!result) {
    if (isEmbeddingsLoading) {
      return (
        <div className="card bg-base-200 shadow-lg">
          <div className="card-body">
            <h2 className="card-title mb-4">Results</h2>
            <div className="text-center py-8">
              <LoadingSpinner />
              <div className="text-base-content/50 mt-4">
                Loading embeddings for analogy computation...
              </div>
            </div>
          </div>
        </div>
      )
    }
    
    return (
      <div className="card bg-base-200 shadow-lg">
        <div className="card-body">
          <h2 className="card-title mb-4">Results</h2>
          <div className="text-center text-base-content/50 py-8">
            Enter an analogy to see results
          </div>
        </div>
      </div>
    )
  }

  if (result.error) {
    return (
      <div className="card bg-base-200 shadow-lg">
        <div className="card-body">
          <h2 className="card-title mb-4">Results</h2>
          <div className="alert alert-error">
            <div>
              <h3 className="font-bold">Error</h3>
              <div className="text-xs">{result.error}</div>
              {result.missingTokens && (
                <div className="text-xs mt-2">
                  Missing tokens: {result.missingTokens.join(', ')}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  const { input, results } = result
  const topResult = results[0]

  return (
    <div className="card bg-base-200 shadow-lg">
      <div className="card-body">
        <h2 className="card-title mb-4">Results</h2>
        
        {/* Query display */}
        <div className="mb-6">
          <div className="text-sm text-base-content/60 mb-2">Query:</div>
          <div className="bg-base-300 rounded-lg p-3 font-mono text-sm">
            {embeddingEngine.cleanToken(input.tokenA)} : {embeddingEngine.cleanToken(input.tokenB)} :: {embeddingEngine.cleanToken(input.tokenC)} : ?
          </div>
        </div>

        {/* Top result */}
        {topResult && (
          <div className="mb-6">
            <div className="text-sm text-base-content/60 mb-2">Best answer:</div>
            <div className="bg-primary text-primary-content rounded-lg p-4">
              <div className="text-2xl font-bold mb-2">
                {embeddingEngine.cleanToken(topResult.token)}
              </div>
              <div className="text-sm opacity-75">
                Similarity: {(topResult.similarity * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}

        {/* All results */}
        <div>
          <div className="text-sm text-base-content/60 mb-2">Top 10 candidates:</div>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {results.map((resultItem, index) => (
              <div
                key={resultItem.token}
                className={`flex justify-between items-center p-3 rounded-lg ${
                  index === 0 
                    ? 'bg-primary text-primary-content' 
                    : 'bg-base-300'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="text-xs font-mono w-6">
                    #{index + 1}
                  </div>
                  <div className="font-medium">
                    {embeddingEngine.cleanToken(resultItem.token)}
                  </div>
                </div>
                <div className="text-sm opacity-75 font-mono">
                  {(resultItem.similarity * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Mathematical explanation */}
        <div className="mt-6 text-xs text-base-content/50">
          <div className="bg-base-300 rounded p-3">
            <div className="font-mono">
              result = embedding({embeddingEngine.cleanToken(input.tokenC)}) + 
              (embedding({embeddingEngine.cleanToken(input.tokenB)}) - 
              embedding({embeddingEngine.cleanToken(input.tokenA)}))
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

ResultsDisplay.propTypes = {
  result: PropTypes.shape({
    error: PropTypes.string,
    missingTokens: PropTypes.arrayOf(PropTypes.string),
    input: PropTypes.shape({
      tokenA: PropTypes.string.isRequired,
      tokenB: PropTypes.string.isRequired,
      tokenC: PropTypes.string.isRequired,
    }),
    results: PropTypes.arrayOf(PropTypes.shape({
      token: PropTypes.string.isRequired,
      similarity: PropTypes.number.isRequired,
    })),
  }),
  isEmbeddingsLoading: PropTypes.bool,
}

export default ResultsDisplay
