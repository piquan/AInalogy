import { useState, useRef } from 'react'
import PropTypes from 'prop-types'
import { embeddingEngine } from '../embeddings'

const TokenInput = ({ value, onChange, placeholder, onFocus, onBlur, disabled = false }) => {
  const [suggestions, setSuggestions] = useState([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const inputRef = useRef(null)

  const handleInputChange = (e) => {
    const newValue = e.target.value
    
    if (newValue.length > 0) {
      const matches = embeddingEngine.getTokensWithPrefix(newValue, 10)
      setSuggestions(matches)
      setShowSuggestions(matches.length > 0)
      setSelectedIndex(-1)
      
      // If user types a word, try to find the best match with space prefix
      if (matches.length > 0 && !newValue.includes('▁')) {
        const exactMatch = matches.find(m => 
          m.display.toLowerCase() === newValue.toLowerCase()
        );
        if (exactMatch) {
          onChange(exactMatch.token);
          return;
        }
      }
    } else {
      setShowSuggestions(false)
    }
    
    onChange(newValue)
  }

  const handleKeyDown = (e) => {
    if (!showSuggestions) return

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setSelectedIndex(prev => 
          prev < suggestions.length - 1 ? prev + 1 : prev
        )
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedIndex(prev => prev > 0 ? prev - 1 : -1)
        break
      case 'Enter':
        e.preventDefault()
        if (selectedIndex >= 0) {
          selectSuggestion(suggestions[selectedIndex])
        }
        break
      case 'Escape':
        setShowSuggestions(false)
        setSelectedIndex(-1)
        break
    }
  }

  const selectSuggestion = (suggestion) => {
    onChange(suggestion.token)
    setShowSuggestions(false)
    setSelectedIndex(-1)
    inputRef.current?.blur()
  }

  const handleFocus = (e) => {
    onFocus?.(e)
    if (value && suggestions.length > 0) {
      setShowSuggestions(true)
    }
  }

  const handleBlur = (e) => {
    // Delay hiding suggestions to allow clicking on them
    setTimeout(() => {
      setShowSuggestions(false)
      setSelectedIndex(-1)
    }, 150)
    onBlur?.(e)
  }

  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="text"
        className="input input-bordered w-full placeholder:text-base-content/40"
        value={embeddingEngine.cleanToken(value) || ''}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onFocus={handleFocus}
        onBlur={handleBlur}
        placeholder={placeholder}
        disabled={disabled}
        autoComplete="off"
      />
      
      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute z-10 w-full mt-1 bg-base-100 border border-base-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
          {suggestions.map((suggestion, index) => (
            <div
              key={suggestion.token}
              className={`px-4 py-2 cursor-pointer hover:bg-base-200 ${
                index === selectedIndex ? 'bg-primary text-primary-content' : ''
              }`}
              onClick={() => selectSuggestion(suggestion)}
            >
              {suggestion.display}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

TokenInput.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
  placeholder: PropTypes.string,
  onFocus: PropTypes.func,
  onBlur: PropTypes.func,
  disabled: PropTypes.bool,
}

const AnalogyForm = ({ onSubmit }) => {
  const [tokenA, setTokenA] = useState('')
  const [tokenB, setTokenB] = useState('')
  const [tokenC, setTokenC] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!tokenA || !tokenB || !tokenC) return

    setIsSubmitting(true)
    try {
      await onSubmit({ tokenA, tokenB, tokenC })
    } finally {
      setIsSubmitting(false)
    }
  }

  const fillExample = (example) => {
    setTokenA(example.a)
    setTokenB(example.b)
    setTokenC(example.c)
  }

  const examples = [
    { a: '▁man', b: '▁woman', c: '▁king', description: 'man : woman :: king : ?' },
    { a: '▁Paris', b: '▁France', c: '▁London', description: 'Paris : France :: London : ?' },
    { a: '▁good', b: '▁better', c: '▁bad', description: 'good : better :: bad : ?' },
    { a: '▁cat', b: '▁cats', c: '▁dog', description: 'cat : cats :: dog : ?' },
  ]

  return (
    <div className="card bg-base-200 shadow-lg">
      <div className="card-body">
        <h2 className="card-title mb-4">Word Analogy</h2>
        
        {/* Examples */}
        <div className="mb-6">
          <div className="text-sm font-medium mb-2">Try these examples:</div>
          <div className="flex flex-wrap gap-2">
            {examples.map((example, index) => (
              <button
                key={index}
                className="btn btn-sm btn-outline"
                onClick={() => fillExample(example)}
                type="button"
              >
                {example.description}
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Analogy inputs */}
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <TokenInput
                value={tokenA}
                onChange={setTokenA}
                placeholder="man"
              />
            </div>
            
            <div className="text-lg font-bold text-base-content/70 px-1">:</div>
            
            <div className="flex-1">
              <TokenInput
                value={tokenB}
                onChange={setTokenB}
                placeholder="woman"
              />
            </div>
            
            <div className="text-lg font-bold text-base-content/70 px-1">::</div>
            
            <div className="flex-1">
              <TokenInput
                value={tokenC}
                onChange={setTokenC}
                placeholder="king"
              />
            </div>
          </div>

          {/* Visual equation */}
          <div className="text-center text-sm text-base-content/60 font-mono">
            {tokenA || 'A'} is to {tokenB || 'B'} as {tokenC || 'C'} is to ?
          </div>

          <button
            type="submit"
            className={`btn btn-primary w-full ${isSubmitting ? 'loading' : ''}`}
            disabled={!tokenA || !tokenB || !tokenC || isSubmitting}
          >
            {isSubmitting ? 'Computing...' : 'Find Answer'}
          </button>
        </form>
      </div>
    </div>
  )
}

AnalogyForm.propTypes = {
  onSubmit: PropTypes.func.isRequired,
}

export default AnalogyForm
