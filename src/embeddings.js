// Utility functions for working with embeddings

// BF16 to Float32 conversion utilities
function bf16ToFloat32(bf16Uint16) {
  // BF16 is stored as uint16, but represents the top 16 bits of a float32
  // To convert back: shift left by 16 bits to get a valid float32
  const uint32 = new Uint32Array(bf16Uint16.length);
  for (let i = 0; i < bf16Uint16.length; i++) {
    uint32[i] = bf16Uint16[i] << 16;
  }
  return new Float32Array(uint32.buffer);
}

export class EmbeddingEngine {
  constructor() {
    this.embeddings = null;
    this.vocab = null;
    this.reverseVocab = null;
    this.metadata = null;
    this.isLoaded = false;
    this.loadingPromise = null; // Cache the loading promise
  }

  async loadEmbeddings() {
    // If already loaded, return immediately
    if (this.isLoaded) {
      console.log('Embeddings already loaded, skipping duplicate load');
      return;
    }
    
    // If currently loading, return the existing promise
    if (this.loadingPromise) {
      console.log('Embeddings already loading, waiting for existing promise');
      return this.loadingPromise;
    }

    // Start loading and cache the promise
    console.log('Starting embeddings load...');
    this.loadingPromise = this._doLoadEmbeddings();
    
    try {
      await this.loadingPromise;
    } finally {
      // Clear the promise when done (success or failure)
      this.loadingPromise = null;
    }
  }

  async _doLoadEmbeddings() {
    try {
      // Load metadata (use relative path to work with Vite's base configuration)
      const metadataResponse = await fetch('./metadata.json');
      this.metadata = await metadataResponse.json();
      this.vocab = this.metadata.vocab;
      
      // Create reverse vocab lookup
      this.reverseVocab = Object.fromEntries(
        Object.entries(this.vocab).map(([token, id]) => [id, token])
      );

      // Load embeddings binary data
      const embeddingsResponse = await fetch('./embeddings.bin');
      const arrayBuffer = await embeddingsResponse.arrayBuffer();
      
      // Check data type from metadata
      let typedData;
      if (this.metadata.dtype === 'bf16') {
        // Load as uint16 (BF16 format) and convert to float32
        const uint16Data = new Uint16Array(arrayBuffer);
        typedData = bf16ToFloat32(uint16Data);
        console.log('Loaded BF16 embeddings, converted to float32');
      } else {
        // Default: load as float32
        typedData = new Float32Array(arrayBuffer);
        console.log('Loaded float32 embeddings');
      }
      
      // Reshape into 2D array [vocab_size, embedding_dim]
      const vocabSize = this.metadata.vocab_size;
      const embeddingDim = this.metadata.embedding_dim;
      
      this.embeddings = [];
      for (let i = 0; i < vocabSize; i++) {
        const start = i * embeddingDim;
        const end = start + embeddingDim;
        this.embeddings.push(typedData.slice(start, end));
      }

      this.isLoaded = true;
      console.log(`Loaded ${vocabSize} embeddings of dimension ${embeddingDim} (${this.metadata.dtype})`);
    } catch (error) {
      console.error('Failed to load embeddings:', error);
      throw error;
    }
  }

  // Get embedding vector for a token
  getEmbedding(token) {
    if (!this.isLoaded) {
      throw new Error('Embeddings not loaded yet');
    }

    const tokenId = this.vocab[token];
    if (tokenId === undefined) {
      return null;
    }

    return this.embeddings[tokenId];
  }

  // Compute cosine similarity between two vectors
  cosineSimilarity(a, b) {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same length');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (normA * normB);
  }

  // Add two vectors
  addVectors(a, b) {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same length');
    }

    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      result[i] = a[i] + b[i];
    }
    return result;
  }

  // Subtract two vectors
  subtractVectors(a, b) {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same length');
    }

    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      result[i] = a[i] - b[i];
    }
    return result;
  }

  // Perform word analogy: a is to b as c is to ?
  // Returns the token most similar to (c + (b - a))
  performAnalogy(tokenA, tokenB, tokenC, topK = 10) {
    const embA = this.getEmbedding(tokenA);
    const embB = this.getEmbedding(tokenB);
    const embC = this.getEmbedding(tokenC);

    if (!embA || !embB || !embC) {
      return {
        error: 'One or more tokens not found in vocabulary',
        missingTokens: [
          !embA ? tokenA : null,
          !embB ? tokenB : null,
          !embC ? tokenC : null
        ].filter(Boolean)
      };
    }

    // Compute target vector: C + (B - A)
    const diff = this.subtractVectors(embB, embA);
    const target = this.addVectors(embC, diff);

    // Find most similar tokens
    const similarities = [];
    const inputTokens = new Set([tokenA, tokenB, tokenC]);
    
    // Also exclude case variants and cleaned versions of input tokens
    const inputVariants = new Set();
    const spaceString = this.metadata.space_string;
    
    for (const token of [tokenA, tokenB, tokenC]) {
      const cleaned = this.cleanToken(token);
      inputVariants.add(token);
      inputVariants.add(cleaned);
      inputVariants.add(cleaned.toLowerCase());
      inputVariants.add(cleaned.toUpperCase());
      inputVariants.add(cleaned.charAt(0).toUpperCase() + cleaned.slice(1).toLowerCase());
      
      // Add versions with and without space prefix
      inputVariants.add(spaceString + cleaned);
      inputVariants.add(spaceString + cleaned.toLowerCase());
      inputVariants.add(spaceString + cleaned.toUpperCase());
      inputVariants.add(spaceString + cleaned.charAt(0).toUpperCase() + cleaned.slice(1).toLowerCase());
    }

    for (const [token, id] of Object.entries(this.vocab)) {
      // Skip input tokens and their variants
      const cleanedToken = this.cleanToken(token);
      if (inputVariants.has(token) || inputVariants.has(cleanedToken) || 
          inputVariants.has(cleanedToken.toLowerCase()) ||
          inputVariants.has(cleanedToken.toUpperCase()) ||
          inputVariants.has(cleanedToken.charAt(0).toUpperCase() + cleanedToken.slice(1).toLowerCase())) {
        continue;
      }

      const embedding = this.embeddings[id];
      const similarity = this.cosineSimilarity(target, embedding);
      similarities.push({ token, similarity });
    }

    // Sort by similarity and return top K
    similarities.sort((a, b) => b.similarity - a.similarity);
    
    return {
      results: similarities.slice(0, topK),
      targetVector: target
    };
  }

  // Get all tokens that start with a prefix (for autocomplete)
  getTokensWithPrefix(prefix, maxResults = 50) {
    if (!this.isLoaded || !prefix) return [];

    const normalizedPrefix = prefix.toLowerCase();
    const spaceString = this.metadata.space_string;
    
    const matches = new Map(); // Use Map to avoid duplicates based on display text
    
    for (const token of Object.keys(this.vocab)) {
      const cleanToken = token.replace(spaceString, '');
      const cleanLower = cleanToken.toLowerCase();
      
      // Try matching both with and without space prefix
      if (cleanLower.startsWith(normalizedPrefix) ||
          token.toLowerCase().startsWith(normalizedPrefix) ||
          token.toLowerCase().startsWith(spaceString + normalizedPrefix)) {
        
        // Use cleanToken as key to avoid duplicates, but prefer exact case match
        const key = cleanLower;
        
        if (!matches.has(key)) {
          matches.set(key, {
            token,
            display: cleanToken || token,
            priority: this.getMatchPriority(cleanToken, prefix, normalizedPrefix, token)
          });
        } else {
          // If we already have this word, prefer exact case match
          const existing = matches.get(key);
          const newPriority = this.getMatchPriority(cleanToken, prefix, normalizedPrefix, token);
          if (newPriority < existing.priority) {
            matches.set(key, {
              token,
              display: cleanToken || token,
              priority: newPriority
            });
          }
        }
      }
    }

    return Array.from(matches.values())
      .sort((a, b) => {
        // Sort by priority first, then by length, then alphabetically
        if (a.priority !== b.priority) return a.priority - b.priority;
        if (a.display.length !== b.display.length) return a.display.length - b.display.length;
        return a.display.localeCompare(b.display);
      })
      .slice(0, maxResults);
  }

  // Helper function to determine match priority
  getMatchPriority(cleanToken, originalPrefix, normalizedPrefix, originalToken) {
    // Prefer space-prefixed tokens (standalone words)
    const hasSpacePrefix = originalToken.startsWith(this.metadata.space_string);
    
    // Exact case match gets highest priority
    if (cleanToken === originalPrefix) {
      return hasSpacePrefix ? 0 : 1;
    }
    
    // Exact case match but longer
    if (cleanToken.startsWith(originalPrefix)) {
      return hasSpacePrefix ? 2 : 3;
    }
    
    // Case-insensitive exact match
    if (cleanToken.toLowerCase() === normalizedPrefix) {
      return hasSpacePrefix ? 4 : 5;
    }
    
    // Case-insensitive prefix match
    if (cleanToken.toLowerCase().startsWith(normalizedPrefix)) {
      return hasSpacePrefix ? 6 : 7;
    }
    
    // Everything else
    return hasSpacePrefix ? 8 : 9;
  }

  // Clean token for display (remove space markers)
  cleanToken(token) {
    if (!this.metadata) return token;
    return token.replace(this.metadata.space_string, '');
  }
}

// Create a singleton instance
export const embeddingEngine = new EmbeddingEngine();
