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
    this.normalizedEmbeddings = null; // Pre-normalized for fast cosine similarity
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

      // Pre-normalize all embeddings for fast dot-product similarity
      console.log('Normalizing embeddings for fast similarity calculation...');
      this.normalizedEmbeddings = [];
      for (let i = 0; i < vocabSize; i++) {
        this.normalizedEmbeddings.push(this.normalizeVector(this.embeddings[i]));
      }

      this.isLoaded = true;
      console.log(`Loaded ${vocabSize} embeddings of dimension ${embeddingDim} (${this.metadata.dtype})`);
      console.log('Pre-normalized embeddings for fast similarity calculation');
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

    // First try direct lookup
    let tokenId = this.vocab[token];
    
    // If not found, try to find the best match
    if (tokenId === undefined) {
      const bestMatch = this.findBestTokenMatch(token);
      if (bestMatch) {
        tokenId = this.vocab[bestMatch];
      }
    }
    
    if (tokenId === undefined) {
      return null;
    }

    return this.embeddings[tokenId];
  }

  // Normalize a vector to unit length
  normalizeVector(vector) {
    let norm = 0;
    for (let i = 0; i < vector.length; i++) {
      norm += vector[i] * vector[i];
    }
    norm = Math.sqrt(norm);
    
    if (norm === 0) return vector; // Avoid division by zero
    
    const normalized = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
      normalized[i] = vector[i] / norm;
    }
    return normalized;
  }

  // Compute dot product (for normalized vectors, this equals cosine similarity)
  dotProduct(a, b) {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same length');
    }

    let result = 0;
    for (let i = 0; i < a.length; i++) {
      result += a[i] * b[i];
    }
    return result;
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

    // Compute target vector: C + (B - A), then normalize it
    const diff = this.subtractVectors(embB, embA);
    const target = this.addVectors(embC, diff);
    const normalizedTarget = this.normalizeVector(target);

    // Generate all variants of input tokens to exclude from results
    const inputVariants = new Set();
    for (const token of [tokenA, tokenB, tokenC]) {
      const variants = this.generateTokenVariants(token);
      for (const variant of variants) {
        inputVariants.add(variant);
      }
    }

    // Find most similar tokens using fast dot product with normalized embeddings
    const similarities = [];
    for (const [token, id] of Object.entries(this.vocab)) {
      // Skip input tokens and their variants
      if (inputVariants.has(token)) {
        continue;
      }

      const normalizedEmbedding = this.normalizedEmbeddings[id];
      const similarity = this.dotProduct(normalizedTarget, normalizedEmbedding);
      similarities.push({ token, similarity });
    }

    // Sort by similarity and return top K
    similarities.sort((a, b) => b.similarity - a.similarity);
    
    return {
      results: similarities.slice(0, topK),
      targetVector: normalizedTarget
    };
  }

  // Get all tokens that start with a prefix (for autocomplete)
  getTokensWithPrefix(prefix, maxResults = 50) {
    if (!this.isLoaded || !prefix) return [];

    const normalizedPrefix = prefix.toLowerCase();
    const spaceString = this.metadata.space_string;
    const matches = new Map(); // Use Map to avoid duplicates based on display text
    
    for (const token of Object.keys(this.vocab)) {
      const cleanToken = this.cleanToken(token);
      const cleanLower = cleanToken.toLowerCase();
      
      // Check if this token matches our prefix
      const matchesPrefix = cleanLower.startsWith(normalizedPrefix) ||
                           token.toLowerCase().startsWith(normalizedPrefix) ||
                           token.toLowerCase().startsWith(spaceString + normalizedPrefix);
      
      if (matchesPrefix) {
        const key = cleanLower; // Use cleaned token as key to avoid duplicates
        const hasSpacePrefix = token.startsWith(spaceString);
        
        // Calculate priority: prefer space-prefixed tokens and exact matches
        let priority = hasSpacePrefix ? 0 : 10; // Base priority for space-prefixed vs not
        
        // Bonus for exact case match
        if (cleanToken === prefix) priority += 0;
        else if (cleanToken.startsWith(prefix)) priority += 1;
        else if (cleanLower === normalizedPrefix) priority += 2;
        else priority += 3;
        
        if (!matches.has(key) || matches.get(key).priority > priority) {
          matches.set(key, {
            token,
            display: cleanToken || token,
            priority
          });
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

  // Clean token for display (remove space markers)
  cleanToken(token) {
    if (!this.metadata) return token;
    return token.replace(this.metadata.space_string, '');
  }

  // Generate all possible variants of a token for matching
  generateTokenVariants(token) {
    const variants = new Set();
    const spaceString = this.metadata.space_string;
    const cleaned = this.cleanToken(token);
    
    // Add the original token
    variants.add(token);
    
    // Add cleaned version
    variants.add(cleaned);
    
    // Add case variations of cleaned token
    variants.add(cleaned.toLowerCase());
    variants.add(cleaned.toUpperCase());
    variants.add(cleaned.charAt(0).toUpperCase() + cleaned.slice(1).toLowerCase());
    
    // Add versions with space prefix
    variants.add(spaceString + cleaned);
    variants.add(spaceString + cleaned.toLowerCase());
    variants.add(spaceString + cleaned.toUpperCase());
    variants.add(spaceString + cleaned.charAt(0).toUpperCase() + cleaned.slice(1).toLowerCase());
    
    return variants;
  }

  // Find the best token match for a given word, preferring space-prefixed versions
  findBestTokenMatch(word) {
    const spaceString = this.metadata.space_string;
    const candidates = [];
    
    // Look for exact matches first
    const variants = this.generateTokenVariants(word);
    for (const variant of variants) {
      if (this.vocab.hasOwnProperty(variant)) {
        const hasSpacePrefix = variant.startsWith(spaceString);
        candidates.push({
          token: variant,
          priority: hasSpacePrefix ? 0 : 1, // Prefer space-prefixed
          isExact: true
        });
      }
    }
    
    // If we found exact matches, return the best one
    if (candidates.length > 0) {
      candidates.sort((a, b) => a.priority - b.priority);
      return candidates[0].token;
    }
    
    // No exact match found
    return null;
  }
}

// Create a singleton instance
export const embeddingEngine = new EmbeddingEngine();
