# AInalogy

A simple webapp to demonstrate word embedding vector arithmetic using Mistral-7B embeddings.

## Features

- Interactive word analogy form ("A is to B as C is to ?")
- Real-time autocomplete with vocabulary suggestions
- Top 10 most similar results with similarity scores
- Clean, modern UI built with React and daisyUI
- Entirely client-side - works on GitHub Pages

## How it works

The app performs vector arithmetic on word embeddings:
1. Takes three input words (A, B, C)
2. Computes: `embedding(C) + (embedding(B) - embedding(A))`
3. Finds the vocabulary word with the highest cosine similarity to this result

## Setup

1. **Generate embeddings** (requires access to Mistral-7B model):
   ```bash
   python fetch-embedding.py
   ```
   This creates `public/embeddings.bin` and `public/metadata.json`

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```

4. **Build for production**:
   ```bash
   npm run build
   ```

## Examples

Try these classic analogies:
- man : woman :: king : ? → queen
- Paris : France :: London : ? → England  
- good : better :: bad : ? → worse
- cat : cats :: dog : ? → dogs

## Technical Details

- Uses Mistral-7B embeddings (4096-dimensional vectors)
- Embeddings are stored as binary Float32 data for efficient loading
- Cosine similarity for finding closest matches
- React with React Router Data API
- Styled with Tailwind CSS and daisyUI
