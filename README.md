# AInalogy

[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-blue?style=for-the-badge&logo=github)](https://piquan.github.io/AInalogy/)
[![Deploy to GitHub Pages](https://github.com/piquan/AInalogy/actions/workflows/deploy.yml/badge.svg)](https://github.com/piquan/AInalogy/actions/workflows/deploy.yml)

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

### For Development

1. **Set up Python environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Build embeddings data**:

   ```bash
   venv/bin/python fetch-embedding.py
   ```

   This creates `public/embeddings.bin` and `public/metadata.json` from the pre-built `mistral-embedding.pt`

3. **Install dependencies**:

   ```bash
   npm install
   ```

4. **Run development server**:

   ```bash
   npm run dev
   ```

5. **Build for production**:

   ```bash
   npm run build
   ```

### For Model Checkpoint Generation (Advanced)

The `mistral-embedding.pt` file is pre-built and included in the repository. If you need to regenerate it:

**⚠️ Warning**: This requires ~25GB of RAM and a Hugging Face token.

```bash
# Remove the existing checkpoint to force regeneration
rm mistral-embedding.pt

# Set your Hugging Face token
export HF_TOKEN=your_token_here

# This will download Mistral-7B and extract embeddings
python fetch-embedding.py
```

## GitHub Pages Deployment

This project includes a GitHub Actions workflow for automatic deployment to GitHub Pages.

**Setup:**

1. Enable GitHub Pages in Settings > Pages, set source to "GitHub Actions"
2. Push to main branch to trigger deployment

The workflow will automatically:

- Use the pre-built model checkpoint (`mistral-embedding.pt`)
- Generate embeddings from the checkpoint
- Build and deploy the React app

**Note**: No Hugging Face token is required for deployment since the model checkpoint is included in the repository.

## Examples

Try these classic analogies:

- man : woman :: king : ? → queen
- Paris : France :: London : ? → England  
- good : better :: bad : ? → worse
- cat : cats :: dog : ? → dogs

## Technical Details

- Uses Mistral-7B embeddings (4096-dimensional vectors)
- Pre-built model checkpoint (`mistral-embedding.pt`) included in repository (~500MB)
- Embeddings are stored as binary Float32 data for efficient loading
- Cosine similarity for finding closest matches
- React with React Router Data API
- Styled with Tailwind CSS and daisyUI
