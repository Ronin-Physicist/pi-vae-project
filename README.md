PI-VAE Project: Bridging SMILES to E-GNN via Semantic Priors

This repository contains the ongoing implementation and experiments for the Physics-Informed VAE (PI-VAE) project. It chronicles the journey from raw SMILES strings to Equivariant Graph Neural Networks (E-GNN), guided by the concept of "Semantic Materials Informatics."

📖 The Story So Far (Context)

If you have been following the Medium series "The One-Person Lab," you know the overarching goal: building a "Tower of Babel" (PI-VAE) that combines deterministic physics with probabilistic AI.
Episode 0 & 0.5: We identified the "Labyrinth of SMILES" — the fundamental difficulty of translating 1D topological strings (SMILES) into 3D geometric structures required for E-GNNs.
The Pivot: We realized that relying solely on RDKit's conformer generation or standard scalar descriptors (like Moment of Inertia) fails to capture the "gestalt" (macroscopic shape intuition) that human chemists use.

🚀 Current Phase (Episode 1): Semantic Labeling

To bridge the gap between 1D SMILES and 3D E-GNN Geometry, we are currently testing a novel pipeline. Instead of just calculating coordinates, we use a Multimodal LLM (Gemini 1.5 Pro) to act as a "Virtual Chemist."
The pipeline in this repository does the following:
Takes a raw SMILES string.
Generates a 2D image via RDKit.
Uses Gemini to extract a Semantic Vector (a continuous L1-normalized soft label like [rod_like: 0.8, y_shaped: 0.1, ...]).

Why do this?
These semantic vectors are intended to serve as a Geometric Prior for the E-GNN. By injecting human-intuitive shape categories, we aim to guide the VAE's latent space to become semantically disentangled and physically meaningful, bypassing the limitations of traditional SMILES manipulation.

🛠 Project Structure

src/: Core logic for semantic extraction and future E-GNN implementation.
notebooks/: Google Colab compatible notebooks for PoC and data validation (Our "Dirty Engineering" playground).
data/: Sample datasets and validated semantic labels.

🧪 Quick Start (Google Colab)

Experience the Semantic Extraction PoC firsthand:
Open In Colab（coming soon）

"In this lab, we do not just write code; we orchestrate intelligence."
