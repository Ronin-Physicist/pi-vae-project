<div align="center">
<h1>🌌 PI-VAE Project</h1>
<p><b>Bridging SMILES to E-GNN via Semantic Priors</b></p>
</div>
This repository contains the ongoing implementation and experiments for the Physics-Informed VAE (PI-VAE) project. It chronicles the journey from raw SMILES strings to Equivariant Graph Neural Networks (E-GNN), guided by the concept of "Semantic Materials Informatics."<br>
📖 The Story So Far (Context)<br>
If you have been following the Medium series "The One-Person Lab," you know the overarching goal: building a "Tower of Babel" (PI-VAE) that combines deterministic physics with probabilistic AI.<br>
• The Labyrinth of SMILES:<br>
We identified the fundamental difficulty of translating 1D topological strings (SMILES) into 3D geometric structures required for E-GNNs.<br>
• The Pivot: <br>
Relying solely on RDKit's conformer generation or standard scalar descriptors (like Moment of Inertia) fails to capture the macroscopic "gestalt" (shape intuition) that human chemists use.<br>
🚀 Current Phase: Semantic Labeling<br>
To bridge the gap between 1D SMILES and 3D E-GNN Geometry, we are testing a novel pipeline. Instead of just calculating coordinates, we use a Multimodal LLM (Gemini 1.5 Pro) to act as a "Virtual Chemist."<br>
The pipeline does the following:<br>
1.	Takes a raw SMILES string.<br>
2.	Generates a 2D image via RDKit.<br>
3.	Uses Gemini to extract a Semantic 
Vector (a continuous L1-normalized soft label, e.g., [rod_like: 0.8, y_shaped: 0.1, ...]).<br>
Why do this? <br>
These semantic vectors serve as a Geometric Prior for the E-GNN. By injecting human-intuitive shape categories, we aim to guide the VAE's latent space to become semantically disentangled and physically meaningful.<br>
🛠 Project Structure<br>
• src/ : Core logic for semantic extraction and E-GNN implementation.<br>
• notebooks/ : Google Colab compatible notebooks for PoC (Our "Dirty Engineering" playground).<br>
• data/ : Sample datasets and validated semantic labels.<br>
🧪 Quick Start (Colab)<br>
Experience the Semantic Extraction PoC firsthand:<br>
Open In Colab (coming soon)<br><br>
"In this lab, we do not just write code; we orchestrate intelligence."
