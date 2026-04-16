# ACE Agent (Automated Clustering Expert)

ACE Agent is a production-grade, multi-agent autonomous system designed for advanced clustering analysis. It simulates the end-to-end workflow of a senior data scientist: **Data Profiling → Strategy Routing → Parallel Experimentation → Metric Evaluation → Interpretation → Professional Reporting.**

## 🚀 Key Features

### 1. Plug-and-Play Algorithm Zoo
Consolidated 10+ industry-standard algorithms into a unified registry:
- **Centroid**: KMeans, MiniBatchKMeans, GaussianMixture
- **Density/Topology**: DBSCAN, HDBSCAN, OPTICS, MeanShift
- **Connectivity**: Agglomerative, Spectral, Birch, AffinityPropagation
- Supports **Exhaustive Mode** for parallel brute-force benchmarking.

### 2. Meta-Learning Master Router
An intelligent brain that profiles data using statistical features:
- **Statistical Profiling**: Automatically calculates Skewness, Kurtosis, Sparsity, and Correlation.
- **Rule-Engine + LLM**: Routes tasks to specialized experts (Centroid, Topology, Dimension, Deep, etc.) based on data geometry.
- **Intent Recognition**: Seamlessly distinguishes between a **New Task** and a **Follow-up Inquiry**.

### 3. Rich Dataset Ecosystem
Supports a wide range of benchmarks for rigorous testing:
- **Synthetic**: Blobs, Moons, S-Curve, Smile, High-Dim.
- **Classic**: Iris, Wine, Optdigits.
- **Deep/High-Dim**: MNIST (Image pixels), 20 Newsgroups (Text TF-IDF).
- **Multi-View**: UCI Multiple Features (Consensus clustering benchmark).
- **Custom**: Robust CSV/Excel uploader with automatic imputer and label detection.

### 4. Interactive Follow-up & Interpretation
Beyond raw metrics, ACE Agent provides deep reasoning:
- **Stateful Sessions**: Remembers previous results and conversation history.
- **Explaining "Why"**: Ask "Why is Spectral better than KMeans?" and get an analytical answer based on manifold theory and data metrics without re-running experiments.
- **Stateful Web UI**: Built with Streamlit, supporting continuous chat and interactive parameter tuning.

### 5. Professional Artifacts
- **Automated LaTeX Reports**: Generates academic-grade `.tex` and `.pdf` reports including decision traces, metric rankings, and visualization plots.
- **Secure Sandbox**: All generated code runs in a restricted `CoderSandbox` for safety and reproducibility.

---

## 🛠️ Quick Start

### 1. Environment Setup
```bash
conda create -n ACE_Agent python=3.10
conda activate ACE_Agent
pip install -r requirements.txt
cp .env.example .env  # Configure your LLM API Key here
```

### 2. Launch Interactive Web UI
```bash
streamlit run web_demo.py
```

### 3. CLI Interactive Mode (New!)
Analyze a dataset and enter a continuous reasoning session:
```bash
python demo_runner.py --dataset iris --interactive
```
*Example Follow-up: `[追问] > 请介绍一下为什么在该数据集下 Spectral 算法最好？`*

---

## 🏗️ System Architecture

1.  **Agent Brain (`agent_brain/`)**: Houses taxonomy rules and metric logic.
2.  **Expert Sub-Agents (`expert_sub_agents/`)**: Vertical specialists (Centroid, Topology, Zoo, etc.).
3.  **Agent Core (`agent_core/`)**: The `MasterRouter` (Profiling & Routing) and `ACESupervisor` (Orchestration).
4.  **Tools (`tools/`)**: `AlgorithmZoo`, `DataFactory`, `CoderSandbox`, and `LatexGenerator`.
5.  **Outputs (`outputs/`)**: Structured experiment logs, plots, and reports.

---

## 📝 Recent Updates
- ✅ **Dataset Expansion**: Added MNIST, 20 Newsgroups, and UCI Mfeat.
- ✅ **State Persistence**: Fixed session state in Web UI and CLI for follow-up questions.
- ✅ **Exhaustive Mode**: Parallel multi-threaded execution for the entire algorithm library.
- ✅ **Robust Routing**: Enhanced profiling with skewness and kurtosis detection for better density-clustering routing.
