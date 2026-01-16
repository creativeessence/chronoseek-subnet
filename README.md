# Semantic Video Moment Retrieval (SVMR) Subnet

> **A Bittensor subnet for decentralized semantic video moment retrieval.**

The **SVMR Subnet** enables semantic search over video content by mapping natural-language scene descriptions to precise timestamp intervals within a video.

---

## 📚 Project Documentation

This project is organized into the following key documents:

- **[Problem Statement](docs/PROBLEM_STATEMENT.md)**  
  *Why this subnet exists, the "dark data" problem, and the limitations of current search tools.*

- **[System Design](docs/DESIGN.md)**  
  *Technical architecture, including Miner/Validator logic, Synthetic Task Generation, and SOTA research references (CLIP, LLMs).*

- **[Business Logic & Market Rationale](docs/BUSINESS_LOGIC.md)**  
  *Market size ($94B+), commercialization strategy, and competitive advantage against centralized giants.*

---

## 🚀 Quick Start (Ideathon)

### 1. The Core Concept
We are building a decentralized protocol where:
*   **Miners** use AI models (CLIP, Transformers) to "watch" videos and find specific moments.
*   **Validators** generate synthetic queries to grade miners and serve organic requests.
*   **Users** get precise timestamps (e.g., "04:12 - 04:18") for natural language queries.

### 2. Architecture Overview
```
User / Client
   │
   ▼
Validator (Gateway)
   ├─ Synthetic evaluation (scoring & weights)
   └─ Organic query routing
   │
   ▼
Miners
   └─ Semantic video analysis (CLIP / SOTA Models)
```

## �📦 Installation & Setup

This project uses `poetry` for dependency management.

### Prerequisites
- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) installed

### 1. Clone & Install
```bash
git clone https://github.com/creativeessence/chronoseek-subnet.git
cd chronoseek-subnet

# Install dependencies and create virtualenv
poetry install
```

### 2. Activate Virtual Environment
```bash
poetry env activate
```

## 🏃‍♂️ Running Locally

### 1. Start the Miner
The miner listens for HTTP requests from validators.
```bash
# Starts miner on port 8000
poetry run python miner.py
```
*You can configure the wallet/hotkey via environment variables (see `miner.py`).*

### 2. Start the Validator
The validator generates synthetic tasks, queries miners, and scores them.
```bash
# Starts validator loop
poetry run python validator.py
```
*Note: For local testing without a running Bittensor chain, ensure the code handles mock metagraphs appropriately.*

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WALLET_NAME` | Name of your coldkey | `default` |
| `HOTKEY_NAME` | Name of your hotkey | `default` |
| `NETUID` | Subnet NetUID | `1` |
| `NETWORK` | Network (finney, test, local) | `finney` |
| `PORT` | Miner HTTP Port | `8000` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
