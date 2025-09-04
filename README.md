# Music Similarity Agent


## Music Similarity Agent

This project provides a **modular and extensible music similarity pipeline** designed to compare music recordings using multiple complementary similarity tools, including tonal/harmonic audio embeddings (HPCP, CENS) and semantic metadata comparison powered by sentence transformers.

## Features

- **Multiple similarity tools:** Easily plug in new audio or metadata-based similarity measures.
- **Robust semantic metadata similarity:** Uses sentence transformers to measure artist, title, and label similarity beyond exact matches.
- **Flexible aggregation:** Supports weighted combinations or learned models (e.g., #TODO: XGBoost) for final similarity scoring.
- **Explainability:** Returns detailed explanations per similarity tool and supports SHAP-based visual feature importance for learned aggregators.
- **Scalable code structure:** Clean separation of concerns with tools organized under `src/tools/`, and an orchestrating agent in `src/agent/`.
- **Loader integration:** Compatible with DaTACOS data and ready for new feature sources like CLAP or MERT embeddings.

## Usage

1. Instantiate your data loader.
2. Initialize the `MusicSimilarityAgent` with the loader.
3. Call `predict_similarity(perf_id1, perf_id2)` to get:
   - A similarity score [0,1]
   - A natural language explanation of contributing features

---

This framework enables easy experimentation and deployment of audio and metadata similarity models with transparency and extensibility.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ElenaDiamantidou/agentic_audio_similarity.git
cd agentic_audio_similarity
pip install -r requirements.txt
```

## Usage

```
main.py
```

## Project Structure

```
agentic_audio_similarity/
│
├── data/       # data utils and loaders
│   ├── __init__.py
│   ├── loader.py
│   └── create_train_set_for_aggregator_model.py
│
├── examples/   # test examples
│   ├── find_similar_performance.py
│   └── shap_explain_weights.py
│
│
├── models/   # models
│   └── aggregator_model.py
│
│
├── src/   
│    ├── agent/
│    │   └── music_similarity_agent.py # Orchestrates similarity tools and aggregation
│    ├── tools/
│         ├── aggregator/ # Aggregation strategies and explainability tools
│         ├── metadata/ # Metadata processing and semantic similarity utilities
│         ├── similarity/ # Audio and metadata similarity tools implementations
│    └── utils/
│         └── dataframe_utils.py # Utilities for DataFrame creation/manipulation (e.g., training data prep)
│
│
├── main.py               # main script to run the agent
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Dependencies

- Python 3.6+
- Other dependencies listed in `requirements.txt`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
