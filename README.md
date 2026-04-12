# ACE Agent Demo

ACE Agent is a lightweight multi-agent clustering demo built around a master router,
specialized expert agents, synthetic sklearn-style datasets, and a Streamlit web UI.

## What this demo includes

- A master agent that profiles data and routes work to expert agents
- Five expert agents for centroid, topology, dimensionality, deep representation,
  and multi-view consensus clustering
- Deterministic code generation plus sandboxed execution for each expert
- Synthetic demo datasets: blobs, moons, s-curve, and smile
- Automatic metrics, figures, and LaTeX report generation
- A Streamlit page for model configuration, chat-driven analysis, and visible
  decision trace

## Run

```bash
streamlit run ACE_Agent/web_demo.py
```

## Smoke test

```bash
python ACE_Agent/demo_runner.py --dataset smile
```

## Notes

- The web UI stores model settings in `ACE_Agent/.ace_demo_config.json`.
- LLM access is optional. If no API is configured, the system still runs the full
  clustering pipeline and produces deterministic reports.
- The UI shows a "decision trace" rather than raw hidden chain-of-thought.

