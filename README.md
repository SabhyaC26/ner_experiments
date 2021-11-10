# ner_experiments

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 -m src.runners.bilstm_crf_runner --out './src/out/' --glove './src/util/embeddings/glove/' --run-id 3
```

### wandb link
https://wandb.ai/sabhyac26/ner_experiments/overview
