# ner_experiments

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
cd src
python3 -m bilstm_crf_runner --out './out/' --glove './util/embeddings/glove/' --run-id 3
```

### wandb link
https://wandb.ai/sabhyac26/ner_experiments/overview
