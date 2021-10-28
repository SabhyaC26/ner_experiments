# ner_experiments

```python
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python3 -m src.runners.bilstm_crf_runner --out './src/out/' --glove './src/util/embeddings/glove/' --run-id 3
```
