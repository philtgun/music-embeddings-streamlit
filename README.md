# Music Embeddings with Streamlit

![Screenshot](screenshot.png)

```shell
python3.8 -m venv venv
source venv/bin/activate.fish
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run
Necessary environment variables:
```shell
DATA_PATH=/path/to/data
```

Run:
```shell
streamlit run src/main.py
```

## Dev
```shell
pip install pre-commit
pre-commit install
```
