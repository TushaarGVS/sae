# Sparse autoencoder

```shell
cd $HOME/sae
# Poetry creates a venv if not already in a specific conda env:
# https://stackoverflow.com/a/77807639/6907625.
poetry install
```

__Note.__ The package uses `triton-3.0.0` which needs to be installed from the source.

---

## Running

```shell
cd $HOME/sae

# Extract activations for a huggingface dataset. See `submit.sh` file for
# exact calling args.
python scripts/get_recurrentgemma_activations.py

# Runs the app on `localhost:8000`.
python visualization/app.py
```
