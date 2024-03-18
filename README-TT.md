## Basic Usage (for debugging)

At the top of `test.py`, set whether the model is a vanilla transformer or transposed transformer in `model_config`. Then run

```
python test.py --data_filepath="data/<DATASET>.txt" --system.work_dir="out/<OUT_DIR>"
```

The model will train, saving every 500 iterations.
