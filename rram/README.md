# Installation
Tested python version: 3.10.12

1. Install `requirements.txt` in a venv.

1. Install `larq-zoo` and update package after *every* change:

```bash
python setup.py sdist bdist_wheel
pip install ./dist/larq_zoo-2.3.2-py3-none-any.whl
```
To update any changes, use the argument `--force-reinstall`.

1. Run `train_bnns.py`.
