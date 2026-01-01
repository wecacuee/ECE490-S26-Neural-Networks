find notebooks/ -name '*.ipynb' ! \( -name '*Colab.ipynb' -o -path '*.ipynb_checkpoints*' \) -exec python3 scripts/export_ipynb_to_colab.py \{} \;
