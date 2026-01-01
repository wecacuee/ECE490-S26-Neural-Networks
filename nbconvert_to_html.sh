find notebooks/ -name '*Colab.ipynb' ! \( -path '*.ipynb_checkpoints*' \) -exec jupyter nbconvert \{} --to html --output-dir build-html/ \;
