# TT-ATP

Implementing an ATP using TreeLSTM and Transformer as NN, and MetaMath set.mm as the dataset.

## Usage

1. Install libraries `pip freeze > requirements.txt`
2. Download set.mm file `wget https://github.com/metamath/set.mm/raw/develop/set.mm`
3. Parse set.mm and save dataset `python3 src/dataloader.py`
4. Train the model `python3 src/training.py` three times and save model states and results in different files
5. Test the model `python3 src/test.py` three times and save results in different files
6. Plot the results `python3 src/plot.py`
