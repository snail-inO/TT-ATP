# TT-ATP

Implementing an ATP using TreeLSTM and Transformer as NN, and MetaMath set.mm as the dataset.

## Usage

1. Go to project root dir `cd TT-ATP`
2. Install libraries `pip install -r requirements.txt`
3. Download set.mm file `wget https://github.com/metamath/set.mm/raw/develop/set.mm`
4. Parse set.mm and save dataset `python3 src/dataloader.py`
5. Train the model `python3 src/training.py` three times and save model states and results in different files
6. Test the model `python3 src/test.py` three times and save results in different files
7. Plot the results `python3 src/plot.py`
