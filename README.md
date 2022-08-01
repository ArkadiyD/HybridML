# Code for the paper:
> **Reconstructing Phylogenetic Networks via Cherry Picking and Machine Learning**  
> *Giulia Bernardini, Leo van Iersel, Esther Julien and Leen Stougie*

To run the code, the following packages have to be installed: `numpy`, `pandas`, `scikit-learn`, `networkx`, and 
`joblib`.

### Cherry picking heuristic (CPH)
The CPH is implemented in `CPH.py`. To experiment with this file,

RUN in terminal:
```bash
python main_heuristic.py <instance num.> <ML model name> <leaf number> <bool (0/1) for exact input> <option>
```
`option`: 
if `exact input` = 0:
    `option` = reticulation number,
else:
    `option` = forest size

EXAMPLE: 
```bash
python main_heuristic.py 0 N10_maxL100_random_balanced 20 0 50
```
In the file `test_data_gen.py`, tree sets used for testing can be generated.

### Generate training data
The code for initializing and updating the features can be found in `Features.py`.

For training the random forest, we first have to generate training data. 
The code for this is given in `train_data_gen.py`. 
RUN in terminal:
```bash
python train_data_gen.py <number of networks> <maxL>
```
`maxL`: maximum number of leaves per network
EXAMPLE:
```bash
python train_data_gen.py 10 20
```

### Train random forest
In the folder `LearningCherries` you can find the code for training a random forest. 
In the folder `LearningCherries/TrainedRFModels`, there are some trained random forests in `joblib` format.

### Preprocessed data
The `Data` folder consists of:
- `Results` of CPH instances. The instances are divided in 4 categories: FTS (Full Tree Set), Real, RealSmall, and RTS (Restricted Tree Set).
  - `FINAL_heuristic_scores_ML[<random forest model used>]_TEST[<test instance type used>]_<number of instances>.pickle`
- `Test` for different test instances
- `Train` for generated train data
