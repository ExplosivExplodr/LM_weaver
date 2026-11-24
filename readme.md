This project lets you train your own small language model and learn the basics of how language models work. It is going to be extended further, but right now it is pretty straightforward.


## 1. Install Dependencies
Open your terminal and install the required Python libraries:

```bash
pip3 install torch inputimeout
```

(NumPy is not required; it only comes in as a transitive dependency of torch.)
## 2. Add Training Data
Open train_data.txt and paste your training text (such as dialogue examples, essays, or book excerpts). Separate entries with a blank line. Avoid using excessively large datasets (e.g., 70 MB or more).
## 3. Train the Model
Run the training script via python

```bash
python3 trainer.py
```

Monitor the training and validation loss as the model learns. The validation loss
is measured on the last 10% of your data, which is never used for training -- if
it starts rising while the training loss keeps falling, the model is overfitting.
Once complete, give your model a name and save it.
## 4. Run the Model
Use loader.py to load and interact with your saved model.

```bash
python3 loader.py
```

Checkpoints store their architecture settings alongside the weights, so you can
change the hyperparameters in trainer.py without breaking existing models.

Project Files

- trainer.py: Trains the model using train_data.txt and saves the output file.
- loader.py: Loads a trained model so you can test or chat with it.
- train_data.txt: The dataset file where you place the text used to train the model.


The scripts train small models. While the model can learn basic grammar and sentence structure, it lacks reasoning abilities and cannot perform math.
