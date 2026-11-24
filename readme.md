This project lets you train your own small language model and learn the basics of how language models work.
Note: An example model is included. It is a small 1.1-million-parameter model trained on plain text using trainer.py.


## 1. Install Dependencies
Open your terminal and install the required Python libraries:bash
pip3 install torch numpy inputimeout
Kodu dikkatli kullanın.
## 2. Add Training Data
Open train_data.txt and paste your training text (such as dialogue examples, essays, or book excerpts). Avoid using excessively large datasets (e.g., 70 MB or more).
## 3. Train the Model
Run the training script via python

python3 trainer.py

Monitor the training loss as the model learns. Once complete, give your model a name and save it.
## 4. Run the Model
Use loader.py to load and interact with your saved model.

Project Files

trainer.py: Trains the model using train_data.txt and saves the output file.
loader.py: Loads a trained model so you can test or chat with it.
train_data.txt: The dataset file where you place the text used to train the model.


The scripts train small models. While the model can learn basic grammar and sentence structure, it lacks reasoning abilities and cannot perform math.
