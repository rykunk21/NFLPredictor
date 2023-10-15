import tensorflow as tf
from keras.layers import Input, Dense, Concatenate
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import joblib
from keras.regularizers import l2

import os


"""
RELEVANT CODE FOR GENERTAING A DNN FOR THE ENSEMBLE
"""

VECTS_DIR = "C:/Users/rykun/Documents/Projects/NFLPredictorLegacy/src/vecs/2023/final.tsv"
SCORES = "C:/Users/rykun/Documents/Projects/NFLPredictorLegacy/datasets/gameScores/2023.xlsx"

VECTS_BASE_DIR = "C:/Users/rykun/Documents/Projects/NFLPredictorLegacy/src/vecs/"
SCORES_BASE_DIR = "C:/Users/rykun/Documents/Projects/NFLPredictorLegacy/datasets/gameScores/"

def get_available_years(base_dir):
	"""Return a list of available years based on the directory names."""
	return [dir_name for dir_name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, dir_name))]

def generate_combined_training_data():
	all_data_train = []
	all_labels = []

	# Get all available years
	years = get_available_years(VECTS_BASE_DIR)

	for year in years:
		vecs_dir = os.path.join(VECTS_BASE_DIR, year, "final.tsv")
		scores_file = os.path.join(SCORES_BASE_DIR, f"{year}.xlsx")

		if os.path.exists(vecs_dir) and os.path.exists(scores_file):
			xls = pd.ExcelFile(scores_file)
			sheet_names = xls.sheet_names

			data_train, labels = process_game_data(scores_file, sheet_names, vecs_dir)

			# Convert data_train and labels to numpy arrays before appending
			data_train = np.array(data_train)
			labels = np.array(labels)

			all_data_train.append(data_train)
			all_labels.append(labels)

			

	# Convert lists of arrays to single arrays
	all_data_train = np.vstack(all_data_train)
	all_labels = np.hstack(all_labels)

	return all_data_train, all_labels



def tsv_to_dict(filename):
	"""
	Convert a TSV file into a dictionary where the first column (string) is the key 
	and the next 128 floating point values are the value (in a list).
	"""
	data_dict = {}

	with open(filename, 'r') as file:
		for line in file:
			# Split each line based on tab delimiter
			parts = line.strip().split('\t')
			
			# Extract the string (key) and the list of floats (value)
			key = parts[0]
			values = [float(x) for x in parts[1:]]
			
			# Add to the dictionary
			data_dict[key] = values
			
	return data_dict

def visualize(history):
  """
  Visualization code for logistic regressors
  """
  # Plot training & validation accuracy values
  plt.figure(figsize=(12, 4))

  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')

  # Plot training & validation loss values
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')

  plt.tight_layout()
  plt.show()
  plt.savefig("learningCurves.pdf", format='pdf')


def generateTrainingData(): ## TODO
  """
  Reads TSVvectors and returns training data from all team as np vectors
  Team1 vector, Team2 vector, HomeWin
  
  """
  teamvecs = tsv_to_dict(VECTS_DIR)

 
  sheet = 'W3'   
  df = pd.read_excel(SCORES, sheet_name = sheet, engine='openpyxl')

  labels = df['HOMEWIN']

  data1_train = list(map(lambda x: teamvecs[x], df['HOME']))
  data2_train = list(map(lambda x: teamvecs[x], df['AWAY']))

  return np.array(data1_train), np.array(data2_train), np.array(labels)

def process_game_data(filename, sheet_names, vecs_dir):
	# Create an empty dataframe to store the processed data
	processed_df = pd.DataFrame()
	teamvecs = tsv_to_dict(vecs_dir)
	all_labels = []

	for sheet in sheet_names:
		# Read the specific sheet from the xlsx file
		df = pd.read_excel(filename, sheet_name=sheet)
		
		# Drop rows with NaN values
		df = df.dropna()

		# Sort the dataframe by ID to ensure teams of the same game are consecutive
		df = df.sort_values(by='ID')
		
		# Temporary lists to store the processed data and labels for the current sheet
		temp_data = []
		temp_labels = []
		
		# Iterate over the dataframe with a step of 2 (since each game has 2 rows)
		for i in range(0, len(df), 2):
			team1 = df.iloc[i]['teamname']
			team2 = df.iloc[i+1]['teamname']
			result1 = df.iloc[i]['win']  # Assuming a "win" column exists

			team1_vec = teamvecs.get(team1, [])  # Use get() to avoid KeyError
			team2_vec = teamvecs.get(team2, [])

			# Append data in the desired format
			if not pd.isna(result1):
				temp_data.append(team1_vec + team2_vec)
				temp_data.append(team2_vec + team1_vec)  # This is the mirrored data
				temp_labels.append(result1)
				temp_labels.append(1 - result1)  # This is the opposite result for the mirrored data

		# Convert the temporary data to a DataFrame and append to the main dataframe
		temp_df = pd.DataFrame(temp_data)
		processed_df = pd.concat([processed_df, temp_df], ignore_index=True)
		all_labels.extend(temp_labels)

	return processed_df, all_labels

def generatePredictor(home, away, LR=False):
	teamvecs = tsv_to_dict(VECTS_DIR)

	home_pred = teamvecs[home]
	away_pred = teamvecs[away]
	if LR:
		return np.array(home_pred), np.array(away_pred)
	return [np.concatenate((home_pred, away_pred))]

def logisticRegressor():
	"""
	Defines and trains a logisctic regressor
	returns -> 
	"""

	x, y = generate_combined_training_data()
	 
	if np.isnan(x).sum() > 0:
	# Option 1: Drop rows with NaN values
		mask = ~np.isnan(x).any(axis=1)
		x = x[mask]
		y = y[mask]

	xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

	data1_train = xTrain[:, :128]
	data2_train = xTrain[:, 128:]

	# Split the test data similarly
	data1_test = xTest[:, :128]
	data2_test = xTest[:, 128:]

	# Define the three input layers
	input1 = Input(shape=(128,), name='input_128_1')
	input2 = Input(shape=(128,), name='input_128_2')


	# Concatenate the inputs
	concatenated = Concatenate()([input1, input2])

	# Add a hidden dense layer with 256 units and a ReLU activation
	hidden1 = Dense(256, activation='relu', kernel_regularizer=l2(0.1))(concatenated)

	# Add a hidden dense layer with 128 units and a ReLU activation
	hidden2 = Dense(128, activation='relu', kernel_regularizer=l2(0.1))(hidden1)

	# Optionally, add dropout to prevent overfitting
	dropout = tf.keras.layers.Dropout(0.5)(hidden2)

	# Add a hidden dense layer with 64 units and a ReLU activation
	hidden3 = Dense(64, activation='relu', kernel_regularizer=l2(0.1))(dropout)

	# Define the logistic regression layer for binary classification
	output = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(concatenated)



	# Create the model
	model = tf.keras.Model(inputs=[input1, input2], outputs=output)

	optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


	# Compile the model (binary_crossentropy is used for binary classification)
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

	# Print model summary
	model.summary()

	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

	history = model.fit([data1_train, data2_train], 
						yTrain,
						epochs=200, 
						validation_split=0.2,
						callbacks=[reduce_lr])


	loss, accuracy = model.evaluate([data1_test, data2_test], yTest)
	print("Test Loss:", loss)
	print("Test Accuracy:", accuracy)

	visualize(history)

	return model

def rfClassifier():
	
	# List of sheet names to process

	x, y = generate_combined_training_data()
	 
	if np.isnan(x).sum() > 0:
	# Option 1: Drop rows with NaN values
		mask = ~np.isnan(x).any(axis=1)
		x = x[mask]
		y = y[mask]

	xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

	# Initialize the RandomForestRegressor
	rf_classifier = RandomForestClassifier(n_estimators=1000, verbose = 2)

	# Fit the model to the training data
	rf_classifier.fit(xTrain, yTrain)

	# Predict on the test set
	y_pred = rf_classifier.predict(xTest)

	# Calculate and print the mean squared error
	mse = mean_squared_error(yTest, y_pred)
	print(f"Mean Squared Error: {mse:.2f}")

	probabilities = rf_classifier.predict_proba(xTest)
	# For each true label and predicted probability pair...
	testSet = []
	for true, prob in zip(yTest, probabilities[:, 1]):  # We're using [:, 1] to get the probabilities for class '1'
		testSet.append((f'true: {true}', f'pred: {prob}'))
	return mse, testSet, rf_classifier

def getCurrentGames(week):
   """
   Gets the current games as (home, away) tuples
   """
   pass

def main():
	 
	# models = []
	# for _ in range(100):
	# 	mse, _, model = rfClassifier()
	# 	models.append((mse,model))

	# mse, model = sorted(models, key = lambda x: x[0])[0]
	# print(mse)

	model = joblib.load('./model1.pkl')
	# # joblib.dump(model, 'model1.pkl')

	# model = rfClassifier()


	games = [
		('raiders','patriots'),
		('rams','cardinals'),
		('jets','eagles'),		
		('buccaneers','lions'),		
		('bills','giants'),
		('titans','ravens')
		
	]
	 
	for game in games:
		t1, t2 = game    # Print the shapes of the inputs
		
		inputs = generatePredictor(t1, t2, False)

		# input1 = np.array(inputs[0]).reshape(1, -1)
		# input2 = np.array(inputs[1]).reshape(1, -1)
		print(f'{t1} vs {t2}', model.predict_proba(inputs))

	




if __name__ == '__main__':
   main()


