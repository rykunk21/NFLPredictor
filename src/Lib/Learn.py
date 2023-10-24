from src.Lib.Dependencies import *
from src.Lib import Data


class Preprocessor:

    class VecHandler:

        # CONSTANTS
        SEED = 42
        AUTOTUNE = tf.data.AUTOTUNE
        MAX_TOKENS = 5000

        def __init__(self, year) -> None:

            self.fileList = glob.glob(f'{RECAPS_DIR}/{year}/**/*txt', recursive=True)
            self.textDS = tf.data.TextLineDataset(self.fileList).filter(lambda x: tf.cast(tf.strings.length(x), bool))

            total_samples = sum(1 for _ in self.textDS)
            if total_samples <= 128:  # Arbitrary threshold
                self.BATCH_SIZE = total_samples
            else:
                self.BATCH_SIZE = 256  # Another arbitrary choice, you can adjust based on your needs



            self.vectorize_layer = self._create_vectorization_layer()
            self.inverse_vocab = self.vectorize_layer.get_vocabulary()
            self.sequences = self._get_sequences()

            self.VOCAB_SIZE = len(self.inverse_vocab)

        ### PUBLIC

        def generateTrainingData(self, window_size, num_ns):
            # Elements of each training example are appended to these lists.
            targets, contexts, labels = [], [], []


            # Build the sampling table for `vocab_size` tokens.
            sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(self.VOCAB_SIZE)

            # Iterate over all sequences (sentences) in the dataset.
            for sequence in tqdm.tqdm(self.sequences):

                # Generate positive skip-gram pairs for a sequence (sentence).
                positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                    sequence,
                    vocabulary_size=self.VOCAB_SIZE,
                    sampling_table=sampling_table,
                    window_size=window_size,
                    negative_samples=0)

                # Iterate over each positive skip-gram pair to produce training examples
                # with a positive context word and negative samples.
                for target_word, context_word in positive_skip_grams:
                    context_class = tf.expand_dims(
                        tf.constant([context_word], dtype="int64"), 1)
                    
                    negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                        true_classes=context_class,
                        num_true=1,
                        num_sampled=num_ns,
                        unique=True,
                        range_max=self.VOCAB_SIZE,
                        seed=self.SEED,
                        name="negative_sampling")

                    # Build context and label vectors (for one target word)
                    context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
                    label = tf.constant([1] + [0]*num_ns, dtype="int64")

                    # Append each element from the training example to global lists.
                    targets.append(target_word)
                    contexts.append(context)
                    labels.append(label)

            return targets, contexts, labels

        def custom_standardization(self, input_data):

            lowercase = tf.strings.lower(input_data)
            return tf.strings.regex_replace(
                    lowercase,'[%s]' % re.escape(string.punctuation), ''
                )

        def allExist(self):

            vocab = self.vectorize_layer.get_vocabulary()
            missing_teams = [team for team in NFLTEAMS if team not in vocab]

            if not missing_teams:
                print("All team names are in the vocabulary!")
            else:
                print(f"Missing teams: {missing_teams}")
        
        ### PRIVATE
        def _create_vectorization_layer(self):

            # Extend the dataset with team names
            team_names_ds = tf.data.Dataset.from_tensor_slices(NFLTEAMS)
            extended_textDS = self.textDS.concatenate(team_names_ds)

            # Create the TextVectorization layer
            vectorize_layer = layers.TextVectorization(
                standardize=self.custom_standardization,
                max_tokens=self.MAX_TOKENS, 
                output_mode='int',
                output_sequence_length=self._compute_optimal_sequence_length())

            vectorize_layer.adapt(extended_textDS.batch(self.BATCH_SIZE))
            return vectorize_layer

        def _get_sequences(self):
            text_vector_ds = self.textDS.batch(self.BATCH_SIZE).prefetch(self.AUTOTUNE).map(self.vectorize_layer).unbatch()
            return list(text_vector_ds.as_numpy_iterator())
        
        def _compute_optimal_sequence_length(self):
            # Compute the lengths of the sequences
            sequence_lengths = []
            for text in self.textDS.as_numpy_iterator():
                sequence_lengths.append(len(text.split()))

            # Sort the lengths
            sorted_lengths = sorted(sequence_lengths)

            # Compute the sequence length for which 95% of the sequences fit within
            percentile_95_index = int(0.95 * len(sorted_lengths))
            optimal_sequence_length = sorted_lengths[percentile_95_index]

            return optimal_sequence_length

    class DNNHandler:
        def __init__(self, year, testSize = .2, randomState = 42) -> None:
            self.year = year
            self.testSize = testSize
            self.randomState = randomState

            self._initializeData()

        def TrainingData(self):            
            return self.xTrain, self.yTrain      
            
        def TestingData(self):
            return self.xTest, self.yTest     

        
        def _initializeData(self):
            rows = []

            # build rows
            for year in range(2001, self.year+1):

                self.vectors = Word2Vec.retrieve([year], NFLTEAMS).get(year)
                self.wbPath = os.path.join(SCORES_DIR, f'{year}.xlsx')

                if os.path.exists(self.wbPath):
                    self.wb = openpyxl.load_workbook(self.wbPath)
                else:
                    raise Exception('No workbook available')
                
                for week in self.wb.sheetnames:
                    
                    # Hard Exclude Preseason
                    if week in ['P1', 'P2', 'P3', 'P4']:
                        continue
                    
                    ws = self.wb[week]

                    col_names = {cell.value: idx for idx, cell in enumerate(next(ws.iter_rows()))}  # Assuming first row contains column names
                    
                    # For each game in the sheet:
                    for row in ws.iter_rows(min_row=2):
                        
                        # Create a Game instance using data from the row
                        game = Data.Game(week=week, year=year, team=row[col_names['teamname']].value)

                        # retrieve vecs
                        homeVec = self.vectors[game.home.name]
                        awayVec = self.vectors[game.away.name]
                        
                        if homeVec is None or awayVec is None: # TODO for commanders
                            continue

                        homeWin = 1 if game.result() else 0
                        homeWin = np.array(homeWin).reshape(1,)

                        # Concatenate the arrays as a sequence
                        homeRow = np.concatenate((homeVec, awayVec, homeWin))
                        awayRow = np.concatenate((awayVec, homeVec, 1 - homeWin))

                        # Append the concatenated rows to the 'rows' list
                        rows.append(homeRow)
                        rows.append(awayRow)

                        print(f'GENERATED ROW {str(game)}: {year}')


            self.data = np.array(rows, dtype=np.float32)

            X, Y =  self.data[:,:-1], self.data[:,-1]        

            self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
                X, Y, test_size=self.testSize, random_state=self.randomState
            )
        
    
    class BayesHandler:
        def __init__(self) -> None:
            pass


class Word2Vec:
    def __init__(self, year, embedding_dim=128, num_ns=15, windowSize=2):
        """
        Initialize the Word2Vec model.

        :param year: Year for which the embeddings are to be generated.
        :param embedding_dim: Dimension of the embedding vectors.
        :param num_ns: Number of negative samples.
        :param windowSize: Size of the window for generating skip-gram pairs.
        """
        handler = Preprocessor.VecHandler(year)
        self.year = year
        self.num_ns = num_ns


        self.targets, self.contexts, self.labels = handler.generateTrainingData(
            window_size=windowSize,
            num_ns=num_ns
        )

        self.vocab_size = len(handler.inverse_vocab)
        self.inverse_vocab = handler.inverse_vocab
        self.embedding_dim = embedding_dim

        # Embedding layers
        self.target_embedding = layers.Embedding(self.vocab_size,
                                                 self.embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(self.vocab_size,
                                                  self.embedding_dim,
                                                  input_length=num_ns+1)

        # Compile the model
        self.compile()

    def call(self, pair):
        """
        Forward pass for the Word2Vec model.

        :param pair: Tuple containing target and context tensors.
        :return: Dot product of target and context embeddings.
        """
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots

    def compile(self):
        """
        Compile the Word2Vec model with optimizer, loss, and metrics.
        """
        target_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name="target_input")
        context_input = tf.keras.Input(shape=(self.num_ns+1,), dtype=tf.int32, name="context_input")

        # Get embeddings
        target_emb = tf.squeeze(self.target_embedding(target_input), axis=1)
        context_emb = self.context_embedding(context_input)

        # Compute the dot product
        dots = tf.einsum('be,bce->bc', target_emb, context_emb)

        self.model = tf.keras.Model(inputs=[target_input, context_input], outputs=dots)
        self.model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

    def train(self, epochs=100, save=False, callbacks=None):
        """
        Train the Word2Vec model.

        :param epochs: Number of epochs for training.
        :param save: Flag to indicate whether to save the entire model.
        :param callbacks: List of callbacks to be used during training.
        """
        dataset = tf.data.Dataset.from_tensor_slices(((self.targets, self.contexts), self.labels))
        dataset = dataset.shuffle(buffer_size=1024).batch(32)
        self.model.fit(dataset, epochs=epochs, callbacks=callbacks)

        if save:
            class_state = {
                # Add your class-specific attributes here
                'year': self.year,
                'inverse_vocab': self.inverse_vocab,
                'embedding_dim': self.embedding_dim,
                'vectors': self.target_embedding.get_weights()[0]
            }
            with open(os.path.join(VECS_DIR, f'{self.year}_w2v.pkl'), 'wb') as f:
                pickle.dump(class_state, f)

    @classmethod
    def load_class_state(cls, year):
        """
        Load the entire class state from a saved file.

        :param file_path: Path to the saved class state file.
        :return: Loaded class instance.
        """
        filePath = os.path.join(VECS_DIR, f'{year}_w2v.pkl')

        with open(filePath, 'rb') as f:
            loaded_instance = pickle.load(f)
        return loaded_instance
    
    @staticmethod
    def retrieve(years, teams):
        """
        Retrieve the vector for a specific team from the saved class state.

        :param year: Year associated with the class state to load.
        :param team: Team name for which to retrieve the vector.
        :return: Vector for the specified team or None if not found.
        """
        vectors_dict = {}
        for year in years:
            # Load the class state for the specified year
            loaded_instance = Word2Vec.load_class_state(year)

            # Access the inverse_vocab and vectors from the loaded instance
            inverse_vocab = loaded_instance['inverse_vocab']
            vectors = loaded_instance['vectors']

            team_vectors = {}
            for team in teams:
                if team in inverse_vocab:
                    word_index = inverse_vocab.index(team)
                    if 0 <= word_index < len(vectors):
                        team_vectors[team] = vectors[word_index]
                else:
                    team_vectors[team] = None

            vectors_dict[year] = team_vectors

        return vectors_dict


    def plot(vectors, labels):
        """
        Reduces the dimensionality of the given vectors to 2 using PCA and plots them.
        
        Args:
        - vectors (list of list of float): The list of vectors to reduce and plot.
        - labels (list of str): Names corresponding to each vector.
        
        Returns:
        None
        """
        # Convert the list of vectors to a numpy array
        vectors = np.array(vectors)

        # Initialize PCA and reduce dimensionality to 2
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)

        # Extract the 2D coordinates
        x_coords = reduced_vectors[:, 0]
        y_coords = reduced_vectors[:, 1]

        # Plot the 2D representation of the vectors
        plt.scatter(x_coords, y_coords)
        
        # Add labels above each point
        for x, y, label in zip(x_coords, y_coords, labels):
            plt.text(x, y, label, fontsize=9, ha='center', va='bottom')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Reduced Vectors')
        plt.show()


class DNN:

    class Architecture:
        def __init__(self, layers, activations=None):
            self.layers = []
            if activations is None:
                self.activations = ['relu'] * len(layers)
            else:
                self.activations = activations
                assert len(activations) == len(layers)
            
            for i, nodeNum in enumerate(layers):
                self.layers.append(Dense(nodeNum, activation=self.activations[i]))

    def __init__(self, year, arch = Architecture([]), optimizer = Adam(learning_rate=0.01)):
        self.year = year
        self.arch = arch
        self.optimzier = optimizer
        self.xTrain = None
        self.yTrain = None
        self.xTest = None
        self.yTest = None
        
        # Initialize data using Preprocessor.DNNHandler
        handler = Preprocessor.DNNHandler(year)
        self.xTrain, self.yTrain = handler.TrainingData()
        self.xTest, self.yTest = handler.TestingData()
        self.vectors = Word2Vec.retrieve([year], NFLTEAMS).get(year)
        self.model = self.build_model()

    def build_model(self):
        """
        Build the DNN model with the specified architecture and optimizer.
        """

        model = tf.keras.Sequential()

        # Add input layer
        model.add(layers.Input(shape=(self.xTrain.shape[1],)))

        # Add hidden layers

        for layer in self.arch.layers:
            model.add(layer)

        # Add output layer
        model.add(layers.Dense(1, activation='softmax'))  # Adjust activation based on your problem

        # Compile the model
        model.compile(optimizer=self.optimzier, loss='binary_crossentropy', metrics=['accuracy'])  # Adjust loss based on your problem

        self.model = model

    def train(self, epochs=10, batch_size=32, visual=False, save=False):
        """
        Train the DNN model.

        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        """
        if self.model is None:
        
            self.build_model()

        history = self.model.fit(self.xTrain, self.yTrain, epochs=epochs, batch_size=batch_size,
                                 validation_data=(self.xTest, self.yTest))
        
        if save:  # Check if the save flag is True
            self.save()  

        if visual:
            self.plot(history)
        return history
    
    def test(self):
        """
        Evaluate the DNN model on the test data and print the test accuracy.
        """
        if self.model is None:
            print("Error: Model has not been built or loaded.")
            return

        # Evaluate the model on the test data
        test_loss, test_accuracy = self.model.evaluate(self.xTest, self.yTest)

        # Print the test accuracy
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    def save(self):

        self.model.save('DNNV2')

    def plot(self, history):
        """
        Plot the training and validation loss and accuracy.

        :param history: Training history returned by model.fit.
        """
        # Plot training & validation loss values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


        # self.model.predict(predictor)


class EloPredictor:
    def __init__(self, year):
        # Any initialization or configuration can go here
        self.year = year

    def predict(self, week, team):
        """
        Predict the outcome of a single game based on ELO.
        
        :param week: Week of the game.
        :param year: Year of the game.
        :param team: Team name.
        :return: Predicted outcome based on ELO.
        """
        game = Data.Game(week=week, year=self.year, team=team)
        return game.eloPrediction(team)

    def evaluate(self):
        """
        Predict the outcomes of multiple games based on ELO.
        
        :param games: List of dictionaries with game parameters.
        :return: List of predicted outcomes based on ELO.
        """

        self.wbPath = os.path.join(SCORES_DIR, f'{self.year}.xlsx')

        if os.path.exists(self.wbPath):
            self.wb = openpyxl.load_workbook(self.wbPath)
        else:
            raise Exception('No workbook available')
        
        predictions = []
        actuals = []
        misclass = []
        for week in self.wb.sheetnames:
                    
            # Hard Exclude Preseason
            if week in ['P1', 'P2', 'P3', 'P4', 'W1', 'W2', 'W3']:
                continue
            
            ws = self.wb[week]

            col_names = {cell.value: idx for idx, cell in enumerate(next(ws.iter_rows()))}  # Assuming first row contains column names
            
            # For each game in the sheet:
            for row in ws.iter_rows(min_row=2):
                team = row[col_names['teamname']].value
                game = Data.Game(week=week, year=self.year, team=team)

                prediction = 0 if game.eloPrediction(team) < .5 else 1
                predictions.append(prediction)

                # retrieve actual outcome
                actual = 1 if game.won(team) else 0
                actuals.append(actual)

                print(f'GAME {game}: {actual == prediction}')
                if not actual == prediction:
                    misclass.append((team, game.eloPrediction(team)))
                
        print(misclass)
        correct_predictions = sum(p == a for p, a in zip(predictions, actuals))
        return correct_predictions / len(predictions)


class BayesianEnsemble:
    def __init__(self) -> None:
        pass


def train():
    pass



def main():
    pass
if __name__ == '__main__':
    from .Dependencies import *
    main()

