
from src.Lib import Dependencies
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

    class DNNHandler:
        def __init__(self) -> None:
            pass

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



    def train(self, epochs=100, save = False, callbacks=None):
        """
        Train the Word2Vec model.

        :param epochs: Number of epochs for training.
        :param callbacks: List of callbacks to be used during training.
        """
        dataset = tf.data.Dataset.from_tensor_slices(((self.targets, self.contexts), self.labels))
        dataset = dataset.shuffle(buffer_size=1024).batch(32)
        self.model.fit(dataset, epochs=epochs, callbacks=callbacks)

        if save:
            # Assuming you have a method or attribute that gets the vectors
            vectors = self.get_vectors()
            self.save_vectors(vectors, os.path.join(VECS_DIR, f'{self.year}'))


    def get_vectors(self):
        """
        Retrieve the vectors (embeddings) from the model.

        :return: Numpy array of vectors.
        """
        # Get the weights from the target_embedding layer
        return self.target_embedding.get_weights()[0]

    @staticmethod
    def save_vectors(vectors, filename):
        """
        Save vectors to a binary file using numpy. If the directory doesn't exist, it will be created.

        :param vectors: Numpy array of vectors.
        :param filename: Name of the file to save to.
        """
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        np.save(filename, vectors)


    def retrieve(self, word):
        """
        Retrieve the embedding vector for a given word.

        :param word: Word for which the embedding is to be retrieved.
        :return: Embedding vector of the word.
        """
        def word_to_index(word):
            try:
                return self.inverse_vocab.index(word)
            except ValueError:
                return None

        word_index = word_to_index(word)
        if word_index is None:
            print(f"Word '{word}' not found in vocabulary!")
            return None

        word_vector = self.target_embedding(tf.constant([[word_index]]))
        return tf.squeeze(word_vector, axis=0).numpy()
    

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
    def __init__(self) -> None:
        pass


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

    def batch_predict(self, games):
        """
        Predict the outcomes of multiple games based on ELO.
        
        :param games: List of dictionaries with game parameters.
        :return: List of predicted outcomes based on ELO.
        """
        predictions = []
        for game_params in games:
            predictions.append(self.predict(**game_params))
        return predictions

    def evaluate(self, predictions, actual_outcomes):
        """
        Evaluate the accuracy of the ELO predictions.
        
        :param predictions: List of predicted outcomes.
        :param actual_outcomes: List of actual outcomes.
        :return: Accuracy of the predictions.
        """
        correct_predictions = sum(p == a for p, a in zip(predictions, actual_outcomes))
        return correct_predictions / len(predictions)


class BayesianEnsemble:
    def __init__(self) -> None:
        pass


def train():
    pass


def main():

    ep = EloPredictor(2023)
    print(ep.predict('W3', 'bears'))

if __name__ == '__main__':

    main()

else:
    from src.Lib.Dependencies import *
    print('alternate import')