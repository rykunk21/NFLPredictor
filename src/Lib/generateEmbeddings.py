import io
import re
import string
import tqdm
import glob

import numpy as np

import tensorflow as tf
from keras import layers

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
CURRENTYEAR=2022
CURRENTWEEK=1

path_to_dir = f'datasets/gameRecaps/{CURRENTYEAR}'


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
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
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels



all_files = glob.glob(f'{path_to_dir}/W{CURRENTWEEK}/*txt', recursive=True)

text_ds = tf.data.TextLineDataset(all_files).filter(lambda x: tf.cast(tf.strings.length(x), bool))

# create a custom standardization function to lowercase the text and
# remove punctuation.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


# Define the vocabulary size and the number of words in a sequence.
vocab_size = 4096
sequence_length = 10

# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)


vectorize_layer.adapt(text_ds.batch(1024))
inverse_vocab = vectorize_layer.get_vocabulary()

text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())

for seq in sequences[:5]:
  print(f"{seq} => {[inverse_vocab[i] for i in seq]}")


targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=15,
    vocab_size=vocab_size,
    seed=SEED)

targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

print(tf.shape(targets))
print(tf.shape(contexts))
print(tf.shape(labels))

BATCH_SIZE = 238
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE)


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                            embedding_dim,
                                            input_length=1,
                                            name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                            embedding_dim,
                                            input_length=15+1)

    def call(self, pair):
        target, context = pair

        # Ensure that word_emb always has a rank of 2
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        word_emb = tf.expand_dims(word_emb, axis=1)  # Ensure shape is (batch_size, embedding_dim)

        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)

        return dots

def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])


word2vec.fit(dataset, epochs=100)

def retreieve(word):
  def word_to_index(word):
    try:
        return inverse_vocab.index(word)
    except ValueError:
        return None
  word_index = word_to_index(word)
  word_vector = word2vec.target_embedding(tf.constant([[word_index]]))
  return tf.squeeze(word_vector, axis=0).numpy()

nfl_teams = [
    "cardinals",
    "falcons",
    "ravens",
    "bills",
    "panthers",
    "bears",
    "bengals",
    "browns",
    "cowboys",
    "broncos",
    "lions",
    "packers",
    "texans",
    "colts",
    "jaguars",
    "chiefs",
    "chargers",
    "rams",
    "dolphins",
    "vikings",
    "patriots",
    "saints",
    "giants",
    "jets",
    "raiders",
    "eagles",
    "steelers",
    "49ers",
    "seahawks",
    "buccaneers",
    "titans",
    "commanders"]

vecs = list(map(retreieve, nfl_teams))
vecs = list(map(np.squeeze, vecs))

def write_to_tsv(data, filename):
    """
    Write a list of lists to a TSV file.

    :param data: List of lists containing the data
    :param filename: Name of the TSV file to write to
    """
    with open(filename, 'w') as f:
        for row in data:
            f.write('\t'.join(map(str, row)))
            f.write('\n')



if len(nfl_teams) != len(vecs):
    raise ValueError("Both input lists must have the same length.")

data = []
for team, array in zip(nfl_teams, vecs):
    modified_array = [team] + array.tolist()
    data.append(modified_array)



write_to_tsv(data, '../vecs/2022/final.tsv')
