"""Code for each data owner in the collaboration."""
import tensorflow as tf


class BloomRegressor:
  def __init__(self):
    self.label_square = None
    self.covariate_label_product = None
    self.covariate_square = None
    self.beta_estimator = None

  @classmethod
  def estimator_fn(cls, x_p, y_p):
    # Recall beta = np.inv(X.T @ X) * (X.T @ y)
    yy_p = tf.matmul(y_p, y_p, transpose_a=True)  # per-party y.T @ y
    xy_p = tf.matmul(x_p, y_p, transpose_a=True)  # per-party X.T @ y
    xx_p = tf.matmul(x_p, x_p, transpose_a=True)  # per-party X.T @ X
    n_p = x_p.shape[0]                            # per-party sample size
    return yy_p, xy_p, xx_p, n_p

  def fit(self,
          training_players,
          epochs=1,
          verbose=0,
          validation_split=None,
          batch_size=1):
    if validation_split is not None:
      raise NotImplementedError()
    if epochs != 1 or batch_size != 1:
      raise ValueError("Invalid arguments for training with normal equations.")

    partial_estimators = [player.compute_estimators(self.estimator_fn)
                          for player in training_players]

    accumulator = {}
    keys = ["label_square", "covariate_label_product", "covariate_square"]

    for results_tuple in zip(*partial_estimators)):
      for i, estimator


    if not verbose:
      return self
    # TODO: everything below here
    self._print_training_stats()
    if verbose > 1:
      self.print_stderror_estimates()
      self.print_pvalues()


class DataOwner:
  """Contains code meant to be executed by a data owner Player."""
  def __init__(
      self,
      player_name,
      training_set_path,
      test_set_path,
      batch_size,
  ):
    self.player_name = player_name
    self.num_players = num_players
    self.training_set_path = training_set_path
    self.test_set_path = test_set_path
    self.batch_size = batch_size
    self.train_initializer = None
    self.test_initializer = None

  @property
  def initializer(self):
    return tf.group(self.train_initializer, self.test_initializer)

  def _build_training_data(self):
    """Preprocess training dataset

    Return single batch of training dataset
    """
    def norm(x, y):
      return tf.cast(x, tf.float32), tf.expand_dims(y, 0)

    x_raw = tf.random.uniform(
        minval=-.5,
        maxval=.5,
        shape=[self.training_set_size, self.num_features])

    y_raw = tf.cast(tf.reduce_mean(x_raw, axis=1) > 0, dtype=tf.float32)

    train_set = tf.data.Dataset.from_tensor_slices((x_raw, y_raw)) \
        .map(norm) \
        .repeat() \
        .shuffle(buffer_size=self.batch_size) \
        .batch(self.batch_size)

    train_set_iterator = train_set.make_initializable_iterator()
    self.train_initializer = train_set_iterator.initializer

    x, y = train_set_iterator.get_next()
    x = tf.reshape(x, [self.batch_size, self.num_features])
    y = tf.reshape(y, [self.batch_size, 1])

    return x, y

  def _build_testing_data(self):
    """Preprocess testing dataset

    Return single batch of testing dataset
    """
    def norm(x, y):
      return tf.cast(x, tf.float32), tf.expand_dims(y, 0)

    x_raw = tf.random.uniform(
        minval=-.5,
        maxval=.5,
        shape=[self.test_set_size, self.num_features])

    y_raw = tf.cast(tf.reduce_mean(x_raw, axis=1) > 0, dtype=tf.float32)

    test_set = tf.data.Dataset.from_tensor_slices((x_raw, y_raw)) \
        .map(norm) \
        .batch(self.test_set_size)

    test_set_iterator = test_set.make_initializable_iterator()
    self.test_initializer = test_set_iterator.initializer

    x, y = test_set_iterator.get_next()
    x = tf.reshape(x, [self.test_set_size, self.num_features])
    y = tf.reshape(y, [self.test_set_size, 1])

    return x, y

  @tfe.local_computation
  def compute_estimator(self, estimator_fn):
    x, y = self._build_training_data()
    partial_beta = estimator_fn(x, y)
    return partial_beta
