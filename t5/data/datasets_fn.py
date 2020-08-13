import tensorflow.compat.v1 as tf
import functools

def unsupervised_dataset_fn(split, shuffle_files=False):
  del shuffle_files
  
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/'
  FILES = tf.io.gfile.listdir(DATA_DIR)
  FILES_PATH = [DATA_DIR + FILE for FILE in FILES]

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds

def translate_dataset_fn(split, shuffle_files=False):
  del shuffle_files
  # Load lines from the text file as examples.
  DATA_DIR = 'gs://t5_swe_bucket/Data/T/'
  FILES = tf.io.gfile.listdir(DATA_DIR)
  FILES_PATH = [DATA_DIR + FILE for FILE in FILES]

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["en", "sv"], ex)))
  return ds
