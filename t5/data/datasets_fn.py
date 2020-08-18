import tensorflow.compat.v1 as tf
import functools
'=============sv_udf_news=================='
def sv_udf_news(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_SE/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*news*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============sv_udf_wiki=================='
def sv_udf_wiki(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_SE/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*wiki*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============sv_udf_subs=================='
def sv_udf_subs(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_SE/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*wiki*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============sv_udf_rs=================='
def sv_udf_rs(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_SE/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*rs*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============sv_udf_cc=================='
def sv_udf_cc(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_SE/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*cc*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============sv_udf_fb=================='
def sv_udf_fb(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_SE/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*fb*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============sv_udf_fl=================='
def sv_udf_fl(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_SE/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*fl*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============sv_udf_papers=================='
def sv_udf_papers(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_SE/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*papers*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============dk_udf_dedupe=================='
def dk_udf_dedup(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_DK/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*dedup*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============dk_udf_subs=================='
def dk_udf_subs(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_DK/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*subs*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============no_udf_subs=================='
def no_udf_subs(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_NO/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*subs*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============no_udf_speech=================='
def no_udf_speech(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_NO/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*speech*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============no_udf_news=================='
def no_udf_news(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_NO/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*news*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============no_udf_wiki=================='
def no_udf_wiki(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_NO/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*wiki*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds
'=============no_udf_dedupe=================='
def no_udf_dedup(split, shuffle_files=False):
  del shuffle_files
  DATA_DIR = 'gs://t5_swe_bucket/Data/U/U_NO/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*dedup*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["title", "text"], ex)))
  return ds










'=============Translate to english=================='
def tte_dataset_fn(split, shuffle_files=False):
  del shuffle_files
  # Load lines from the text file as examples.
  DATA_DIR = 'gs://t5_swe_bucket/Data/T/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*TTE*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["sv", "en"], ex)))
  return ds
'=============Translate to swedish=================='
def tts_dataset_fn(split, shuffle_files=False):
  del shuffle_files
  # Load lines from the text file as examples.
  DATA_DIR = 'gs://t5_swe_bucket/Data/T/'
  FILES_PATH = tf.io.gfile.glob(DATA_DIR + "*TTS*")

  ds = tf.data.TextLineDataset([FILES_PATH])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["en", "sv"], ex)))
  return ds
