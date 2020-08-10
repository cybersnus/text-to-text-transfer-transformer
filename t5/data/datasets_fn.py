import tensorflow.compat.v1 as tf
import pandas as pd

def unsupervised_dataset_fn(split, shuffle_files=False):
  "test med expressen, hårdkodat. OBS! Man kan ej ha .gin config på path till ds, det fuckar argesen som skickas till _validate_args i utils.py. Det blir alltså hårdkodat"
  path_to_file = "gs://t5_swe_bucket/Data"
  dumps = ["/expressen1.json"]
  del shuffle_files
  for i, dump in enumerate(dumps):
    df = pd.read_json(path_to_file + dump)
    df.head()
    df.drop(['published','url','title', 'description'], inplace=True, axis=1)
    df.dropna(inplace=True)
    ds = tf.data.Dataset.from_tensor_slices((None, df['text'].str.lower()))
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
  return ds
