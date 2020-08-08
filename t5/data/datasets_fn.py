import gin
import tensorflow.compat.v1 as tf


@gin.configurable
def unsupervised_dataset_fn(split, shuffle_files=False, path_to_file=gin.REQUIRED):
  "test med expressen, hårdkodat"                      
  dumps = ["/expressen1.json","/expressen2.json","/expressen3.json","/expressen4.json","/expressen5.json"]
  del shuffle_files
  for i, dump in enumerate(dumps):
    df = pd.read_json(path_to_file + dump)
    df.head()
    df.drop(['published','url','title', 'description'], inplace=True, axis=1)
    df.dropna(inplace=True)
    ds = tf.data.Dataset.from_tensor_slices((None, df['text'].str.lower()))
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
    if i==0:
      combined_dataset = ds
    else:
      combined_dataset = combined_dataset.concatenate(ds)

  return combined_dataset
