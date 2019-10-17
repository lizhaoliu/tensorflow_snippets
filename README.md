# Tensorflow Snippets From the Field.

## Table of Contents
- [System Setup](#system-setup)
- [On Convolutions](#on-convolutions)
- [Indexing](#indexing)
- [Numerical Stability](#numerical-stability)
- [Shapes](#shapes)
- [Tensor Contraction (More Generalized Matrix Multiplication)](#tensor-contraction-more-generalized-matrix-multiplication)
- [```tf.estimator``` API](#tfestimator-api)
- [Load A saved_model and Run Inference (in Python)](#load-a-saved_model-and-run-inference-in-python)
- [Input Features! ```tf.train.Example``` and ```tf.train.SequenceExample```](#input-features-tftrainexample-and-tftrainsequenceexample)
- [Misc](#misc)

#### System Setup
- Install CUDA 10.0 on Ubuntu 18.04 LTS GPU server:
```sh
# 1. Install NVIDIA driver either through "Additional Drivers", or:
$ sudo apt install --no-install-recommends nvidia-driver-430
# Reboot and then check that GPUs are visible using the command: nvidia-smi.

# 2. Add NVIDIA package repositories
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo apt update
$ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
$ sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
$ sudo apt update

# 3. Install development and runtime libraries.
$ sudo apt install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.6.2.24-1+cuda10.0  \
    libcudnn7-dev=7.6.2.24-1+cuda10.0

# 4. Install TensorRT. Requires that libcudnn7 is installed above.
$ sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 \
    libnvinfer-dev=5.1.5-1+cuda10.0
```
- Install Tensorflow 2.0 GPU version:
```sh
$ python3 -m pip install --upgrade pip
$ python3 -m pip install --user tensorflow-gpu
```

- Setup SSH server on GPU server:
  * Install OpenSSH server: `$ sudo apt install openssh-server`.
  * Add port forwarding rule for port 22.
  
- Setup SSH client on our ultra book:
  * Create SSH key: `$ ssh-keygen -t rsa -b 4096`.
  * Install SSH key on the GPU server as an authorized key: `$ ssh-id-copy <user>@<server-ip>`.
  * Now we can connect to the GPU server by: `$ ssh -i <ssh-key> <user>@<server-ip>`.
  
- Add [PyCharm remote Python interpreter](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html) on GPU server via SSH.

- Happy machine learning!

#### On Convolutions
- Typically there are two options for ```padding```:
  * ```SAME```: Make sure result has *same* spatial shape as input tensor, this often requires padding 0's to input tensor.
  * ```VALID```: No paddings please, only use *valid* points . Result can have different spatial shape.
- Tensorflow by default performs *centered* convolution (kernel is centered around current point). With ```k```: kernel size, ```d```: dilation rate, then extended kernel size ```k' = d * (k - 1) + 1```. For each spatial dimension, convolution at index ```i``` is computed using points between indices (inclusive) ```[i - (k' - 1) // 2, i + k' // 2]```. 
- *Causal* convolution uses points between indices ```[i - d * (k - 1), i]```, a simple solution is to pad ```d * (k - 1)``` of 0's at the beginning of that dimension then perform a normal convolution with ```VALID``` padding.
- Convolution kernel has shape ```[spatial_dim[0], ..., spatial_dim[n - 1], num_input_channels, num_output_channels]```. For each output channel ```k```, ```output[..., k] = sum_over_i {input[..., i] * kernel[..., i, k]}```, here ```*``` is convolution operator.

#### Indexing
- ```tf.gather_nd(params, indices)``` retrieves slices from ```params``` by ```indices```. The rule is simple: *only the last dimension of ```indices``` does slice ```params```, and that dimension is "replaced" with those slices*. It's easy to see that:
  * ```indices.shape[-1] <= rank(params)```: The last dimension of ```indices``` must be no greater than the rank of ```params```.
  * Result tensor shape is ```indices.shape[:-1] + params.shape[indices.shape[-1]:]```, example:
  ```python
  # params has shape [4, 5, 6].
  params = tf.reshape(tf.range(0, 120), [4, 5, 6])
  # indices has shape [3, 2].
  indices = tf.constant([[2, 3], [0, 1], [1, 2]], dtype=tf.int32)
  # slices has shape [3, 6].
  slices = tf.gather_nd(params, indices)
  ```
- ```tf.gather_nd``` and Numpy fancy indexing: ```x[indices]``` == ```tf.gather_nd(x, zip(*indices))```; ```tf.gather_nd(x, indices)``` == ```x[zip(*indices)]```. Where ```x``` is Numpy array and ```indices``` is indexing array (dim > 1).

#### Numerical Stability
- ```Inf``` morphs to ```NaN``` while plugged into back-prop (chain rule).
- Watch out! ```tf.where``` Can Spawn NaN in Gradients
If either branch in ```tf.where``` contains Inf/NaN then it produces NaN in gradients, e.g.:
```python
log_s = tf.constant([-100., 100.], dtype=tf.float32)
# Computes 1.0 / exp(log_s), in a numerically robust way.
inv_s = tf.where(log_s >= 0.,
                 tf.exp(-log_s),  # Creates Inf when -log_s is large.
                 1. / (tf.exp(log_s) + 1e-6))  # tf.exp(log_s) is Inf with large log_s.
grad_log_s = tf.gradients(inv_s, [log_s])
with tf.Session() as sess:
    inv_s, grad_log_s = sess.run([inv_s, grad_log_s])
    print(inv_s)  # [  1.00000000e+06   3.78350585e-44]
    print(grad_log_s)  # [array([ nan,  nan], dtype=float32)]
```

#### Shapes
- ```tensor.shape``` returns tensor's static shape, while the graph is being built.
- ```tensor.shape.as_list()``` returns the static shape as a integer list.
- ```tensor.shape[i].value``` returns the static shape's i-th dimension size as an integer.
- ```tf.shape(t)``` returns t's run-time shape as a tensor.
- An example:
```python
x = tf.placeholder(tf.float32, shape=[None, 8]) # x shape is non-deterministic while building the graph.
print(x.shape) # Outputs static shape (?, 8).
shape_t = tf.shape(x)
with tf.Session() as sess:
    print(sess.run(shape_t, feed_dict={x: np.random.random(size=[4, 8])})) # Outputs run-time shape (4, 8).
```
- [] (empty square brackets) as a shape denotes a scalar (0 dim). E.g. tf.FixedLenFeature([], ..) is a scalar feature.
- Broadcasting on two arrays starts with the _trailing_ dimensions, and works its way _backward_ to the leading dimensions. E.g. 
```
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```

#### Tensor Contraction (More Generalized Matrix Multiplication)
```python
# Matrix multiplication
tf.einsum('ij,jk->ik', m0, m1)  # output[i, k] = sum_j m0[i, j] * m1[j, k]
# Dot product
tf.einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]
# Outer product
tf.einsum('i,j->ij', u, v)  # output[i, j] = u[i]*v[j]
# Transpose
tf.einsum('ij->ji', m)  # output[j, i] = m[i,j]
# Batch matrix multiplication
tf.einsum('aij,jk->aik', s, t)  # out[a, i, k] = sum_j s[a, i, j] * t[j, k]
# Batch tensor contraction
tf.einsum('nhwc,nwcd->nhd', s, t)  # out[n, h, d] = sum_w_c s[n, h, w, c] * t[n, w, c, d]
```

#### ```tf.estimator``` API
- A typical input_fn (used for train/eval) for tf.estimator API:
```python
def make_input_fn(mode, ...):
    """Return input_fn for train/eval in tf.estimator API.

    Args:
        mode: Must be tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.EVAL.
        ...
    Returns:
        The input_fn.
    """
    def _input_fn():
        """The input function.

        Returns:
            features: A dict of {'feature_name': feature_tensor}.
            labels: A tensor of labels.
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            features = ...
            labels = ...
        elif mode == tf.estimator.ModeKeys.EVAL:
            features = ...
            labels = ...
        else:
            raise ValueError(mode)
        return features, labels

    return _input_fn
```

- A typical model_fn for tf.estimator API:
```python
def make_model_fn(...):
    """Return model_fn to build a tf.estimator.Estimator.

    Args:
        ...
    Returns:
        The model_fn.
    """
    def _model_fn(features, labels, mode):
        """Model function.

        Args:
            features: The first item returned from the input_fn for train/eval, a dict of {'feature_name': feature_tensor}. If mode is ModeKeys.PREDICT, same as in serving_input_receiver_fn.
            labels: The second item returned from the input_fn, a single Tensor or dict. If mode is ModeKeys.PREDICT, labels=None will be passed.
            mode: Optional. Specifies if this training, evaluation or prediction. See ModeKeys.
        """
        if mode == tf.estimator.ModeKeys.PREDICT:
            # Calculate the predictions.
            predictions = ...
            # For inference/prediction outputs.
            export_outputs = {
                tf.saved_model.signature_constants.PREDICT_METHOD_NAME: tf.estimator.export.PredictOutput({
                    'output_1': predict_output_1,
                    'output_2': predict_output_2,
                    ...
                }),
            }
            ...
        else:
            predictions = None
            export_outputs = None

        if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
            loss = ...
        else:
            loss = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = ...
            # Can use tf.group(..) to group multiple train_op as a single train_op.
        else:
            train_op = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            export_outputs=export_outputs)

    return _model_fn
```

- Use tf.estimator.Estimator to export a saved_model:
```python
# serving_features must match features in model_fn when mode == tf.estimator.ModeKeys.PREDICT.
serving_features = {'serving_input_1': tf.placeholder(...), 'serving_input_2': tf.placeholder(...), ...}
estimator.export_savedmodel(export_dir,
                            tf.estimator.export.build_raw_serving_input_receiver_fn(serving_features))
```

- Use tf.contrib.learn.Experiment to export a saved_model:
```python
# serving_features must match features in model_fn when mode == tf.estimator.ModeKeys.PREDICT.
serving_features = {'serving_input_1': tf.placeholder(...), 'serving_input_2': tf.placeholder(...), ...}
export_strategy = tf.contrib.learn.utils.make_export_strategy(tf.estimator.export.build_raw_serving_input_receiver_fn(serving_features))
expriment = tf.contrib.learn.Experiment(..., export_strategies=[export_strategy], ...)
```

#### Load A saved_model and Run Inference (in Python)
```python
with tf.Session(...) as sess:
    # Load saved_model MetaGraphDef from export_dir.
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

    # Get SignatureDef for serving (here PREDICT_METHOD_NAME is used as export_outputs key in model_fn).
    sigs = meta_graph_def.signature_def[tf.saved_model.signature_constants.PREDICT_METHOD_NAME]

    # Get the graph for retrieving input/output tensors.
    g = tf.get_default_graph()

    # Retrieve serving input tensors, keys must match keys defined in serving_features (when building input receiver fn).
    input_1 = g.get_tensor_by_name(sigs.inputs['input_1'].name)
    input_2 = g.get_tensor_by_name(sigs.inputs['input_2'].name)
    ...

    # Retrieve serving output tensors, keys must match keys defined in ExportOutput (e.g. PredictOutput) in export_outputs.
    output_1 = g.get_tensor_by_name(sigs.outputs['output_1'].name)
    output_2 = g.get_tensor_by_name(sigs.outputs['output_2'].name)
    ...

    # Run inferences.
    outputs_values = sess.run([output_1, output_2, ...], feed_dict={input_1: ..., input_2: ..., ...})
```

#### Input Features! ```tf.train.Example``` and ```tf.train.SequenceExample```
- A [tf.train.Example](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/core/example/example.proto#L88) is roughly a map of *{feature_name: value_list}*.
- A [tf.train.SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/core/example/example.proto#L292) is roughly a ```tf.train.Example``` plus a map of *{feature_name: list_of_value_lists}*.
- Build a tf.train.Example in Python:
```python
# ==================== Build in one line ====================
example = tf.train.Example(features=tf.train.Features(feature={
    'bytes_values': tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[bytes_feature])),
    'float_values': tf.train.Feature(
        float_list=tf.train.FloatList(value=[float_feature])),
    'int64_values': tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int64_feature])),
    ...
}))
# ==================== OR progressivly ====================
example = tf.train.Example()
example.features.feature['bytes_feature'].bytes_list.value.extend(bytes_values)
example.features.feature['float_feature'].float_list.value.extend(float_values)
example.features.feature['int64_feature'].int64_list.value.extend(int64_values)
...
```
- Build a tf.train.SequenceExample in Python:
```python
sequence_example = tf.train.SequenceExample()

# Populate context data.
sequence_example.context.feature[
    'context_bytes_values_1'].bytes_list.value.extend(bytes_values)
sequence_example.context.feature[
    'context_float_values_1'].float_list.value.extend(float_values)
sequence_example.context.feature[
    'context_int64_values_1'].int64_list.value.extend(int64_values)
...

# Populate sequence data.
feature_list_1 = sequence_example.feature_lists.feature_list['feature_list_1']
# Add tf.train.Feature to feature_list_1.
feature_1 = feature_list_1.feature.add()
# Populate feature_1, e.g. feature_1.float_list.value.extend(float_values)
# Add tf.train.Feature to feature_list_1, if any.
...
```

- To parse a SequenceExample:
```python
tf.parse_single_sequence_example(serialized,
    context_features={
        'context_feature_1': tf.FixedLenFeature([], dtype=...),
        ...
    },
    sequence_features={
        # For 'sequence_features_1' shape, [] results with [?] and [k] results with [?, k], where:
        # ?: timesteps, i.e. number of tf.Train.Feature in 'sequence_features_1' list, can be variable.
        # k: number of elements in each tf.Train.Feature in 'sequence_features_1'.
        'sequence_features_1': tf.FixedLenSequenceFeature([], dtype=...),
        ...
    },)
```

- Write tfrecords to sharded files. Reading data from multiple input files can increase I/O throughput in TF:
```python
import multiprocessing
from concurrent import futures

def write_tf_records(file_path, tf_records):
    """Writes TFRecord (Example or SequenceExample) data to output file.

    :param file_path: the full output file path, can add `@N` at the end that
        specifies total number of shards, e.g. /data/training.tfrecords@10.
    :param tf_records: a list or tuple of `tf.train.Example` or
        `tf.train.SequenceExample` instances.
    """

    def _write_data_to_file(file_path, tf_records):
        """Writes data into specified output file path.

        :param file_path: full path of output file.
        :param tf_records: a list of Example or SequenceExample instances.
        """
        with tf.python_io.TFRecordWriter(file_path) as writer:
            for tf_record in tf_records:
                writer.write(tf_record.SerializeToString())

    def _get_shards_paths(file_path):
        """Gets (shard) file paths by parsing from provided file path.

        :param file_path: file path that may contain shard syntax "@N".
        :return: a list of file paths.
        """
        shard_char_idx = file_path.rfind('@')

        # No shards specified.
        if shard_char_idx == -1:
            return [file_path]

        num_shards = int(file_path[shard_char_idx + 1:])
        if num_shards <= 0:
            raise ValueError('Number of shards must be a positive integer.')
        prefix = file_path[:shard_char_idx]
        return ['{}-{}-of-{}'.format(prefix, i, num_shards) for i
                in range(num_shards)]

    if not isinstance(tf_records, list) and not isinstance(tf_records, tuple):
        raise TypeError('tf_records must be a list or tuple.')

    tf_records = list(tf_records)
    shards_paths = _get_shards_paths(file_path)

    if len(shards_paths) > len(tf_records):
        raise ValueError('More data than file shards.')

    with futures.ThreadPoolExecutor(
            max_workers=multiprocessing.cpu_count() - 1) as executor:
        for shard_id, file_path in enumerate(shards_paths):
            executor.submit(_write_data_to_file, file_path,
                            tf_records[shard_id::len(shards_paths)])
```

#### Misc
- Don't Forget to Reset Default Graph in Jupyter Notebook
If you forgot to reset default Tensorflow graph (or create a new graph) in a Jupyter notebook cell, and run that cell for a few times then you may get weird results.
- Visualize Tensorflow Graph in Jupyter Notebook
```python
import numpy as np
from IPython import display

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped {} bytes>".format(size)
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph
        -basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display.display(display.HTML(iframe))
```
Then call ```show_graph(tf.get_default_graph())``` to show in your Jupyter/IPython notebook.
