# import functools
# import json
# import os
#
# from core.component import custom_object_detection_component
# from resources.object_detection.builders import dataset_builder, graph_rewriter_builder
# from resources.object_detection.builders import model_builder
# from resources.object_detection.legacy import trainer
# from resources.object_detection.utils import config_util
# import tensorflow as tf
# from google.protobuf import text_format
# from resources.object_detection import exporter
# from resources.object_detection.protos import pipeline_pb2
#
# custom_object_detection_component.generate_tfrecords()
#
#
#
# tf.logging.set_verbosity(tf.logging.INFO)
#
# flags = tf.app.flags
# flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
# flags.DEFINE_integer('task', 0, 'task id')
# flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
# flags.DEFINE_boolean('clone_on_cpu', False,
#                      'Force clones to be deployed on CPU.  Note that even if '
#                      'set to False (allowing ops to run on gpu), some ops may '
#                      'still be run on the CPU if they have no GPU kernel.')
# flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
#                      'replicas.')
# flags.DEFINE_integer('ps_tasks', 0,
#                      'Number of parameter server tasks. If None, does not use '
#                      'a parameter server.')
# flags.DEFINE_string('train_dir', '',
#                     'Directory to save the checkpoints and training summaries.')
#
# flags.DEFINE_string('pipeline_config_path', '',
#                     'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
#                     'file. If provided, other configs are ignored')
#
# flags.DEFINE_string('train_config_path', '',
#                     'Path to a train_pb2.TrainConfig config file.')
# flags.DEFINE_string('input_config_path', '',
#                     'Path to an input_reader_pb2.InputReader config file.')
# flags.DEFINE_string('model_config_path', '',
#                     'Path to a model_pb2.DetectionModel config file.')
#
# FLAGS = flags.FLAGS
#
#
#
# @tf.contrib.framework.deprecated(None, 'Use object_detection/model_main.py.')
# def main(_):
#   task = 0
#   num_clones = 1
#   clone_on_cpu = False
#
#   train_dir = '/tmp/'
#   pipeline_config_path = '/tmp/ssd_inception_v2_coco.config'
#   train_config_path = ''
#   input_config_path = ''
#   model_config_path = ''
#
#   assert train_dir, '`train_dir` is missing.'
#   if task == 0: tf.gfile.MakeDirs(train_dir)
#   if pipeline_config_path:
#     configs = config_util.get_configs_from_pipeline_file(
#         pipeline_config_path)
#     if task == 0:
#       tf.gfile.Copy(pipeline_config_path,
#                     os.path.join(train_dir, 'pipeline.config'),
#                     overwrite=True)
#   else:
#     configs = config_util.get_configs_from_multiple_files(
#         model_config_path=model_config_path,
#         train_config_path=train_config_path,
#         train_input_config_path=input_config_path)
#     if task == 0:
#       for name, config in [('model.config', model_config_path),
#                            ('train.config', train_config_path),
#                            ('input.config', input_config_path)]:
#         tf.gfile.Copy(config, os.path.join(train_dir, name),
#                       overwrite=True)
#
#   model_config = configs['model']
#   train_config = configs['train_config']
#   input_config = configs['train_input_config']
#
#   model_fn = functools.partial(
#       model_builder.build,
#       model_config=model_config,
#       is_training=True)
#
#   def get_next(config):
#     return dataset_builder.make_initializable_iterator(
#         dataset_builder.build(config)).get_next()
#
#   create_input_dict_fn = functools.partial(get_next, input_config)
#
#   env = json.loads(os.environ.get('TF_CONFIG', '{}'))
#   cluster_data = env.get('cluster', None)
#   cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
#   task_data = env.get('task', None) or {'type': 'master', 'index': 0}
#   task_info = type('TaskSpec', (object,), task_data)
#
#   # Parameters for a single worker.
#   ps_tasks = 0
#   worker_replicas = 1
#   worker_job_name = 'lonely_worker'
#   task = 0
#   is_chief = True
#   master = ''
#
#   if cluster_data and 'worker' in cluster_data:
#     # Number of total worker replicas include "worker"s and the "master".
#     worker_replicas = len(cluster_data['worker']) + 1
#   if cluster_data and 'ps' in cluster_data:
#     ps_tasks = len(cluster_data['ps'])
#
#   if worker_replicas > 1 and ps_tasks < 1:
#     raise ValueError('At least 1 ps task is needed for distributed training.')
#
#   if worker_replicas >= 1 and ps_tasks > 0:
#     # Set up distributed training.
#     server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
#                              job_name=task_info.type,
#                              task_index=task_info.index)
#     if task_info.type == 'ps':
#       server.join()
#       return
#
#     worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
#     task = task_info.index
#     is_chief = (task_info.type == 'master')
#     master = server.target
#
#   graph_rewriter_fn = None
#   if 'graph_rewriter_config' in configs:
#     graph_rewriter_fn = graph_rewriter_builder.build(
#         configs['graph_rewriter_config'], is_training=True)
#
#   trainer.train(
#       create_input_dict_fn,
#       model_fn,
#       train_config,
#       master,
#       task,
#       num_clones,
#       worker_replicas,
#       clone_on_cpu,
#       ps_tasks,
#       worker_job_name,
#       is_chief,
#       train_dir,
#       graph_hook_fn=graph_rewriter_fn)
#
#
#   tf.app.run()
#
#
#
#
# slim = tf.contrib.slim
# flags = tf.app.flags
#
# flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
#                     'one of [`image_tensor`, `encoded_image_string_tensor`, '
#                     '`tf_example`]')
# flags.DEFINE_string('input_shape', None,
#                     'If input_type is `image_tensor`, this can explicitly set '
#                     'the shape of this input tensor to a fixed size. The '
#                     'dimensions are to be provided as a comma-separated list '
#                     'of integers. A value of -1 can be used for unknown '
#                     'dimensions. If not specified, for an `image_tensor, the '
#                     'default shape will be partially specified as '
#                     '`[None, None, None, 3]`.')
# flags.DEFINE_string('pipeline_config_path', None,
#                     'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
#                     'file.')
# flags.DEFINE_string('trained_checkpoint_prefix', None,
#                     'Path to trained checkpoint, typically of the form '
#                     'path/to/model.ckpt')
# flags.DEFINE_string('output_directory', None, 'Path to write outputs.')
# flags.DEFINE_string('config_override', '',
#                     'pipeline_pb2.TrainEvalPipelineConfig '
#                     'text proto to override pipeline_config_path.')
# flags.DEFINE_boolean('write_inference_graph', False,
#                      'If true, writes inference graph to disk.')
# tf.app.flags.mark_flag_as_required('pipeline_config_path')
# tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
# tf.app.flags.mark_flag_as_required('output_directory')
# FLAGS = flags.FLAGS
#
#
# input_type = ''
# input_shape = None
# pipeline_config_path = None
# trained_checkpoint_prefix = None
# output_directory = None
# write_inference_graph = False
# config_override = ''
#
#
# def main(_):
#   pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
#   with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
#     text_format.Merge(f.read(), pipeline_config)
#   text_format.Merge(FLAGS.config_override, pipeline_config)
#   if FLAGS.input_shape:
#     input_shape = [
#         int(dim) if dim != '-1' else None
#         for dim in FLAGS.input_shape.split(',')
#     ]
#   else:
#     input_shape = None
#   exporter.export_inference_graph(
#       FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_prefix,
#       FLAGS.output_directory, input_shape=input_shape,
#       write_inference_graph=FLAGS.write_inference_graph)
#
#
# if __name__ == '__main__':
#   tf.app.run()
#
#
#
