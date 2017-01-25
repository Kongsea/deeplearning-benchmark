# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to train Inception using multiple replicas with synchronous update.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import sys
import time

import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")


# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


def _tower_loss(images, labels, num_classes, scope, reuse):
  """Calculate the total loss on a single tower running the ImageNet model.
  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
  then each tower will operate on an batch of 16 images.
  Args:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
    num_classes: number of classes
    scope: unique prefix string identifying the ImageNet tower, e.g.
      'tower_0'.
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # When fine-tuning a model, we do not restore the logits but instead we
  # randomly initialize the logits. The number of classes in the output of the
  # logit is the number of classes in specified Dataset.
  restore_logits = False

  # Build inference Graph.
  with tf.variable_scope('inference', reuse=reuse):
    logits = inception.inference(images, num_classes, for_training=True,
                                 restore_logits=restore_logits,
                                 scope=scope)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  split_batch_size = images.get_shape().as_list()[0]
  inception.loss(logits, labels, batch_size=split_batch_size)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def _average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat_v2(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train(target, dataset, cluster_spec):
  """Train Inception on a dataset for a number of steps."""
  # Number of workers and parameter servers are infered from the workers and ps
  # hosts string.
  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])
  # If no value is given, num_replicas_to_aggregate defaults to be the number of
  # workers.
  if FLAGS.num_replicas_to_aggregate == -1:
    num_replicas_to_aggregate = num_workers
  else:
    num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

  # Both should be greater than 0 in a distributed training.
  assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                         'num_parameter_servers'
                                                         ' must be > 0.')

  # Choose worker 0 as the chief. Note that any worker could be the chief
  # but there should be only one chief.
  is_chief = (FLAGS.task_id == 0)

  # Ops are assigned to worker by default.
  with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % FLAGS.task_id,
      cluster=cluster_spec)):
    # Variables and its related init/assign ops are assigned to ps.
    # with tf.device('/job:worker/task:%d' % FLAGS.task_id):
    # with slim.scopes.arg_scope(
    #     [slim.variables.variable, slim.variables.global_step],
    #     device=slim.variables.VariableDeviceChooser(num_parameter_servers)):
      # Create a variable to count the number of train() calls. This equals the
      # number of updates applied to the variables.
      global_step = slim.variables.global_step()

      assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
          'Batch size must be divisible by number of GPUs')

      # Calculate the learning rate schedule.
      num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                               FLAGS.batch_size)
      # Decay steps need to be divided by the number of replicas to aggregate.
      decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay /
                        num_replicas_to_aggregate)

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True)
      # Add a summary to track the learning rate.
      # tf.scalar_summary('learning_rate', lr)

      # Create an optimizer that performs gradient descent.
      opt = tf.train.RMSPropOptimizer(lr,
                                      RMSPROP_DECAY,
                                      momentum=RMSPROP_MOMENTUM,
                                      epsilon=RMSPROP_EPSILON)
      num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
      images, labels = image_processing.distorted_inputs(
          dataset,
          batch_size=FLAGS.batch_size,
          num_preprocess_threads=num_preprocess_threads)

      # Number of classes in the Dataset label set plus 1.
      # Label 0 is reserved for an (unused) background class.
      num_classes = dataset.num_classes() + 1

      images_splits = tf.split(images, FLAGS.num_gpus, 0)
      labels_splits = tf.split(labels, FLAGS.num_gpus, 0)

      tower_grads = []
      reuse_variables = None
      with tf.variable_scope('model'):
        for i in xrange(FLAGS.num_gpus):
	    with tf.device('/gpu:%d' % i):
	      with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)):
	        loss = _tower_loss(images_splits[i], labels_splits[i], num_classes, None, reuse_variables)

	      grads = opt.compute_gradients(loss)

	      tower_grads.append(grads)
              reuse_variables = True

      # We must calculate the mean of each gradient. Note that this is the
      # synchronization point across all towers.
      grads = _average_gradients(tower_grads)

      # Add a summaries for the input processing and global_step.
      # summaries.extend(input_summaries)

      # Add a summary to track the learning rate.
      # summaries.append(tf.scalar_summary('learning_rate', lr))

      # Add histograms for gradients.
      # for grad, var in grads:
      #   if grad is not None:
      #     summaries.append(
      #         tf.histogram_summary(var.op.name + '/gradients', grad))

      # Apply the gradients to adjust the shared variables.

      # Add histograms for model variables.
      # for var in variables_to_average:
      #    tf.histogram_summary(var.op.name, var)

      # Create synchronous replica optimizer.
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=num_replicas_to_aggregate,
          total_num_replicas=num_workers)

      train_op = opt.apply_gradients(grads, global_step=global_step)
      # Get chief queue_runners, init_tokens and clean_up_op, which is used to
      # synchronize replicas.
      # More details can be found in sync_replicas_optimizer.
      chief_queue_runners = [opt.get_chief_queue_runner()]
      init_tokens_op = opt.get_init_tokens_op()

      # Create a saver.
      saver = tf.train.Saver()

      # Build the summary operation based on the TF collection of Summaries.
      # summary_op = tf.merge_all_summaries()

      # Build an initialization operation to run below.
      init_op = tf.global_variables_initializer()

      # We run the summaries in the same thread as the training operations by
      # passing in None for summary_op to avoid a summary_thread being started.
      # Running summaries and training operations in parallel could run out of
      # GPU memory.
      sv = tf.train.Supervisor(is_chief=is_chief,
                               logdir=FLAGS.train_dir,
                               init_op=init_op,
                               summary_op=None,
                               global_step=global_step,
                               saver=saver,
                               save_model_secs=FLAGS.save_interval_secs)

      tf.logging.info('%s Supervisor' % datetime.now())

      sess_config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=True)

      # Get a session.
      sess = sv.prepare_or_wait_for_session(target, config=sess_config)

      # Start the queue runners.
      queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
      sv.start_queue_runners(sess, queue_runners)
      tf.logging.info('Started %d queues for processing input data.',
                      len(queue_runners))

      if is_chief:
        sv.start_queue_runners(sess, chief_queue_runners)
        sess.run(init_tokens_op)

      # Train, checking for Nans. Concurrently run the summary operation at a
      # specified interval. Note that the summary_op and train_op never run
      # simultaneously in order to prevent running out of GPU memory.
      next_summary_time = time.time() + FLAGS.save_summaries_secs
      profile_step = 60
      step = 0
      while not sv.should_stop() and step <= 2000:
        try:
          start_time = time.time()
          loss_value, step = sess.run([train_op, global_step])
          duration = time.time() - start_time
          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          if step > FLAGS.max_steps:
            break

          # TODO(xpan): Is the time sec? Seems to accurate?
          examples_per_sec = FLAGS.batch_size / float(duration)
          format_str = ('Worker %d: %s: step %d, loss = %.2f'
                        '(%.1f examples/sec; %.3f  sec/batch)')
          if step >= 10 and step != profile_step+1:
            tf.logging.info(format_str %
                            (FLAGS.task_id, datetime.now(), step, loss_value,
                             examples_per_sec, duration))
          else:
            tf.logging.info('Not considering step %d (%.1f samples/sec)' %
                            (step, examples_per_sec))
        except:
          if is_chief:
            tf.logging.info('About to execute sync_clean_up_op!')
          raise

      # Stop the supervisor.  This also waits for service threads to finish.
      sv.stop()

      # Save after the training ends.
      if is_chief:
        saver.save(sess,
                   os.path.join(FLAGS.train_dir, 'model.ckpt'),
                   global_step=global_step)
