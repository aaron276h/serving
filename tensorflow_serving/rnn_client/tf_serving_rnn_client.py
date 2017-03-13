#!/usr/bin/env python2.7

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import pickle
import os
import configuration
import json
import path_formatter

from data_fetcher import DataFetcher

tf.app.flags.DEFINE_integer('concurrency', 10, 'Maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('conf', '', 'Configuration file path')
FLAGS = tf.app.flags.FLAGS


class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self, count):
        with self._condition:
            self._error += count

    def inc_done(self, count):
        with self._condition:
            self._done += count
            self._condition.notify()

    def dec_active(self, count):
        with self._condition:
            self._active -= count
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result_counter):
    """Creates RPC callback function.
    Args:
        label: The correct label for the predicted example.
        result_counter: Counter for the prediction result.
    Returns:
        The callback function.
    """
    def _callback(result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
            result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        count = label.shape[0]
        if exception:
            result_counter.inc_error(1)
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            result = result_future.result().outputs['prediction']

            response = numpy.array(result.float_val).reshape((result.tensor_shape.dim[0].size, result.tensor_shape.dim[1].size))

            prediction = numpy.argmax(response, axis=1)

            error_count = numpy.sum(numpy.not_equal(label, prediction))
            result_counter.inc_error(error_count)

        result_counter.inc_done(count)
        result_counter.dec_active(count)
    return _callback


def do_inference(hostport, concurrency, testing_files, scaler):
    """
    :param hostport: The host-port of the server
    :param concurrency: The number of concurrent requests made to the server
    :param testing_files: The files used for testing
    :param scaler: The scaler to use to normalize features
    :return: The inference error
    """
    data_fetcher_test = DataFetcher(filepaths=testing_files, is_train=False, scaler=scaler,
                                    unify_deltas=True, unify_instances=False)

    x_testing, y_testing = data_fetcher_test.fetch_all_classified()
    num_tests = len(x_testing)

    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)

    count = 0
    batch_size = 10000

    print("Running tests on " + str(num_tests) + " examples......")

    while count < num_tests:
        batch_end = min(count + batch_size, num_tests)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'ec2pred_mlp'
        request.inputs['input'].CopyFrom(
                tf.contrib.util.make_tensor_proto(x_testing[count:batch_end], shape=x_testing[count:batch_end].shape, dtype=tf.float32))
        result_counter.throttle()
        result_future = stub.Predict.future(request, 50000.0)  # 5 seconds
        result_future.add_done_callback(
                _create_rpc_callback(numpy.argmax(y_testing[count:batch_end], axis=1), result_counter))
        count += batch_size

    return result_counter.get_error_rate()


def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return

    if not FLAGS.conf:
        conf_file_path = os.path.join(os.getcwd(), 'conf', os.path.splitext(__file__)[0] + '.json')
        print("User did not provide conf file, using the default path " + conf_file_path)
    else:
        conf_file_path = FLAGS.conf

    with open(conf_file_path) as conf_file:
        conf = json.loads(conf_file.read(), object_hook=configuration.configuration_decoder)

    # Load the testing data
    for zone in conf.zones:
        testing_files = []
        for instance_type in conf.instance_types:
            if not conf.unify_instances:
                testing_files = []

            for delta in conf.deltas:
                for test_folder in conf.test_folders:
                    testing_files.append(
                        path_formatter.format_test_file_string(test_folder, zone, instance_type, delta)
                    )

            if not conf.unify_instances:
                scaler = pickle.load(open(path_formatter.format_scaler_path_string(
                    conf.scaler_folder, zone, instance_type), 'r'))
                error_rate = do_inference(FLAGS.server, FLAGS.concurrency, testing_files, scaler)
                print('\nInference error rate for instance %s, zone %s: %s%%' %
                      (instance_type, zone, error_rate * 100))


if __name__ == '__main__':
    tf.app.run()
