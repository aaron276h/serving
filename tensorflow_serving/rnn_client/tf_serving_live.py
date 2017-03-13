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
import zmq
import numpy as np

from data_fetcher import DataFetcher
from serialize import deserialize

tf.app.flags.DEFINE_integer('concurrency', 10, 'Maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host')
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
            print (response)

            prediction = numpy.argmax(response, axis=1)

            error_count = numpy.sum(numpy.not_equal(label, prediction))
            result_counter.inc_error(error_count)

        result_counter.inc_done(count)
        result_counter.dec_active(count)
    return _callback


def do_inference(host, port, concurrency, feature_vector):
    """
    :param host: The IP of the server (usually localhost)
    :param port: The port of the model
    :param concurrency: The number of concurrent requests made to the server
    :param feature_vector: Feature vector already scaled
    :return: Prediction
    """
    # Assuming one look up at a time for now
    feature_vector = feature_vector.reshape(1, -1)

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ec2pred_mlp'
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(feature_vector, shape=feature_vector.shape, dtype=tf.float32))

    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result = result_future.result().outputs['prediction']
    response = numpy.array(result.float_val).reshape((result.tensor_shape.dim[0].size, result.tensor_shape.dim[1].size))
    print (response)

    return response


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

    # Ports for ZMQ si 8000 and 9000+ for inference
    # The + is zone.index * number of instance types + instance_type.index
    # if unified_instances is true, than port is just zone.index

    # Start ZMQ server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:8000")

    # Load classification scalers
    classification_scalers = []
    for zone in conf.zones:
        if conf.unify_instances is not True:
            for instance_type in conf.instance_types:
                classification_scalers.append(pickle.load(open(path_formatter.format_scaler_path_string(
                    conf.scaler_folder, zone, instance_type), 'r')))
        else:
            assert 0

    while True:
        print ('Waiting for messages ...')

        # Get message from BidBrain / simulator
        message = socket.recv()

        print('Got message ...')

        # De-serialize the message
        zone_index, instance_index, feature_vector = deserialize(msg=message, unify_instances=conf.unify_instances)

        assert conf.unify_instances is not True

        # Find the index of the scaler
        model_index = zone_index * len(conf.instance_types) + instance_index
        assert model_index < len(classification_scalers)
        
        # Scale the features
        feature_vector = classification_scalers[model_index].transform(feature_vector)

        # Compute port of the model you need to connect to
        port = 9000 + model_index
        if conf.unify_instances is True:
            port = 9000 + zone_index

        prediction = do_inference(host=FLAGS.server, port=port, concurrency=FLAGS.concurrency,
                                  feature_vector=feature_vector)

        # Send the probability of eviction
        socket.send(str(prediction[0][0]))

if __name__ == '__main__':
    tf.app.run()
