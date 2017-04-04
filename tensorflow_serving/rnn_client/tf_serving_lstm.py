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

# Build script will need to specify symbolic link
import generated_sources.to_tensorflow_serving_pb2 as to_tensorflow_serving_pb2
import generated_sources.from_tensorflow_serving_pb2 as from_tensorflow_serving_pb2

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


def do_inference(host, port, concurrency, feature_vectors):
    """
    :param host: The IP of the server (usually localhost)
    :param port: The port of the model
    :param concurrency: The number of concurrent requests made to the server
    :param feature_vectors: Feature vector already scaled
    :return: Prediction
    """
    # Assuming one look up at a time for now
    feature_vector = feature_vectors.reshape(1, 10, 12)

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ec2pred_mlp'
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(feature_vector, shape=feature_vector.shape, dtype=tf.float32))

    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result = result_future.result().outputs['prediction']
    response = numpy.array(result.float_val).reshape((result.tensor_shape.dim[0].size, result.tensor_shape.dim[1].size))

    return response


def build_feature_vector(features):
    """ Build np feature vector from Proto message

    :param features: to_tensorflow_serving_pb2.features
    :return:
    """
    feature_vector = np.array([features.hour,
                               features.market_price,
                               features.day_of_week,
                               features.is_weekend,
                               features.is_holiday,
                               features.us_east_tod,
                               features.us_central_tod,
                               features.us_west_tod,
                               features.price_diff_2,
                               features.price_diff_10,
                               features.price_diff_30,
                               features.delta])
    return feature_vector


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

    # Ports are specified in config file
    # Ports for Client to receive inference queries is 8000
    # Forwards queries to servers which have ports 9000+
    # The + is zone.index * number of instance types + instance_type.index
    # if unified_instances is true, than port is just zone.index

    # Start ZMQ server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % conf.client_port)

    while True:
        # Get message from BidBrain / simulator
        message = socket.recv()

        # Parse query message
        parsed_message = to_tensorflow_serving_pb2.InferenceQuery()
        parsed_message.ParseFromString(message)

        # Build response message
        response_message = from_tensorflow_serving_pb2.InferenceResults()

        # TODO: combine queries for servers
        for query in parsed_message.model_input:
            zone_index = conf.zones.index(query.zone)
            instance_index = conf.instance_types.index(query.instance_type)

            feature_vectors = np.zeros((1,10,12))
            for index, time_step in enumerate(query.feature_inputs.features):
                feature_vectors[0][index] = build_feature_vector(time_step)

            server_port = conf.base_port + len(conf.instance_types) * zone_index + instance_index
            probability_of_eviction = do_inference(host=FLAGS.server,
                                                   port=server_port,
                                                   concurrency=FLAGS.concurrency,
                                                   feature_vectors=feature_vectors)[0][0]

            query_result = response_message.predictions.add()
            query_result.probability_of_eviction = probability_of_eviction
            query_result.id = query.id

        # Send response
        response_serialized = response_message.SerializeToString()
        socket.send(response_serialized)

if __name__ == '__main__':
    tf.app.run()
