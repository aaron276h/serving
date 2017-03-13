import os.path


def format_test_file_string(folder, zone, instance_type, delta):
    """
    :param folder: The base folder of the test files
    :param zone: The zone of the test file
    :param instance_type: The instance type of the test file
    :param delta: The delta of the test file
    :return: The formatted path to the test file
    """
    return os.path.join(folder, 'testing_' + str(zone) + '_' + str(instance_type) + '_' + \
                        str(delta) + '.csv')


def format_training_file_string(folder, zone, instance_type, delta):
    """
    :param folder: The base folder of the training files
    :param zone: The zone of the training file
    :param instance_type: The instance type of the training file
    :param delta: The delta of the training file
    :return: The formatted path to the training file
    """
    return os.path.join(folder, 'training_' + str(zone) + '_' + str(instance_type) + '_' + \
                        str(delta) + '.csv')


def format_scaler_path_string(folder, zone='', instance_type=''):
    """
    :param folder: The folder path of the scaler
    :param zone: The zone of the data used for training
    :param instance_type: The instance type of the data used for training
    :return: The formatted path to the scaler
    """
    return os.path.join(folder, '_'.join(filter(bool, ['scaler', zone, instance_type])))


def format_model_path_string(folder, model_name, zone='', instance_type=''):
    """
    :param folder: The folder path of the model
    :param model_name: The name of the model
    :param zone: The zone of the data used for training
    :param instance_type: The instance type of the data used for training
    :return: The formatted path to the model
    """
    return os.path.join(folder, '_'.join(filter(bool, [model_name, zone, instance_type])))


def format_serving_export_path_string(folder, model_name, zone='', instance_type=''):
    """
    :param folder: The base folder of the model exports
    :param model_name: The name of the model
    :param zone: The zone of the training files
    :param instance_type: The instance type of the training files
    :return: The formatted path to the folder to export
    """
    return os.path.join(folder, '_'.join(filter(bool, [model_name, zone, instance_type])))
