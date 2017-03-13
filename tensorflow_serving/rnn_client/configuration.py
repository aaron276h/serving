class Configuration(object):
    def __init__(self,
                 deltas,
                 zones,
                 instance_types,
                 unify_instances,
                 training_folders,
                 test_folders,
                 scaler_folder,
                 model_folder,
                 tf_serving_export_folder):
        """
        :param deltas: The deltas
        :param zones: The zones
        :param instance_types: The instance types
        :param unify_instances: Whether or not to use multiple instance types for training a single model
        :param training_folders: The paths to the training folders
        :param test_folders: The paths to the test folders
        :param scaler_folder: The path to the folder to export the scaler
        :param model_folder: The path to the folder to export the model
        :param tf_serving_export_folder: The path to the folder to export the tf serving model
        """

        self.deltas = deltas
        self.zones = zones
        self.instance_types = instance_types
        self.unify_instances = unify_instances
        self.training_folders = training_folders
        self.test_folders = test_folders
        self.scaler_folder = scaler_folder
        self.model_folder = model_folder
        self.tf_serving_export_folder = tf_serving_export_folder


def configuration_decoder(obj):
    """
    :param obj: A parsed JSON dictionary.
    :return: A Configuration object.
    """
    return Configuration(
        obj['deltas'], obj['zones'], obj['instance_types'], obj['unify_instances'],
        obj['training_folders'], obj['test_folders'], obj['scaler_folder'], obj['model_folder'],
        obj['tf_serving_export_folder']
    )
