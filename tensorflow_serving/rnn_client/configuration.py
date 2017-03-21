class Configuration(object):
    def __init__(self,
                 zones,
                 instance_types,
                 base_port,
                 client_port):
        """
        :param zones: The zones
        :param instance_types: The instance types
        :param base_port: port on which firse tf serving server is started
        :param client_port: port to which tf serving client should bind to
        """

        self.zones = zones
        self.instance_types = instance_types
        self.base_port = base_port
        self.client_port = client_port


def configuration_decoder(obj):
    """
    :param obj: A parsed JSON dictionary.
    :return: A Configuration object.
    """
    return Configuration(
        obj['zones'], obj['instance_types'], obj['base_port'], obj['client_port']
    )
