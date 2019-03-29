from torch import nn


class AppendOperationToNetwork(nn.Module):
    """Appends operation to the end / beggining of network."""

    def __init__(self, network, operation, append_after_network=True):
        super(AppendOperationToNetwork, self).__init__()
        self._network = network
        self._operation = operation
        self._append_after_network = append_after_network

    def forward(self, *args):
        if self._append_after_network:
            network_output = self._network(*args)
            return self._operation(network_output)
        operation_output = self._operation(*args)
        return self._network(operation_output)


def compute_number_of_parameters(network):
    number_of_parameters = 0
    for parameter in network.parameters():
        number_of_parameters += parameter.numel()
    return number_of_parameters


def remove_module(network, module_to_remove):
    """Removes module from nn.Sequential and nn.ModuleList."""
    if isinstance(network, nn.Sequential) or \
       isinstance(network, nn.ModuleList):
        no_changes = True
        while no_changes:
            no_changes = False
            for index in range(0, len(network)):
                if isinstance(network[index], module_to_remove):
                    del network[index]
                    no_changes = True
                    break
    for child_network in network.children():
        remove_module(child_network, module_to_remove)


def are_networks_have_same_modules(reference_network, source_network):
    source_modules = list(source_network.modules())
    for module_index, module in enumerate(reference_network.modules()):
        if type(module) != type(source_modules[module_index]):
            return False
    return True


def is_module_inside_network(network, module):
    for network_module in network.modules():
        if isinstance(network_module, module):
            return True
    return False


def convert_module(network, module_to_substitute, conversion_function):
    """Apply "conversion function" to every "module"."""
    if isinstance(network, nn.Sequential) or \
       isinstance(network, nn.ModuleList):
        for index in range(0, len(network)):
            if isinstance(network[index], module_to_substitute):
                network[index] = conversion_function(network[index])
            else:
                for child_network in network[index].children():
                    convert_module(child_network, module_to_substitute,
                                   conversion_function)
    else:
        for child_network in network.children():
            convert_module(child_network, module_to_substitute,
                           conversion_function)
