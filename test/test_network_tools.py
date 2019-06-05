from torch import nn
import torch as th

from tools import network_tools


def test_gradient_reverse():
    x = th.rand(1)
    x.requires_grad = True
    y = network_tools.gradient_reverse(x)
    y.backward()
    assert x.grad.item() == -1


def _is_any_parameter_requires_gradient(network):
    for parameter in network.parameters():
        if parameter.requires_grad:
            return True
    return False


def test_save_module_io():
    network = nn.Sequential(nn.Linear(3, 10), nn.ReLU(), nn.Linear(10, 1))
    hook = network_tools.SaveModuleIO(network[0], is_save_input=False)
    network_input = th.Tensor(2, 3)
    network(network_input)
    network(network_input)
    tensors = hook.get_saved_tensors()
    assert len(tensors) == 2
    assert tensors[0].size() == (2, 10)
    hook.clean_saved_tensors()
    tensors = hook.get_saved_tensors()
    assert len(tensors) == 0


def test_set_requires_gradient_for_network():
    network = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
    network_tools.set_requires_gradient_for_network(
        network, is_requires_gradient=False)
    assert ~_is_any_parameter_requires_gradient(network)
    network_tools.set_requires_gradient_for_network(
        network, is_requires_gradient=True)
    assert _is_any_parameter_requires_gradient(network)


def test_convert_module():
    # Substitute ReLU by LeakyReLU
    network = nn.Sequential(
        nn.Conv2d(1, 2, (3, 3)), nn.ReLU(),
        nn.Sequential(nn.Conv2d(1, 2, (3, 3)), nn.ReLU()))
    network_tools.convert_module(network, nn.ReLU, lambda x: nn.LeakyReLU())
    assert not network_tools.is_module_inside_network(network, nn.ReLU)


def test_are_networks_have_same_modules():
    assert network_tools.are_networks_have_same_modules(
        nn.Conv2d(1, 2, (3, 3)), nn.Conv2d(1, 2, (3, 3)))
    assert not network_tools.are_networks_have_same_modules(
        nn.Conv2d(1, 2,
                  (3, 3)), nn.Sequential(nn.Conv2d(1, 2, (3, 3)), nn.ReLU()))


def test_append_to_network():
    # Add function that extracts embedding from the center of a patch.
    net = network_tools.AppendOperationToNetwork(
        nn.Conv2d(2, 3, kernel_size=3, padding=1),
        operation=lambda x: th.mean(x))
    input = th.rand(1, 2, 3, 5)
    output = net(input)
    print(output)
    assert output.numel() == 1


def test_remove_module():
    network = nn.ModuleList([
        nn.Sequential(nn.Linear(10, 10), nn.InstanceNorm1d(10)),
        nn.InstanceNorm1d(10)
    ])
    network_tools.remove_module(network, nn.InstanceNorm1d)
    assert not network_tools.is_module_inside_network(network,
                                                      nn.InstanceNorm1d)
