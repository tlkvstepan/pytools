from torch import nn
import torch as th

from tools import network_tools


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
        nn.Conv2d(1, 2, (3, 3)),
        nn.Sequential(nn.Conv2d(1, 2, (3, 3)), nn.ReLU()))


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
