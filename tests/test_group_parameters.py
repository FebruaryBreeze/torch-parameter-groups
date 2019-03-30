import unittest

import torch
import torch.nn as nn

from torch_parameter_groups import GroupRule, group_parameters


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x: torch.Tensor):
        pass


class MyTestCase(unittest.TestCase):
    def test_group_parameters(self):
        rules = [
            GroupRule.factory(config={
                'module_type_list': [
                    'Conv2d'
                ],
                'param_name_list': [
                    'weight'
                ]
            }),
            GroupRule.factory(config={
                'module_type_list': [
                    'Conv2d'
                ]
            }),
            GroupRule.factory()
        ]

        model = MockModule()
        results = group_parameters(model, rules=rules)
        self.assertEqual(len(results), 3)
        self.assertEqual(set(id(p) for p in results[0]), {id(model.conv1.weight), id(model.conv2.weight)})
        self.assertEqual(set(id(p) for p in results[1]), {id(model.conv1.bias), id(model.conv2.bias)})
        self.assertEqual(set(id(p) for p in results[2]), {id(model.bn1.weight), id(model.bn1.bias)})


if __name__ == '__main__':
    unittest.main()
