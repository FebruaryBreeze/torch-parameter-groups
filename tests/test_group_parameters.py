import unittest

import torch
import torch.nn as nn

from torch_parameter_groups import GroupRule, group_parameters


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x: torch.Tensor):
        pass


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.sub = SubModule()
        self.classifier = nn.Linear(in_features=32, out_features=64)

    def forward(self, x: torch.Tensor):
        pass


class MyTestCase(unittest.TestCase):
    def test_group_parameters(self):
        rules = [
            GroupRule.factory(config={
                'module_name_list': [
                    'classifier'
                ]
            }),
            GroupRule.factory(config={
                'module_type_list': [
                    'SubModule'
                ]
            }),
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
            GroupRule.factory(config={
                'param_name_list': [
                    'weight'
                ]
            }),
            GroupRule.factory()
        ]

        model = MockModule()
        results = group_parameters(model, rules=rules)
        self.assertEqual(len(results), 6)
        print(results[0])

        self.assertEqual(set(map(id, results[0])), {id(model.classifier.weight), id(model.classifier.bias)})
        self.assertEqual(set(map(id, results[1])), {id(model.sub.bn1.weight), id(model.sub.bn1.bias)})
        self.assertEqual(set(map(id, results[2])), {id(model.conv1.weight), id(model.conv2.weight)})
        self.assertEqual(set(map(id, results[3])), {id(model.conv1.bias), id(model.conv2.bias)})
        self.assertEqual(set(map(id, results[4])), {id(model.bn1.weight)})
        self.assertEqual(set(map(id, results[5])), {id(model.bn1.bias)})


if __name__ == '__main__':
    unittest.main()
