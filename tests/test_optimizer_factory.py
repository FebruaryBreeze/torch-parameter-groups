import unittest

import torch
import torch.nn as nn

import torch_parameter_groups


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x).mean(3).mean(2)


class MyTestCase(unittest.TestCase):
    def test_optimizer_wrapper(self):
        model = MockModule()
        optimizer = torch_parameter_groups.optimizer_factory(
            model=model,
            config={
                'type': 'SGD',
                'kwargs': {
                    'momentum': 0.9,
                    'nesterov': True,
                    'weight_decay': 0.0001,
                },
                'rules': [
                    {
                        'param_name_list': ['weight'],
                        'kwargs': {
                            'weight_decay': 0.0
                        }
                    },
                    {
                    }
                ]
            },
        )
        self.assertIsNotNone(optimizer)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.0)

        loss = model(torch.randn(1, 3, 16, 16)).sum()
        loss.backward()
        optimizer.step(closure=None)
        self.assertTrue(repr(optimizer).lstrip().startswith('SGD'))
        self.assertTrue(isinstance(optimizer.param_groups, list))


if __name__ == '__main__':
    unittest.main()
