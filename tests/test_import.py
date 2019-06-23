import unittest


class MyTestCase(unittest.TestCase):
    def test_import(self):
        import torch_parameter_groups
        self.assertIsNotNone(torch_parameter_groups)


if __name__ == '__main__':
    unittest.main()
