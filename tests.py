from unittest import TestCase

from .main import binarize_targets


class TestUtils(TestCase):
    def setUp(self):
        pass

    def test_binary_target(self):
        targets = [0, 1, 2, 3, 4, 5, 0, 1, 1]
        new_targets = binarize_targets(targets=targets, positive_label=1)
        scikit_targets = label_binarize([1], targets)[0]
        self.assertListEqual(new_targets, [0, 1, 0, 0, 0, 0, 0, 1, 1])
