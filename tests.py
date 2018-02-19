from unittest import TestCase

from .main import binarize_targets
from .cw import find_minimum, load_numbers, count_result

numbers_mock = [0, 3, 30, 36, 255, 258, 288, 291, 2046, 2052, 2079, 2082, 2304, 2307, 2334, 2340, 16383, 16386, 16416, 16419, 16638, 16644, 16671, 16674, 18432, 18435, 18462, 18468, 18687, 18690, 18720, 18723]


class TestUtils(TestCase):
    def setUp(self):
        pass

    def testLoadLumbers(self):
        numbers = load_numbers('scikit/numbers.txt')
        self.assertTrue(numbers)

    def testFindMinimum(self):
        minimum = find_minimum(numbers_mock, 5)
        self.assertEqual(minimum, 3)
        minimum = find_minimum(numbers_mock, 4)
        self.assertEqual(minimum, 3)
        minimum = find_minimum(numbers_mock, 60)
        self.assertEqual(minimum, 36)
        minimum = find_minimum(numbers_mock, 1000)
        self.assertEqual(minimum, 291)

    def testCount(self):
        _, counter, x = count_result(numbers_mock, 3)
        self.assertEqual(counter, 0)
        self.assertEqual(x, 3)
        _, counter, x = count_result(numbers_mock, 5)
        self.assertEqual(counter, 1)
        self.assertEqual(x, 2)
        _, counter, x = count_result(numbers_mock, 35)
        self.assertEqual(counter, 2)
        self.assertEqual(x, 2)
        _, counter, x = count_result(numbers_mock, 10)
        self.assertEqual(counter, 3)
        self.assertEqual(x, 1)
        _, counter, x = count_result(numbers_mock, 15)
        self.assertEqual(counter, 4)
        self.assertEqual(x, 3)
