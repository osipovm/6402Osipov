import unittest
import warnings
import numpy as np 
import pandas as pd 
from analitica.operations.statistic_operations import StockAnalysis

class TestStockAnalysis(unittest.TestCase):

    def setUp(self):
        ''' Задаём тестовые данные '''
        self.data = np.array([2, 1, 3, 4, 7, 6, 5])
        self.analiz = StockAnalysis(self.data)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def test_moving_average(self):
        ''' Тестируем скользящее среднее '''
        expected_MA = pd.Series([np.nan, 1.5, 2.0, 3.5, 5.5, 6.5, 5.5])
        actual_MA = self.analiz.moving_average(window_size=2)
        self.assertTrue(actual_MA.equals(expected_MA))

    def test_diff(self):
        ''' Тестируем вычисление дифференциала '''
        expected_diff = pd.Series([np.nan, -1.0, 2.0, 1.0, 3.0, -1.0, -1.0])
        actual_diff = self.analiz.diff()
        self.assertTrue(actual_diff.equals(expected_diff))


    def test_autocorr(self):
        ''' Тестируем автокорреляцию '''
        expected_autocorr = pd.Series([0.6927178575499473, 0.3434014098717226, -0.39999999999999997, -0.5, 0.9999999999999999, np.nan], index=np.arange(1, 7))
        actual_autocorr = self.analiz.autocorr(lag=1)
        pd.testing.assert_series_equal(actual_autocorr, expected_autocorr, check_exact=False, check_index=False, check_dtype=False)


    def test_maximum(self):
        ''' Тестируем максимум '''
        expected_max = pd.Series([7], index=[4])
        result_max = self.analiz.maximum()
        self.assertEqual(result_max.index.tolist(), expected_max.index.tolist())
        self.assertEqual(result_max.values.tolist(), expected_max.values.tolist())

    def test_minimum(self):
        ''' Тестируем минимум '''
        expected_min = pd.Series([1], index=[1])
        result_min = self.analiz.minimum()
        self.assertEqual(result_min.index.tolist(), expected_min.index.tolist())
        self.assertEqual(result_min.values.tolist(), expected_min.values.tolist())
