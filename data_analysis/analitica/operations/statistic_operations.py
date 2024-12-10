import numpy as np
import pandas as pd
from typing import Any, Callable, Generator

def function_logger(func: Callable) -> Callable:

    '''
    Декоратор используется для логгирования функции
    Аргумент: func - Callable-объект
    Результат: Callable-объект
    '''

    def wrapper(*args: Any, **kwargs: Any):
        print(f"Вызов {func.__name__} с аргументами {args} {kwargs}")
        result = func(*args, **kwargs)
        print(f"Результат - {result}")
        return result
    return wrapper


class StockAnalysis:
    ''' Класс для анализа акции. 
        data: массив np.ndarray, принимаемые данные
       '''
    def __init__(self, data: np.ndarray) -> None:
        ''' Инициализация класса
        Аргумент: data - np.ndarray, массив данных
        Без возвращаемого значения
        '''
        self.data = pd.Series(data)

    def moving_average(self, window_size: int = 3) -> pd.Series:
        """Подсчет скользящего среднего для временного ряда.

        Аргумент - window_size (int) - размер окна для скользящего среднего.
        Результат - скользящее среднее временного ряда.
        """
        return self.data.rolling(window=window_size).mean()

    def diff(self) -> pd.Series:
        """ Вычисление дифференциала для временного ряда """
        return self.data.diff()

    def autocorr(self, lag: int = 1) -> pd.Series:
        """ Вычисление автокорреляции временного ряда
        Аргумент: lag - лаг автокорреляции
        Результат - значение автокорреляции """
        maxumum_lag = len(self.data) - 1
        data_without_nan = self.data.dropna()
        autocorr = []
        for lag in range(1, maxumum_lag + 1):
            autocorr.append(data_without_nan.autocorr(lag))
        return pd.Series(autocorr, index=np.arange(1, maxumum_lag + 1))

    def maximum(self) -> pd.Series:
        """ Поиск максимумов временного ряда (локальных максимумов) """
        return self.data[(self.data.shift(1) < self.data) &
                         (self.data.shift(-1) < self.data)]

    def minimum(self) -> pd.Series:
        """ Поиск минимумов временного ряда (локальных минимумов) """
        return self.data[(self.data.shift(1) > self.data)
                         & (self.data.shift(-1) > self.data)]


    def save(self, result: pd.Series, name: str) -> Generator[pd.DataFrame, None, None]:
        """Сохраняем результат с определённым названием.
        Аргументы: result - результат вычислений, name - имя столбца.
        Возвращает: генератор - DataFrame-результат вычислений.
        """
        
        yield pd.DataFrame({name: result})

    @function_logger
    def get_results(self) -> pd.DataFrame:
        ''' Получение результатов анализа.
        Возвращаемый результат: итоговый DataFrame.
        '''
        ''' Перебираемся по итерируемому объекту с помощью next() '''
        df = next(self.save(self.data, 'Stock Cost'))

        for name, func in [
            ('Moving average', self.moving_average),
            ('Differential', self.diff),
            ('Autocorrelation', self.autocorr),
            ('Max', self.maximum),
            ('Min', self.minimum),
        ]:

            df = df.join(next(self.save(func(), name)), how='outer')
        return df
