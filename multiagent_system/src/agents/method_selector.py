"""
Агент-селектор методов.
Выбирает подходящий статистический метод для проверки гипотезы.
"""

class MethodSelector:
    def __init__(self):
        # Можно загрузить правила из конфигурационного файла
        self.rules = {
            'mean_difference': self._select_for_mean_difference,
            'linear_relationship': self._select_for_linear_relationship,
            'normality': self._select_for_normality
        }

    def select(self, hypothesis, dataset):
        """
        Выбор метода на основе гипотезы и данных.

        :param hypothesis: словарь с описанием гипотезы
        :param dataset: объект Dataset
        :return: строковое название метода или словарь с параметрами
        """
        hypothesis_type = hypothesis.get('type')
        if hypothesis_type in self.rules:
            return self.rules[hypothesis_type](hypothesis, dataset)
        else:
            return None

    def _select_for_mean_difference(self, hypothesis, dataset):
        """
        Выбор метода для сравнения средних.
        """
        # Если две группы, то t-тест, если больше - ANOVA
        col = hypothesis['column2']
        unique_vals = dataset.df[col].nunique()
        if unique_vals == 2:
            return 't_test'
        elif unique_vals > 2:
            return 'anova'
        else:
            return None

    def _select_for_linear_relationship(self, hypothesis, dataset):
        """
        Выбор метода для проверки линейной зависимости.
        """
        return 'pearson_correlation'

    def _select_for_normality(self, hypothesis, dataset):
        """
        Выбор метода для проверки нормальности распределения.
        """
        return 'shapiro_wilk'