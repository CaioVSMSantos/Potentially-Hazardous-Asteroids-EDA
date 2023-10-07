from base_data_science_utils import pd, np, sp, percentage

from pandas.api.types import is_numeric_dtype
from enum import Enum


class OutlierCalculationMethod(Enum):
    INTERQUARTILE = 'interquartile'
    ZSCORE = 'zscore'


class OutlierHandler:
    def __init__(self, calculation_method: OutlierCalculationMethod = OutlierCalculationMethod.ZSCORE):
        self.calculation_method = calculation_method
        self.__outlier_calculation_methods = {
            OutlierCalculationMethod.INTERQUARTILE: self.__calculate_outliers_interquartile,
            OutlierCalculationMethod.ZSCORE: self.__calculate_outliers_zscore
        }
    
    def __calculate_outliers_interquartile(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        # calculate IQR for column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
    
        # identify outliers
        threshold = 1.5
        outliers = df[(df[column] < Q1 - threshold * IQR) | (df[column] > Q3 + threshold * IQR)]
        return outliers.copy()

    
    def __calculate_outliers_zscore(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        # calculate z-score for column
        z = np.abs(sp.stats.zscore(df[column]))
        
        # identify outliers
        threshold = 3
        outliers = df[z > threshold]

        return outliers.copy()

    
    def __filter_dataframe_out(self, df: pd.DataFrame, filter: pd.DataFrame) -> pd.DataFrame:
        """""
        Returns a DataFrame excluding the elements specified in the filter
        """
        return df[~df.index.isin(filter.index)]


    def get_missing(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """""
        Returns a DataFrame composed of the missing values of the specified column
        """
        return df[df[column].isna()]


    def get_non_missing(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """""
        Returns a DataFrame composed of the non missing values of the specified column
        """
        return df.dropna(subset=[column])


    def get_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Returns a DataFrame composed of the outliers of the specified column
        """

        non_na_df = self.get_non_missing(df, column)
        return self.__outlier_calculation_methods[self.calculation_method](df=non_na_df, column=column)


    def get_non_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Returns a DataFrame without outliers and missing values of all specified columns
        """
        non_na_df = self.get_non_missing(df, column)
        outliers_df = self.get_outliers(df, column)
        return self.__filter_dataframe_out(non_na_df, filter=outliers_df)


    def get_outliers_information(self, df: pd.DataFrame, output_dataframes: bool = True, columns: list[str] = [], log_information: bool = True) -> dict:
        df_total_rows = len(df.index)
        result = {}
        for column in df.columns:
            if is_numeric_dtype(df[column]) and (bool(columns) is False or column in columns):
                missing_values_df = self.get_missing(df, column)
                non_missing_values_df = self.get_non_missing(df, column)
                outlier_values_df = self.get_outliers(df, column)
                non_outlier_values_df = self.get_non_outliers(df, column)

                missing_values_count = len(missing_values_df.index)
                outlier_values_count = len(outlier_values_df.index)
                non_outlier_values_count = len(non_outlier_values_df.index)

                if missing_values_count + outlier_values_count + non_outlier_values_count != df_total_rows:
                    print(f'WARNING: Value Counts for column {column} does not match')
                
                result[column] = {
                    'missing_values_count': missing_values_count,
                    'missing_values_percentage': percentage(missing_values_count, df_total_rows),
                    'outlier_values_count': outlier_values_count,
                    'outlier_values_percentage': percentage(outlier_values_count, df_total_rows),
                    'non_outlier_values_count': non_outlier_values_count,
                    'non_outlier_values_percentage': percentage(non_outlier_values_count, df_total_rows),
                }
                if output_dataframes:
                    result[column]['missing_values_df'] = missing_values_df
                    result[column]['outlier_values_df'] = outlier_values_df
                    result[column]['non_outlier_values_df'] = non_outlier_values_df

        if log_information:
            self.__log_column_information(result)
        return result


    def __log_column_information(self, outliers_information: dict[str, any]) -> None:
        justify_length = 25
        for column, information in outliers_information.items():
            print(column)
            print(f'{"Missing Values = ".ljust(justify_length)}{information["missing_values_count"]} ({information["missing_values_percentage"]} %)')
            print(f'{"Outlier Values = ".ljust(justify_length)}{information["outlier_values_count"]} ({information["outlier_values_percentage"]} %)')
            print(f'{"Non Outlier Values = ".ljust(justify_length)}{information["non_outlier_values_count"]} ({information["non_outlier_values_percentage"]} %)')
            print('\n')
