import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from functools import partialmethod
pd.DataFrame.head = partialmethod(pd.DataFrame.head, n=5)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 50)

''' Define the problem

    **Given**:

    Data from the Irish National Household Survey for the year 2015.

    The data contains information on the respondents' labour status (are they unemployed/working/studying etc), and a range of variables about their educational and demographic background (such as education level, education field, age, marital status, region of residency etc).

    **Goal**: Categorise respondents into unemployed/employed based on information about their educational/demographic background.



    #### Description about variables:
    employed: 1=employed, 2=unemployed.

    sex: 1=male, 2=female

    marriage status: 1=single, 2=married, 3=widowed, 4=divorced

    nationality: 1= ireland, 2=uk, 3=eu15, 4=rest of eu, 5=usa/canada, 5=rest of world

        nationality_1: almost the same

        nationality_3: almost the same

    **education level**:
    - 000=no formal education,
    - 100=ISCED1,
    - 200=ISCED2,
    - 300=ISCED3, 303=ISCED 3, 304=ISCED3,
    - 400=ISCED4,
    - 500=ISCED5,
    - 600=ISCED6,
    - 700=SCIED7,
    - 800=ISCED8,
    - 999=Not applicable. (where ISCED1-3~Primary school, ISCED4-5~Secondary School and ISCED6-8~Tertiary level)

    **education field**:
    - 000= General programmes Data available from Q1 2004 to Q4 2013
    - 100= Teacher training and education science
    - 200= Humanities, languages and arts
    - 222= Foreign languages
    - 300= Social sciences, business and law
    - 400= Science, mathematics and computing (no distinction possible)
    - 420= Life science (including Biology and Environmental science)
    - 440= Physical science (including Physics, Chemistry and Earth
    - science)
    - 460= Mathematics and statistics
    - 481= Computer science
    - 482= Computer use
    - 500= Engineering, manufacturing and construction
    - 600= Agriculture and veterinary
    - 700= Health and welfare
    - 800= Services
    - 900= Unknown
    - 999= Not applicable (HATLEVEL= 00,11,21,99, Blank)
    Blank No answer


    **age_group**:
    - 01= Aged 0 to 4 years
    - 02= Aged 5 to 9 years
    - 03= Aged 10 to 14 years
    - 04= Aged 15 to 19 years
    - 05= Aged 20 to 24 years
    - 06= Aged 25 to 29 years
    - 07= Aged 30 to 34 years
    - 08= Aged 35 to 39 years
    - 09= Aged 40 to 44 years
    - 10= Aged 45 to 49 years
    - 11= Aged 50 to 54 years
    - 12= Aged 55 to 59 years
    - 13= Aged 60 to 64 years
    - 14= Aged 65 to 69 years
    - 15= Aged 70 to 74 years
    - 16= Aged 75 to 79 years
    - 17= Aged 80 to 84 years
    - 18= Aged 85 or more

    household_type:
    1= Couple family unit – no children
    2= Couple family unit – with children
    3= Lone parent family unit
    9= Not applicable (FAMILYUNITGROUPNUMBER=0)

    region:
    01= Border, Midland and Western
        011=Border
        012=Midland
        013=Western
    02= Southern and Eastern
        021=Dublin
        022=Mid-East
        023=Mid-West
        024=South-East
        025South-West

'''
def load_data(path):
    ### load the data
    df=pd.read_csv(path)

    return df

def preprocess_data(df):
    ### rename columns
    df.rename(columns={"mainstat":"employed",
                       "marstat":"marriage_status",
                       "national_summary_x1":"nationality_1",
                       "national_summary_x3": "nationality_3",
                       "national_summary":"nationality",
                       "hatlevel":"education_level",
                       "hatfield":"education_field",
                       "ageclass":"age_group",
                       "familytypesummary":"household_type"}, inplace=True)

def clean_data(df):
    ''' What to clean

        Target Variable: exclude values from 'employed' which indicate a status other than employed or unemployed (e.g. student or XXX)

    '''
    mask=np.array((df.employed==2) | (df.employed==1))
    df=df[mask]

    # df.reset_index(inplace=True)
    # df.drop("index", axis=1, inplace=True)

def data_insights(df):
    ### get some data indights:
    print(f'df_info:')

    df.info() # no need to print()

    print(f'df.shape: \n{df.shape}')
    print(f'df.describe(): \n{df.describe()}')

    ### unique values:
    # df.employed.unique()

    ### unique values and counts:
    # better to print after cleaning, or to skip NaNs:
    values, counts = np.unique(df.employed[~np.isnan(df.employed)], return_counts=True)
    # values, counts = np.unique(df.employed, return_counts=True)
    print(f'unique values: \n{values}')
    print(f'unique counts: \n{counts}')

def show_nan_per_coulumn():
    # get all NaN counts per each column:
    nan = df.isnull().sum()

    # make new df with NaN values counts
    df_count_nan = pd.DataFrame(nan, columns=['number of NaN values'])
    df_count_nan

    # replace with 'All' for columns with NaN values only
    df_count_nan.replace(df.shape[0], 'ALL')

    return df_count_nan


if __name__ == "__main__":
    path_to_data = "./ML_Project/QNHS_2015.csv"
    df = load_data(path_to_data)

    preprocess_data(df)
    data_insights(df)
    show_nan_per_coulumn(df)