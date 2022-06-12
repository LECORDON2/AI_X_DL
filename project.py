# https://medium.com/@ljw11821/%EC%82%AC%EC%9A%A9%EC%9E%90-%ED%8A%B9%EC%84%B1%EC%97%90-%EB%94%B0%EB%A5%B8-%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8F%B0-%EC%A0%9C%EC%A1%B0%EC%82%AC-%EC%98%88%EC%B8%A1-5771ecc25053
# https://medium.com/@oloxle0814/2021-ai-x-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B8%B0%EB%A7%90-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%ED%95%B4%EC%99%B8-%EC%97%AC%ED%96%89-%EA%B2%BD%ED%97%98-%EC%97%AC%EB%B6%80-%EC%98%88%EC%B8%A1-774a2a428692
# https://medium.com/@jimin991203/%EC%A7%80%ED%95%98%EC%B2%A0-%EB%AF%B8%EC%84%B8%EB%A8%BC%EC%A7%80-%EC%98%88%EC%B8%A1-4d7439525102
# https://medium.com/@yong08655/2021-aix-fall-36d48556d24c
# https://prismspirit.github.io/ml/dl/Diamond/
# https://velog.io/@skyepodium/kaggle-%ED%83%80%EC%9D%B4%ED%83%80%EB%8B%89-%EC%83%9D%EC%A1%B4%EC%9E%90-%EC%98%88%EC%B8%A1
# https://inuplace.tistory.com/570
# https://injo.tistory.com/30

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.metrics import classification_report

# Title ##############################################################################################################################################
# Member: 독어독문학과 이사열 2016036280 goddltkduf@naver.com

# Proposal ###########################################################################################################################################
# Motivation: 비디오 게임을 발매할때의 여러 정보들을 토대로 게임의 판매량을 예측할 수 있다면 실제 실물로서 비디오 게임을 판매하는 업자들에게 도움이 될 것이라고 
#             생각하였고 실제 그러한 정보들 간의 상관관계가 얼마나 있는지 궁금하여 이 주제를 정하게 되었습니다. 최종적으로는 대략적인 범위로나마 판매량을 예측 
#             할 수 있기를 바랍니다.

# Datasets ###########################################################################################################################################
# Datasets: https://www.kaggle.com/datasets/gregorut/videogamesales

# Describing Dataset
# 데이터의 전반적인 구성은 다음과 같습니다.
# Rank - Ranking of overall sales
# Name - The games name
# Platform - Platform of the games release (i.e. PC,PS4, etc.)
# Year - Year of the game's release
# Genre - Genre of the game
# Publisher - Publisher of the game
# NA_Sales - Sales in North America (in millions)
# EU_Sales - Sales in Europe (in millions)
# JP_Sales - Sales in Japan (in millions)
# Other_Sales - Sales in the rest of the world (in millions)
# Global_Sales - Total worldwide sales.

filename = 'C:/Users/mycom/Desktop/DL/final_project/vgsales.csv'
VG = pd.read_csv(filename)

# 좀더 자세히 살펴보기 위해 csv파일을 불러오고 확인해 봅니다. 16598개의 행과 11개의 열로 되었있습니다. 저는 최종적으로 Global_Sales을 예측해 보려고합니다.
print(VG)
#         Rank                                              Name Platform    Year         Genre   Publisher  NA_Sales  EU_Sales  JP_Sales  Other_Sales  Global_Sales
# 0          1                                        Wii Sports      Wii  2006.0        Sports    Nintendo     41.49     29.02      3.77         8.46         82.74
# 1          2                                 Super Mario Bros.      NES  1985.0      Platform    Nintendo     29.08      3.58      6.81         0.77         40.24
# 2          3                                    Mario Kart Wii      Wii  2008.0        Racing    Nintendo     15.85     12.88      3.79         3.31         35.82
# 3          4                                 Wii Sports Resort      Wii  2009.0        Sports    Nintendo     15.75     11.01      3.28         2.96         33.00
# 4          5                          Pokemon Red/Pokemon Blue       GB  1996.0  Role-Playing    Nintendo     11.27      8.89     10.22         1.00         31.37
# ...      ...                                               ...      ...     ...           ...         ...       ...       ...       ...          ...           ...
# 16593  16596                Woody Woodpecker in Crazy Castle 5      GBA  2002.0      Platform       Kemco      0.01      0.00      0.00         0.00          0.01
# 16594  16597                     Men in Black II: Alien Escape       GC  2003.0       Shooter  Infogrames      0.01      0.00      0.00         0.00          0.01
# 16595  16598  SCORE International Baja 1000: The Official Game      PS2  2008.0        Racing  Activision      0.00      0.00      0.00         0.00          0.01
# 16596  16599                                        Know How 2       DS  2010.0        Puzzle    7G//AMES      0.00      0.01      0.00         0.00          0.01
# 16597  16600                                  Spirits & Spells      GBA  2003.0      Platform     Wanadoo      0.01      0.00      0.00         0.00          0.01
# [16598 rows x 11 columns]

# 데이터의 타입도 확인해 보겠습니다. 제가 활용한 것은 Platform ,Genre, Year, Publisher, Global_Sales이므로 데이터의 타입은 후에 알맞는 형태로 변경할 것입니다.
print(VG.dtypes) 
# Rank              int64
# Name             object
# Platform         object
# Year            float64
# Genre            object
# Publisher        object
# NA_Sales        float64
# EU_Sales        float64
# JP_Sales        float64
# Other_Sales     float64
# Global_Sales    float64

# Methodology ########################################################################################################################################
# Algorithms

# 현재 있는 데이터를 알고리즘에 활용하기위해 적절한 형태로 바꾸는 작업을 하겠습니다.
# 우선 결측치를 확인해 보겠습니다. 확인 결과 결측치가 존재하나 많기도하며 적절한 값을 채우기 힘들어보여 결측치가 있는 행은 없애기로 하였습니다.
print(VG.isnull().sum())
# Rank              0
# Name              0
# Platform          0
# Year            271
# Genre             0
# Publisher        58
# NA_Sales          0
# EU_Sales          0
# JP_Sales          0
# Other_Sales       0
# Global_Sales      0
# dtype: int64

# 결측치 있는 행 없애기
VG_NA = VG.dropna(axis=0)

# 결측치를 다시 확인해보니 이제 결측치가 없는 걸 확인 할 수 있습니다.
print(VG_NA.isnull().sum())
# Rank            0
# Name            0
# Platform        0
# Year            0
# Genre           0
# Publisher       0
# NA_Sales        0
# EU_Sales        0
# JP_Sales        0
# Other_Sales     0
# Global_Sales    0
# dtype: int64

# 데이터 프레임에서 이후 예측을 위해서 필요 없는 columns를 없애고 Training데이터와 Target를 나눕니다.
VG_Training = VG_NA.drop(['NA_Sales', 'Rank', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Name'], axis=1)
VG_Target = VG_NA['Global_Sales']

# 문자로된 열, Genre, Platform, Publisher를 정수로 바꾸기위해서 일단 리스트로 만듭니다.
VG_platform_list = VG_Training['Platform'].tolist()
VG_Genre_list = VG_Training['Genre'].tolist()
VG_Publisher_list = VG_Training['Publisher'].tolist()

# 리스트의 값들의 종류와 숫자를 파악하기 위해서 중복을 없애고 종류 별로 넣을 빈 리스트를 만듭니다.
VG_platform_my_list = []
VG_Genre_my_list = []
VG_Publisher_my_list = []

# 리스트안에 중복되는 values들을 제거하고 하나씩만 남겨 만들었던 빈 리스트에 넣습니다.
for v in VG_platform_list:
    if v not in VG_platform_my_list:
        VG_platform_my_list.append(v)
        
for v in VG_Genre_list:
    if v not in VG_Genre_my_list:
        VG_Genre_my_list.append(v)

for v in VG_Publisher_list:
    if v not in VG_Publisher_my_list:
        VG_Publisher_my_list.append(v)


# 전체 list의 values를 정수값으로 바꾸어 넣을 빈 리스트를 만듭니다.
VG_platform_my_list_num = []
VG_Genre_my_list_num = []
VG_Publisher_my_list_num = []

# values의 종류만을 넣었던 list의 index를 활용해 각 list의 값을 정수로 바꿉니다.
for i in VG_platform_list:
    if i in VG_platform_my_list:
        VG_platform_my_list_num.append(VG_platform_my_list.index(i))
        
for i in VG_Genre_list:
    if i in VG_Genre_my_list:
        VG_Genre_my_list_num.append(VG_Genre_my_list.index(i))
                
for i in VG_Publisher_list:
    if i in VG_Publisher_my_list:
        VG_Publisher_my_list_num.append(VG_Publisher_my_list.index(i))
        
# 만들은 정수 list를 활용하여 dataframe의 Platform, Genre, Publisher의 values들을 정수로 바꿉니다.
VG_Training['Platform'] = VG_platform_my_list_num
VG_Training['Genre'] = VG_Genre_my_list_num
VG_Training['Publisher'] = VG_Publisher_my_list_num

# float인 Year를 정수값으로 바꾸어 줍니다.
VG_Training_int = VG_Training.astype({'Year': 'int'})
# VG_Training_int를 확인해 보면 모든 values가 정수로 바뀐것을 볼 수 있습니다.
print(VG_Training_int.dtypes)
# Platform     int64
# Year         int32
# Genre        int64
# Publisher    int64
# dtype: object

# Evaluation & Analysis #############################################################################################################################

# 예측을 위해 VG_Target_int의 'Global_Sales'또한 정수로 바꾸어 줍니다.
VG_Target_int = VG_Target.astype({'Global_Sales': 'int'})

# 테스트 사이즈는 25%로 합니다.
Train_X, Test_X, Train_y, Test_y = train_test_split(VG_Training_int, VG_Target_int, test_size = 0.25)

VG_R = RandomForestClassifier()
VG_R.fit(Train_X, Train_y)

predicted = VG_R.predict(Test_X)
accuracy = accuracy_score(Test_y, predicted)

# 결과를 살펴보면 정확도는 0.8534249938620182 입니다.
print(accuracy)
# 0.8534249938620182

# 더 정확한 성능평가는 아래와 같습니다.
print(classification_report(Test_y, predicted))
#               precision    recall  f1-score   support

#            0       0.90      0.96      0.93      3565
#            1       0.28      0.18      0.22       296
#            2       0.12      0.07      0.09        89
#            3       0.09      0.04      0.05        50
#            4       0.00      0.00      0.00        22
#            5       0.25      0.05      0.09        19
#            6       0.00      0.00      0.00        10
#            7       0.00      0.00      0.00         5
#            8       0.00      0.00      0.00         2
#            9       0.00      0.00      0.00         3
#           11       0.00      0.00      0.00         2
#           12       0.00      0.00      0.00         0
#           13       0.00      0.00      0.00         2
#           14       0.00      0.00      0.00         0
#           15       0.00      0.00      0.00         1
#           18       0.00      0.00      0.00         1
#           20       0.00      0.00      0.00         1
#           22       0.00      0.00      0.00         0
#           23       0.00      0.00      0.00         0
#           24       0.00      0.00      0.00         1
#           28       0.00      0.00      0.00         1
#           30       0.00      0.00      0.00         1
#           33       0.00      0.00      0.00         1
#           40       0.00      0.00      0.00         1
#           82       0.00      0.00      0.00         0

#     accuracy                           0.85      4073
#    macro avg       0.07      0.05      0.06      4073
# weighted avg       0.81      0.85      0.83      4073



# 위의 방식으로 할 때는 소수점 아래 숫자를 무시하기 때문에 0과 1인 데이터가 너무 많이 나옴에 따라 숫자가 낮은 쪽의 정확도만 너무 높기 때문에 좋은 결과라고 보기
# 힘들다고 판단하여 다른 방식으로 해보기 위해 Global_Sales의 값에 100을 곱하고 정수로 바꿉니다.
VG_Target_100 = VG_Target.mul(100)
VG_Target_100_int = VG_Target_100.astype({'Global_Sales': 'int'})

##################################################################################################################################################

# 위의 방식으로 할 때는 소수점 아래 숫자를 무시하기 때문에 0과 1인 데이터가 너무 많이 나옴에 따라 숫자가 낮은 쪽의 정확도만 너무 높기 때문에 좋은 결과라고 보기
# 힘들다고 판단하여 다른 방식으로 해보기 위해 Global_Sales의 값에 100을 곱하고 정수로 바꿉니다.
VG_Target_100 = VG_Target.mul(100)
VG_Target_100_int = VG_Target_100.astype({'Global_Sales': 'int'})

# VG_Target_100_int활용하여 다시 예측을 해 봅니다.
Train_X, Test_X, Train_y, Test_y = train_test_split(VG_Training_int, VG_Target_100_int, test_size = 0.25)

VG_R = RandomForestClassifier()
VG_R.fit(Train_X, Train_y)

predicted = VG_R.predict(Test_X)
accuracy = accuracy_score(Test_y, predicted)

# 결과를 살펴보면 정확도는 0.052786643751534496 입니다.
print(accuracy)
# 0.052786643751534496

# 더 정확한 성능평가는 아래와 같습니다.
print(classification_report(Test_y, predicted))
#              precision    recall  f1-score   support

#            1       0.17      0.17      0.17       138
#            2       0.19      0.27      0.23       248
#            3       0.13      0.12      0.12       219
#            4       0.08      0.07      0.08       173
#            5       0.07      0.07      0.07       158
#            6       0.06      0.07      0.06       138
#            7       0.04      0.05      0.04       109
#            8       0.03      0.03      0.03       125
#            9       0.03      0.03      0.03       114
#           10       0.02      0.02      0.02       100
#           11       0.04      0.05      0.05        94
#           12       0.02      0.03      0.03        76
#           13       0.01      0.01      0.01        72
#           14       0.11      0.08      0.09        88
#           15       0.03      0.03      0.03        63
#           16       0.07      0.04      0.05        72
#           17       0.06      0.04      0.05        73
# ...        ...   ...    ...        ...           ...
#           19       0.03      0.03      0.03        68
#         1346       0.00      0.00      0.00         0
#         1403       0.00      0.00      0.00         1
#         1464       0.00      0.00      0.00         1
#         1476       0.00      0.00      0.00         1
#         1498       0.00      0.00      0.00         0
#         1614       0.00      0.00      0.00         1
#         1638       0.00      0.00      0.00         1
#         2022       0.00      0.00      0.00         0
#         2140       0.00      0.00      0.00         1
#         2272       0.00      0.00      0.00         0
#         2310       0.00      0.00      0.00         1
#         2476       0.00      0.00      0.00         1
#         3137       0.00      0.00      0.00         0
#         8274       0.00      0.00      0.00         1

#     accuracy                           0.06      4073
#    macro avg       0.02      0.03      0.02      4073
# weighted avg       0.06      0.06      0.06      4073

##################################################################################################################################################
# 오히려 values 종류가 너무 많아 정확도가 떨어지는 것으로 보입니다.
# 그래서 정수로 바꾼 values들을 범위로 다시 분류하여 values의 종류를 줄이기 위해서 첫 번재로 list로 만듭니다.
VG_Target_100_int_list = VG_Target_100_int.tolist()

# list안에 있는 값을 25단위로 나누기 위해 일단 분류 후에 값들이 들어갈 빈 list를 만듭니다.
VG_Target_100_int_list_25 = []

# VG_Target_100_int_list에 있는 값들을 25로 나누에 정수인 몫 만을 앞서 만들었던 빈 list에 넣습니다.
for i in VG_Target_100_int_list:
    VG_Target_100_int_list_25.append(i//25)
    
# list인 VG_Target_100_int_list_25를 series로 바꿉니다.
VG_Target_25_int = pd.Series(VG_Target_100_int_list_25)

# VG_Target_25_int를 활용하여 다시 예측을 해 봅니다.
Train_X, Test_X, Train_y, Test_y = train_test_split(VG_Training_int, VG_Target_25_int, test_size = 0.25)

VG_R = RandomForestClassifier()
VG_R.fit(Train_X, Train_y)

predicted = VG_R.predict(Test_X)
accuracy = accuracy_score(Test_y, predicted)

# 결과를 살펴보면 정확도는 0.052786643751534496 입니다.
print(accuracy)
# 0.5470169408298552

# 더 정확한 성능평가는 아래와 같습니다.
print(classification_report(Test_y, predicted))
#               precision    recall  f1-score   support

#            0       0.70      0.81      0.75      2384
#            1       0.27      0.23      0.25       709
#            2       0.16      0.13      0.14       296
#            3       0.16      0.10      0.12       168
#            4       0.05      0.04      0.05       121
#            5       0.05      0.04      0.04        83
#            6       0.00      0.00      0.00        66
#            7       0.00      0.00      0.00        34
#            8       0.03      0.04      0.03        24
#            9       0.00      0.00      0.00        27
#           10       0.00      0.00      0.00        28
#           11       0.15      0.14      0.14        22
#           12       0.00      0.00      0.00        10
#           13       0.00      0.00      0.00         6
#           14       0.00      0.00      0.00        11
#           15       0.00      0.00      0.00        10
#           16       0.00      0.00      0.00        11
#           17       0.00      0.00      0.00         5
#           18       0.00      0.00      0.00         5
#           24       0.00      0.00      0.00         2
# ...        ...   ...    ...        ...           ...
#           61       0.00      0.00      0.00         1
#           63       0.00      0.00      0.00         0
#           64       0.00      0.00      0.00         0
#           65       0.00      0.00      0.00         0
#           85       0.00      0.00      0.00         1
#           87       0.00      0.00      0.00         1
#           88       0.00      0.00      0.00         1
#           90       0.00      0.00      0.00         1
#           93       0.00      0.00      0.00         1
#           99       0.00      0.00      0.00         1
#          116       0.00      0.00      0.00         0
#          132       0.00      0.00      0.00         0
#          330       0.00      0.00      0.00         0

#     accuracy                           0.53      4073
#    macro avg       0.03      0.03      0.03      4073
# weighted avg       0.48      0.53      0.50      4073


##################################################################################################################################################
# 여전히 전체 Target 안 values에서 0과 1같이 판매량이 저조한 비디오게임이 너무 많은 비율을 차지하고 있어서 이것을 제거하면 혹시 더 정확히 예측이 될지 알아보겠습니다.
# VG_Target_25_int의 values들의 비율을 보면 아래와 같이 0과 1이 3분의 2정도 되는 것을 볼 수 있습니다.
print(VG_Target_25_int.value_counts())
# 0      9634
# 1      2697
# 2      1236
# 3       666
# 4       478
#        ...
# 50        1
# 49        1
# 40        1
# 45        1
# 330       1

# 일단 0과 1일 포함된 행을 없애기 위해 VG_Target_100_int_list_25라는 list를 다시 기존의 데이터 프레임인 VG_Training_int에 합칩니다.
VG_Training_int['Global_Sales'] = VG_Target_100_int_list_25

# Global_Sales columns에서 0과 1을 제거합니다. 정확히는 0과 1이 포함된 행을 제거합니다.
VG_Training_int_drop_0 = VG_Training_int.drop(index = VG_Training_int[VG_Training_int['Global_Sales'] == 0].index)
VG_Training_int_drop_1_0 = VG_Training_int_drop_0.drop(index = VG_Training_int_drop_0[VG_Training_int_drop_0['Global_Sales'] == 1].index)


# 전체 행이 16291 - 9634 - 2697 = 3960 으로 줄어든 것을 볼 수 있습니다.
print(VG_Training_int_drop_1_0)
# Platform  Year  Genre  Publisher  Global_Sales
# 0            0  2006      0          0           330
# 1            1  1985      1          0           160
# 2            0  2008      2          0           143
# 3            0  2009      0          0           132
# 4            2  1996      3          0           125
# ...        ...   ...    ...        ...           ...
# 4013        15  1982      0         10             2
# 4014         6  2007      1         28             2
# 4015        15  1981      8         10             2
# 4016        15  1982      8        162             2
# 4017         9  2011      2         25             2
# [3960 rows x 5 columns]

# VG_Training_int_drop_1_0을 이용해 다시 ㅍTraining데이터와 Target를 나눕니다.
VG_Training_drop = VG_Training_int_drop_1_0.drop(['Global_Sales'], axis=1)
VG_Target_drop = VG_Training_int_drop_1_0['Global_Sales']

# 나눈 데이터들을 이용해 다시 예측을 해봅니다.
Train_X, Test_X, Train_y, Test_y = train_test_split(VG_Training_drop, VG_Target_drop, test_size = 0.25)

VG_R = RandomForestClassifier()
VG_R.fit(Train_X, Train_y)

predicted = VG_R.predict(Test_X)
accuracy = accuracy_score(Test_y, predicted)


# 결과를 살펴보면 정확도는 0.2383838383838384 입니다.
print(accuracy)
# 0.2383838383838384

# 더 정확한 성능평가는 아래와 같습니다.
print(classification_report(Test_y, predicted))
#               precision    recall  f1-score   support

#            2       0.38      0.51      0.44       289
#            3       0.26      0.31      0.28       157
#            4       0.21      0.17      0.19       132
#            5       0.09      0.05      0.06       104
#            6       0.07      0.09      0.08        57
#            7       0.07      0.05      0.06        40
#            8       0.00      0.00      0.00        34
#            9       0.00      0.00      0.00        23
#           10       0.00      0.00      0.00        16
#           11       0.08      0.05      0.06        20
#           12       0.00      0.00      0.00        11
#           13       0.00      0.00      0.00         6
#           14       0.00      0.00      0.00        17
#           15       0.00      0.00      0.00         8
#           16       0.00      0.00      0.00        11
#           17       0.00      0.00      0.00         5
#           18       0.00      0.00      0.00         7
#           19       0.00      0.00      0.00         3
#           20       0.00      0.00      0.00         3
#           21       0.00      0.00      0.00         2
#           22       0.00      0.00      0.00         1
#           23       0.00      0.00      0.00         1
# ...        ...   ...    ...        ...           ...
#           58       0.00      0.00      0.00         2
#           59       0.00      0.00      0.00         1
#           61       0.00      0.00      0.00         1
#           63       0.00      0.00      0.00         1
#           64       0.00      0.00      0.00         1
#           69       0.00      0.00      0.00         0
#           73       0.00      0.00      0.00         1
#           83       0.00      0.00      0.00         1
#           85       0.00      0.00      0.00         1
#           88       0.00      0.00      0.00         1
#           92       0.00      0.00      0.00         0
#           93       0.00      0.00      0.00         1
#           99       0.00      0.00      0.00         1
#          116       0.00      0.00      0.00         0
#          121       0.00      0.00      0.00         1
#          132       0.00      0.00      0.00         0
#          143       0.00      0.00      0.00         1

#     accuracy                           0.23       990
#    macro avg       0.02      0.02      0.02       990
# weighted avg       0.20      0.23      0.21       990

# Related Work ####################################################################################################################################
# https://injo.tistory.com/30
# https://inuplace.tistory.com/570
# https://velog.io/@skyepodium/kaggle-%ED%83%80%EC%9D%B4%ED%83%80%EB%8B%89-%EC%83%9D%EC%A1%B4%EC%9E%90-%EC%98%88%EC%B8%A1

# Conclusion: Discussion ###########################################################################################################################
# 당초의 목적은 전체 데이터에서 우리가 활용 할 수 있는 데이터를 통해서 전체적인 판매량을 예측하는 것이었습니다. 그중에서 제가 활용한 Platform ,Genre, Year, Publisher는 
# 보통 비디오 게임이 발매되기 전에 확실히 알 수 있는 것이므로 이를 통해서 비디오 게임의 전체적인 판매량을 예측 한다면 꽤 유용할 것이라고 생각하였습니다. 그러나 판매량이라는 
# 특성 상 약 16000개의 데이터에서 낮은 값들이 대다수이며 아주 높은 값들은 너무 적어 크게 유용하지 못했던 것 같습니다. 이를 타파하고자 데이터를 이리저리 만져 보았지만 썩 
# 만족스러운 결과는 아니었던 것 같습니다. 이를 통해서 예측에는 양질의 데이터가 필요함을 알 수 있었고 데이터의 분포도나 상관도가 중요하게 작용할 수 있음을 약간이나 느꼈습니다. 
# 만약 다음에 같은 주제로 예측을 하게 되는 일이 생긴다면 좀 더 적절하고 다양한 정보로 할 필요가 있음을 알았습니다.