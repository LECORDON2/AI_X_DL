# AI+X: 딥러닝 project

# Member: 독어독문학과 이사열 goddltkduf@naver.com

# Proposal
  
  Motivation: 비디오 게임을 발매할때의 여러 정보들을 토대로 게임의 판매량을 예측할 수 있다면
              실제 실물로서 비디오 게임을 판매하는 업자들에게 도움이 될 것이라고 생각하였고 
              실제 그러한 정보들 간의 상관관계가 얼마나 있는지 궁금하여 이 주제를 정하게 되었습니다.
              최종적으로는 대략적인 범위로나마 판매량을 예측 할 수 있기를 바랍니다.
              
# Datasets: https://www.kaggle.com/datasets/gregorut/videogamesales

# 데이터에 대한 기본적인 정복는 아래와 같습니다.
  Rank - Ranking of overall sales
  Name - The games name
  Platform - Platform of the games release (i.e. PC,PS4, etc.)
  Year - Year of the game's release
  Genre - Genre of the game
  Publisher - Publisher of the game
  NA_Sales - Sales in North America (in millions)
  EU_Sales - Sales in Europe (in millions)
  JP_Sales - Sales in Japan (in millions)
  Other_Sales - Sales in the rest of the world (in millions)
  Global_Sales - Total worldwide sales.

# 좀더 자세히 살펴보기 위해 csv파일을 불러오고 확인해 봅니다. 16598개의 행과 11개의 열로 되었있습니다. 저는 최종적으로 Global_Sales을 예측해 보려고합니다.
filename = 'C:/Users/mycom/Desktop/DL/final_project/vgsales.csv'
VG = pd.read_csv(filename)


# 좀더 자세히 살펴보기 위해 csv파일을 불러오고 확인해 봅니다. 16598개의 행과 11개의 열로 되었있습니다. 우리는 최종적으로 Global_Sales을 예측해 보려고합니다.
# print(VG)
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

# 데이터의 타입도 확인해 보겠습니다. 우리가 활용한 것은 Platform ,Genre, Year, Publisher, Global_Sales이므로 데이터의 타입은 후에 알맞는 형태로 변경할 것입니다.
# print(VG.dtypes) 
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

# 결측치를 확인해 보겠습니다. 결측치가 존재하나 많기도하며 적절한 값을 채우기 힘들어보여 결측치가 있는 행은 없애기로 하였습니다.
# print(VG.isnull().sum())
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
# print(VG_NA.isnull().sum())
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

# Methodology

