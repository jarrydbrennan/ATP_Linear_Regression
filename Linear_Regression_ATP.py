import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from itertools import combinations

# load and investigate the data here:
atp = pd.read_csv('tennis_stats.csv')
print(atp.head())
print(atp.dtypes)

#perform exploratory analysis here:
atp_off = atp[['Aces', 'DoubleFaults', 'FirstServe', 'FirstServePointsWon', 'SecondServePointsWon','BreakPointsFaced','BreakPointsSaved', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalServicePointsWon']]

atp_def = atp[['FirstServeReturnPointsWon','SecondServeReturnPointsWon','BreakPointsOpportunities','BreakPointsConverted','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','TotalPointsWon']]

def create_scatter(x,y,c):
  for column in y.columns:
    plt.scatter(x, y[column], label=column, color = c)
    plt.xlabel('Wins')
    plt.ylabel(column)
    plt.title(f'Wins vs {column}')
    plt.show()
    plt.clf()

create_scatter(atp['Wins'], atp_off, 'blue')
create_scatter(atp['Wins'], atp_def, 'red')
print('The number of wins is highly correlated with increases in the following: BreakPointsFaced, ServiceGamesPlayed, BreakPointsOpportunities, and ReturnGamesPlayed')

#build a single feature linear regression model
#features 
X = atp[['BreakPointsFaced', 'ServiceGamesPlayed', 'BreakPointsOpportunities', 'ReturnGamesPlayed']]
print(X.head())
#outcome
y = atp[['Wins']]
train_test_split
for column in X.columns:
  feature = X[[column]]
  x_train, x_test, y_train, y_test = train_test_split(feature,y,train_size = 0.8, test_size = 0.2, random_state = 6)
  #initiate model
  linreg = LinearRegression()
  #fit & predict
  linreg.fit(x_train,y_train)
  print(f"{column} Train R^2:",linreg.score(x_train,y_train))
  print(f"{column} Test R^2:",linreg.score(x_test,y_test))
  #Plot actual vs predicted
  y_pred = linreg.predict(x_test)
  plt.scatter(y_test, y_pred, alpha=0.4)
  plt.plot([min(y_test.values), max(y_test.values)], [min(y_test.values), max(y_test.values)], linestyle='--', color='red')  # Diagonal line
  plt.title(f'Actual vs Predicted ({column} as Feature)')
  plt.xlabel('Actual Wins')
  plt.ylabel('Predicted Wins')
  plt.show()
  plt.clf()

# perform two feature linear regressions here:
target_variable = 'Winnings'
all_features = atp.columns.tolist()
all_features.remove(target_variable)
all_features.remove('Player')
y = atp[[target_variable]]

r2_list = []
#iterate through all combos of two features
for feature_pair in combinations(all_features, 2):
  current_features = list(feature_pair)
  #extract the current feature
  X = atp[current_features]
  #train_test_split
  x_train, x_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, test_size = 0.2, random_state = 6)
  #initiate model
  linreg = LinearRegression()
  #fit & predict
  linreg.fit(x_train, y_train)
  # Calc R2 values
  train_r2 = linreg.score(x_train,y_train)
  test_r2 = linreg.score(x_test,y_test)
  #append to list
  r2_list.append((current_features, train_r2, test_r2))
#sort list based on r2 test values in descenind order
r2_list.sort(key=lambda x: x[2], reverse=True)
#choose top 5 pairs
top_5_pairs = r2_list[:5]
#Plot actual vs predicted for top 5 pairs
for feature_pair, train_r2, test_r2 in top_5_pairs:
  X = atp[feature_pair]
  x_train,x_test,y_train,y_test = train_test_split(X,y,train_size = 0.8, test_size = 0.2, random_state = 6)
  linreg = LinearRegression()
  linreg.fit(x_train, y_train)
  #plot actual vs predicted
  y_pred = linreg.predict(x_test)
  plt.scatter(y_test, y_pred, alpha=0.4)
  plt.plot([min(y_test.values), max(y_test.values)], [min(y_test.values), max(y_test.values)], linestyle='--', color='red')  # Diagonal line
  plt.title(f'Actual vs Predicted ({", ".join(feature_pair)} as Features)\nTest R^2: {test_r2:.4f}')
  plt.xlabel(f'Actual {target_variable}')
  plt.ylabel(f'Predicted {target_variable}')
  plt.show()
  plt.clf()

## perform multiple feature linear regressions here:
target_variable = 'Winnings'
all_features = atp.columns.tolist()
all_features.remove(target_variable)
all_features.remove('Player')
y = atp[[target_variable]]

r2_list = []
#iterate through all combos of features
#for r in range(1, len(all_features) + 1):  kept timing out, reduced to multiple of 3 features.
for r in range(1,4):
  for feature_combinations in combinations(all_features, r):
    current_features = list(feature_combinations)
    #extract the current feature
    X = atp[current_features]
    #train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, test_size = 0.2, random_state = 6)
    #initiate model
    linreg = LinearRegression()
    #fit & predict
    linreg.fit(x_train, y_train)
    # Calc R2 values
    train_r2 = linreg.score(x_train,y_train)
    test_r2 = linreg.score(x_test,y_test)
    #append to list
    r2_list.append((current_features, train_r2, test_r2))
#sort list based on r2 test values in descening order
r2_list.sort(key=lambda x: x[2], reverse=True)
#choose best feature combo
best_feature_combo = r2_list[0]
print(best_feature_combo)
#Plot actual vs predicted for best
feature_combination, train_r2, test_r2 = best_feature_combo
X = atp[feature_combination]
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size = 0.8, test_size = 0.2, random_state = 6)
linreg = LinearRegression()
linreg.fit(x_train, y_train)
#plot actual vs predicted
y_pred = linreg.predict(x_test)
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([min(y_test.values), max(y_test.values)], [min(y_test.values), max(y_test.values)], linestyle='--', color='red')  # Diagonal line
plt.title(f'Actual vs Predicted ({", ".join(feature_combination)} as Features)\nTest R^2: {test_r2:.4f}')
plt.xlabel(f'Actual {target_variable}')
plt.ylabel(f'Predicted {target_variable}')
plt.show()
plt.clf()

