import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.impute import SimpleImputer



df = pd.read_csv("football_player_data.csv")

print(df.head())
"========================================================"
print(df.info())
"========================================================"
print(df.tail())

"=================================== LINEAR REGRESSION MODEL ====================================="

from sklearn.model_selection import train_test_split
df = pd.read_csv("football_player_data.csv")
features = ['Heading Accuracy', 'Short Passing','Long Passing', 'Skill','Dribbling', 'Curve', 'Volleys', 'Ball Control', 'Crossing', 'FK Accuracy']
X = df[features]
y = df['Attacking']
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

"====================================LINEAR REGRESSION HEADING ACCURACY============================================="

features = ['Heading Accuracy']
colors = ['red', 'yellow', 'green', 'orange', 'violet', 'pink', 'violet', 'brown', 'purple', 'grey']

for i in range(len(features)):
    plt.scatter(X_test[:, i], y_test, color='blue', label='Actual Data')
    plt.plot(X_test[:, i], model.predict(X_test), color=colors[i], label='Regression Line')
    plt.xlabel(features[i])
    plt.ylabel('Attacking')
    plt.title(f'Linear Regression ({features[i]})')
    plt.legend()
    plt.show()

"=====================================LINEAR REGRESSION SHORT PASSING ============================================================"

features = ['Short Passing']
colors = ['red', 'yellow', 'green', 'orange', 'violet', 'pink', 'violet', 'brown', 'purple', 'grey']

for i in range(len(features)):
     plt.scatter(X_test[:, i], y_test, color='blue', label='Actual Data')
     plt.plot(X_test[:, i], model.predict(X_test), color=colors[i], label='Regression Line')
     plt.xlabel(features[i])
     plt.ylabel('Attacking')
     plt.title(f'Linear Regression ({features[i]})')
     plt.legend()
     plt.show()

"======================================LINEAR REGRESSION LONG PASSING=============================================================="
features = ['Long Passing']
colors = ['red', 'yellow', 'green', 'orange', 'violet', 'pink', 'violet', 'brown', 'purple', 'grey']

for i in range(len(features)):
    plt.scatter(X_test[:, i], y_test, color='blue', label='Actual Data')
    plt.plot(X_test[:, i], model.predict(X_test), color=colors[i], label='Regression Line')
    plt.xlabel(features[i])
    plt.ylabel('Attacking')
    plt.title(f'Linear Regression ({features[i]})')
    plt.legend()
    plt.show()

"======================================LINEAR REGRESSION SKILL========================================================================"
features = ['Skill']
colors = ['red', 'yellow', 'green', 'orange', 'violet', 'pink', 'violet', 'brown', 'purple', 'grey']

for i in range(len(features)):
    plt.scatter(X_test[:, i], y_test, color='blue', label='Actual Data')
    plt.plot(X_test[:, i], model.predict(X_test), color=colors[i], label='Regression Line')
    plt.xlabel(features[i])
    plt.ylabel('Attacking')
    plt.title(f'Linear Regression ({features[i]})')
    plt.legend()
    plt.show()

"====================================LINEAR REGRESSION DRIBBLING===================================================================="
features = ['Dribbling']
colors = ['red', 'yellow', 'green', 'orange', 'violet', 'pink', 'violet', 'brown', 'purple', 'grey']

for i in range(len(features)):
    plt.scatter(X_test[:, i], y_test, color='blue', label='Actual Data')
    plt.plot(X_test[:, i], model.predict(X_test), color=colors[i], label='Regression Line')
    plt.xlabel(features[i])
    plt.ylabel('Attacking')
    plt.title(f'Linear Regression ({features[i]})')
    plt.legend()
    plt.show()

"===========================================LINEAR REGRESSION BALL CONTROL==============================================="
features = ['Ball Control']
colors = ['red', 'yellow', 'green', 'orange', 'violet', 'pink', 'violet', 'brown', 'purple', 'grey']

for i in range(len(features)):
    plt.scatter(X_test[:, i], y_test, color='blue', label='Actual Data')
    plt.plot(X_test[:, i], model.predict(X_test), color=colors[i], label='Regression Line')
    plt.xlabel(features[i])
    plt.ylabel('Attacking')
    plt.title(f'Linear Regression ({features[i]})')
    plt.legend()
    plt.show()

"==============================================================================================================================="

df = pd.read_csv("football_player_data.csv")
features = ['Heading Accuracy', 'Short Passing', 'Long Passing', 'Skill','Dribbling', 'Curve', 'Volleys', 'Ball Control', 'Crossing', 'FK Accuracy']
X = df[features]
y = df['Attacking']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print ('The range of heading accuracy is between 5 to 93')
Heading_Accuracy = int(input("Enter the heading accuracy: "))

print ('The range of short passing is between 7 to 94')
Short_Passing = int(input("Enter the short passing: "))

print('The range of long passing is between 5 to 93')
Long_Passing = int(input("Enter the long passing: "))

print('The range of skill is between 40 to 470')
Skill = int(input("Enter the skill: "))

print('The range of dribbling is between 5 to 96')
Dribbling = int(input("Enter the dribbling: "))

print('The range of curve is between 4 to 94')
Curve = int(input("Enter the curve: "))


print ('The range of volleys is between 3 to 90')
Volleys = int(input("Enter the volleys: "))

print ('The range of ball control is between 5 to 96')
Ball_Control = int(input("Enter the ball control: "))

print ('The range of crossing is between 6 to 94')
Crossing = int(input("Enter the crossing: "))

print ('The range of free-kick accuracy is between 5 to 94')
FK_Accuracy = int(input("Enter the free-kick accuracy: "))

input_data = pd.DataFrame({'Heading_Accuracy': [Heading_Accuracy], 'Short_Passing': [Short_Passing],
                           'Long_Passing': [Long_Passing], 'Skill': [Skill],
                           'Dribbling': [Dribbling], 'Curve': [Curve],
                           'Volleys': [Volleys], 'Ball_Control': [Ball_Control],
                           'Crossing': [Crossing], 'FK_Accuracy': [FK_Accuracy]})
predicted_attacking = model.predict(input_data)
print('The range of attacking is 40 to 460')
print("Predicted Attacking:", predicted_attacking[0])


"==================================================================================================================================="

accuracy = model.score(X_test, y_test)
print("Accuracy (R-squared):", accuracy)
print("Accuracy (Percentage): {:.2f}%".format(accuracy * 100))
X = df[['Finishing']]
y = df['Attacking']
x_mean=np.mean(X['Finishing']) # Calculate mean of the 'features' column
y_mean=np.mean(y)


"========================ACCURACY OUTPUT========================"

"Accuracy (R-squared): 0.9892924673600899"
"Accuracy (Percentage): 98.93%"


"======================================================================================================================="














































