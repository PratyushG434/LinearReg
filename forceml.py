import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data_path = "./force_data.csv"

data = pd.read_csv(data_path)

print(data.head())

my_model = LinearRegression()

X = data['Acceleration'].values.reshape(-1, 1) 
y = data['Force']

my_model.fit(X,y)

mass = my_model.coef_[0]

print(mass)

plt.scatter(data['Acceleration'], data['Force'], color='blue', label='Data Points')

# Plot the regression line
predicted_force = my_model.predict(X)
plt.plot(data['Acceleration'], predicted_force, color='red', label='Best-Fit Line')

# Add labels and title
plt.xlabel('Acceleration (m/s²)')
plt.ylabel('Force (N)')
plt.title('Force vs. Acceleration')

# Show legend
plt.legend()

# Show the plot
plt.show()