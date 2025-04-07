import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import seaborn as sns


""" Reading the csv file dataset from current folder """
df = pd.read_csv('surendran_g_uptor_final_project.csv')


""" To read and display the first 5 rows (by default) of dataset """
# pd.set_option("display.max_columns", None)
print(df.head())


""" To read the column name in the dataset """
print("Column Names:\n", df.columns)

""" To read the no. of rows and columns in dataset """
print("\nTotal no. of rows and columns in dataset:", df.shape)


""" To know the no. of columns in dataset in detail like column name, datatype """
print("\n\nDataset Information:")
df.info()


"""Statistical information about the dataset"""
# pd.set_option("display.max_columns", None)
print("\nStatistical information:\n", df.describe())


""" Finding null values in the dataset """
print("\n\nDataset null values:")
print(df.isnull().sum())


""" Compute correlation matrix """
correlation_matrix = df.drop(columns=['User_ID', 'Gender']).corr()

""" Create a correlation heatmap """
plt.figure(figsize=(12, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlations')
plt.show()


""" scatter plot Visualization """
plt.figure(figsize=(8, 6))
plt.scatter(x = df['Duration'], y = df['Calories'], color = "darkblue")
plt.title('Duration vs Calorie burned ')
plt.xlabel('Duration')
plt.ylabel('Calorie Burned')
plt.show()

""" Count plot Visualization """
plt.figure(figsize=(8, 6))
sns.countplot(x = df['Gender'], data=df, color="blue")
plt.title('Count of Male and Female data')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

""" Box plot Visualization """
plt.figure(figsize=(8, 6))
sns.boxplot(x = df['Gender'], y = df['Calories'], data=df)
plt.title('Distribution of Calories Burned based on Gender')
plt.xlabel('Gender')
plt.ylabel('Calorie Burned')
plt.show()


""" Converting Categorical data into numerical values """
lb = LabelEncoder()
df['Gender'] = lb.fit_transform(df['Gender'])
# print(df)


"""" Assigning x and y for training and testing"""
x = df.drop('Calories', axis=1)
y = df['Calories']

""" Splitting the dataset for training and testing """
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=97)


""" Predicting the data using the model """
model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


""" Evaluate the performance of the model """
accuracy = r2_score(y_test,y_pred)
print("\nAccuracy:", accuracy)

""" Predicting with new data """
new_data=pd.DataFrame({'User_ID':[18569603],'Gender':['male'],'Age':[32],'Height':[162],'Weight':[56],
                       'Duration':[20], 'Heart_Rate':[97], 'Body_Temp':[37.2]})

new_data['Gender'] = lb.fit_transform(new_data['Gender'])
predict_new_data = model.predict(new_data)
print("Calorie Prediction for new data:", predict_new_data)


""" Unsupervised Model Algorithm"""
#df['Gender'] = lb.inverse_transform(df['Gender'])
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(x)
df['PCA1'] = reduced_data[:, 0]
df['PCA2'] = reduced_data[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="PCA1", y="PCA2",
    hue="Gender", data=df,
    palette="viridis", s=50, legend= False
)

plt.title("PCA Visualization of Calorie Dataset")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()