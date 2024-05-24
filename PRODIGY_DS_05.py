#!/usr/bin/env python
# coding: utf-8

# # TASK 5

# AIM: To explore and understand the patterns and correlations within a dataset containing information about traffic accidents

# In[1]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium


# In[2]:


# Load the data
file_path = r"C:\Users\rosha\Downloads\RTA Dataset.csv\RTA Dataset.csv"
accident_data = pd.read_csv(file_path)


# In[3]:


# Convert 'Time' column to datetime format and extract hour
accident_data['Time'] = pd.to_datetime(accident_data['Time'], format='%H:%M:%S')
accident_data['Hour'] = accident_data['Time'].dt.hour


# In[4]:


# Display the first few rows of the dataset
accident_data.head()


# In[5]:


# Check the data types and missing values
accident_data.info()


# In[6]:


# Basic statistics
accident_data.describe()


# In[7]:


# Check for missing values
missing_values = accident_data.isnull().sum()
missing_values


# In[8]:


# Plot the distribution of accidents by day of the week
plt.figure(figsize=(10, 6))
sns.countplot(x='Day_of_week', data=accident_data, palette='viridis')
plt.title('Distribution of Accidents by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Accidents')
plt.show()


# In[9]:


# Plot the distribution of accidents by age band of drivers
plt.figure(figsize=(10, 6))
sns.countplot(x='Age_band_of_driver', data=accident_data, palette='viridis')
plt.title('Distribution of Accidents by Age Band of Drivers')
plt.xlabel('Age Band of Driver')
plt.ylabel('Number of Accidents')
plt.show()


# In[10]:


# Plot the distribution of accidents by road surface conditions
plt.figure(figsize=(10, 6))
sns.countplot(x='Road_surface_conditions', data=accident_data, palette='viridis')
plt.title('Distribution of Accidents by Road Surface Conditions')
plt.xlabel('Road Surface Conditions')
plt.ylabel('Number of Accidents')
plt.show()


# In[11]:


# Pie chart: Distribution of accidents by sex of driver
plt.figure(figsize=(8, 8))
accident_data['Sex_of_driver'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Distribution of Accidents by Sex of Driver')
plt.ylabel('')  # Hide y-label
plt.show()


# In[12]:


# Drop non-numeric columns for correlation analysis
numeric_columns = accident_data.select_dtypes(include=['int64', 'float64']).columns
accident_data_numeric = accident_data[numeric_columns]

# Compute correlation matrix
correlation_matrix = accident_data_numeric.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# Correlation analysis reveals relationships between numerical features, helping to understand which features may be more closely related.

# In[13]:


# Select specific non-numeric columns for correlation analysis
selected_columns = ['Day_of_week', 'Weather_conditions', 'Light_conditions', 'Type_of_collision', 'Cause_of_accident']

# Compute correlation matrix for the selected non-numeric columns
correlation_matrix_selected = accident_data[selected_columns].apply(lambda x: x.factorize()[0]).corr()

# Plot the heatmap for the selected non-numeric columns
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_selected, cmap='viridis', annot=True, fmt=".2f")
plt.title('Correlation Heatmap of Selected Non-Numeric Features')
plt.xlabel('Selected Non-Numeric Features')
plt.ylabel('Selected Non-Numeric Features')
plt.show()


# In[14]:


import warnings

# Suppress warnings for this code block
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Pair Plot: Scatterplot matrix of numerical features
    numerical_features = accident_data.select_dtypes(include=['int64', 'float64']).columns
    pairplot = sns.pairplot(accident_data[numerical_features])
    pairplot.fig.suptitle('Pair Plot of Numerical Features', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust figure layout to avoid warning
    plt.show()


# The pair plot offers a visual summary of the relationships between numerical features, which can aid in identifying any potential patterns or trends in the data.

# Overall, this analysis provides valuable information for understanding the factors associated with road traffic accidents and their potential relationships. Further analysis or modeling could be performed based on these insights to improve road safety measures or accident prediction models.
