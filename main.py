import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Assuming 'data' is your DataFrame
# Replace it with your actual dataset or data loading code
data = pd.read_csv('Dataset/Data_UAS.csv', sep=',', header=0, engine='python', encoding='utf-8')

# Soal A
# Descriptive statistics
descriptive_stats = data.describe()

# Display the results
print(descriptive_stats)

# Interpretasi
print("\nInterpretasi Statistik Deskriptif:")
print("===================================")

# Hari (Day)
print("\nHari (Day):")
print("Data ini mencakup 548 observasi dengan nilai rata-rata hari sebesar {:.2f}.".format(descriptive_stats.loc['mean', 'Day']))

# Interaksi (Interaction)
print("\nInteraksi (Interaction):")
print("Variabel interaksi memiliki nilai rata-rata {:.2f} dengan deviasi standar {:.2f}.".format(descriptive_stats.loc['mean', 'Interaction'], descriptive_stats.loc['std', 'Interaction']))

# Tempat Tinggal (Residences)
print("\nTempat Tinggal (Residences):")
print("Variabel Tempat Tinggal memiliki total 704 dengan nilai rata-rata {:.2f}.".format(descriptive_stats.loc['mean', 'Residences']))

# Pengetahuan (Knowledge)
print("\nPengetahuan (Knowledge):")
print("Variabel Pengetahuan memiliki nilai rata-rata {:.2f} dengan deviasi standar {:.2f}.".format(descriptive_stats.loc['mean', 'Knowledge'], descriptive_stats.loc['std', 'Knowledge']))

# Curah Hujan (Rainfall)
print("\nCurah Hujan (Rainfall):")
print("Variabel Curah Hujan memiliki nilai minimum {:.2f} dan nilai maksimum {:.2f}.".format(descriptive_stats.loc['min', 'Rainfall'], descriptive_stats.loc['max', 'Rainfall']))

# Kelembaban (%) (Humidity (%))
print("\nKelembaban (%):")
print("Variabel Kelembaban memiliki nilai rata-rata {:.2f} dengan deviasi standar {:.2f}.".format(descriptive_stats.loc['mean', 'Humidity (%)'], descriptive_stats.loc['std', 'Humidity (%)']))

# Suhu (Temperature)
print("\nSuhu (Temperature):")
print("Variabel Suhu memiliki nilai rata-rata {:.2f} dengan deviasi standar {:.2f}.".format(descriptive_stats.loc['mean', 'Temperature'], descriptive_stats.loc['std', 'Temperature']))

# Daya (Power)
print("\nDaya (Power):")
print("Variabel Daya memiliki nilai rata-rata {:.2f} dan nilai maksimum {:.2f}.".format(descriptive_stats.loc['mean', 'Power'], descriptive_stats.loc['max', 'Power']))

print("\n")

# Soal B
# Check validity and reliability (example: using correlation matrix)
correlation_matrix = data.corr()

# Display the correlation matrix
print("\nMatrix Korelasi:")
print(correlation_matrix)

# Assuming 'data' is your DataFrame
# Replace it with your actual dataset or data loading code

# Soal C
# Descriptive statistics for specific columns
power_stats = data['Power'].describe()
humidity_stats = data['Humidity (%)'].describe()
rainfall_stats = data['Rainfall'].mean()

# Display the results
print("\nPower Stats:", power_stats)
print("Humidity Stats:", humidity_stats)
print("Rainfall Mean:", rainfall_stats)

# Soal D

data = data.dropna()

numeric_columns = data.select_dtypes(include=[float, int]).columns

for column in numeric_columns:
    skewness_value = data[column].skew()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    sns.histplot(data[column], kde=True, color='green', edgecolor='black')
    plt.annotate(f'Skewness: {skewness_value:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)
    plt.title(f'Skewness Plot - {column}')
    plt.xlabel(column)
    plt.ylabel('Frekuensi')
    plt.legend()


# Compute skewness
skewness_power = skew(data['Power'])
skewness_humidity = skew(data['Humidity (%)'])
skewness_rainfall = skew(data['Rainfall'])

# Display skewness
print("Skewness - Power:", skewness_power)
print("Skewness - Humidity:", skewness_humidity)
print("Skewness - Rainfall:", skewness_rainfall)


from sklearn.preprocessing import StandardScaler


# Soal E
# Standardize data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data[['Power', 'Humidity (%)', 'Rainfall']])

print(data_standardized)
# Use data_standardized for further analysis

import statsmodels.api as sm


# Soal F
# Simple linear regression
X = sm.add_constant(data[['Day','Interaction','Residences','Knowledge','Rainfall','Humidity (%)','Temperature' ]])
y = data['Power']

model = sm.OLS(y, X).fit()
print(model.summary())

# 3. Uji Parsial (t-test) untuk variabel 'Popularity' pada data 1000 sample
t_stat, p_value = stats.ttest_ind(data['Popularity'], data['Members'])
print(f"\nUji Parsial (t-test) untuk variabel 'Popularity' pada data 1000 sample:")
print(f"T-Stat: {t_stat}, P-Value: {p_value}")
if p_value < 0.05:
    print("Variabel 'Popularity' signifikan secara parsial.")
else:
    print("Variabel 'Popularity' tidak signifikan secara parsial.")

# 4. Uji Simultan Regresi (F-test) untuk model dengan 10 variabel numerik
f_test = model.wald_test("Score = Episodes = Premiered = Ranked = Popularity = Favorites = Watching = Completed = On-Hold = Dropped = 0")
f_stat = f_test.statistic[0][0]
f_p_value = f_test.pvalue

print(f"\nUji Simultan Regresi (F-test) untuk model dengan 10 variabel numerik:")
format(f_p_value, '.6f')
print(f"F-Stat: {f_stat}, P-Value: {f_p_value}")
if f_p_value < 0.05:
    print("Model regresi secara keseluruhan signifikan.")
else:
    print("Model regresi secara keseluruhan tidak signifikan.")

# 5. Uji Kebaikan Model menggunakan R-squared
r_squared = model.rsquared
print(f"\nR-Squared (Koefisien Determinasi) untuk model dengan 10 variabel numerik:")
print(f"R-Squared: {r_squared}")