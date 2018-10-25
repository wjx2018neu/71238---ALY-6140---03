
import pandas as pd
from datetime import datetime
import calendar
import matplotlib.pyplot as plt

def strtostamp(str):
    datetime_object = datetime.strptime(str, '%Y-%m-%d  %H:%M:%S')
    return calendar.timegm(datetime_object.utctimetuple())

energydata_complete = pd.read_csv("energydata_complete.csv")
energydata_complete.head()

# check if the records in column "date" is correct
for i in range(len(energydata_complete) - 1):
    if strtostamp(energydata_complete.iloc[i, 0]) + 600 != strtostamp(energydata_complete.iloc[i + 1, 0]):
        print("date wrong with index {0}".format(i))

energydata_complete.describe()

pd.isnull(energydata_complete).head()
pd.isnull(energydata_complete).sum()

fig, axs = plt.subplots(1, 5, sharey=True)
axs[0].hist(energydata_complete['T1'], bins=20)
axs[1].hist(energydata_complete['T2'], bins=20)
axs[2].hist(energydata_complete['T3'], bins=20)
axs[3].hist(energydata_complete['T4'], bins=20)
axs[4].hist(energydata_complete['T5'], bins=20)

fig, axs = plt.subplots(1, 5, sharey=True)
axs[0].hist(energydata_complete['RH_1'], bins=20)
axs[1].hist(energydata_complete['RH_2'], bins=20)
axs[2].hist(energydata_complete['RH_3'], bins=20)
axs[3].hist(energydata_complete['RH_4'], bins=20)
axs[4].hist(energydata_complete['RH_5'], bins=20)


print("min of \"Appliances\" is: {0} \nmax of \"Appliances\" is: {1}".format(
energydata_complete["Appliances"].min(),
energydata_complete["Appliances"].max())
)
print(energydata_complete["Appliances"].unique())
print(energydata_complete["Appliances"].value_counts().size)

plt.hist(energydata_complete['Appliances'], bins=20)

print("min of \"lights\" is: {0} \nmax of \"lights\" is: {1}".format(
energydata_complete["lights"].min(),
energydata_complete["lights"].max())
)
print(energydata_complete["lights"].unique())
print(energydata_complete["lights"].value_counts().size)

plt.hist(energydata_complete['lights'], bins=20)

print((energydata_complete['lights'] == 0).sum())/len(energydata_complete)
