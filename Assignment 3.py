#!/usr/bin/env python
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.5** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 3 - More Pandas
# This assignment requires more individual learning then the last one did - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

# ### Question 1 (20%)
# Load the energy data from the file `Energy Indicators.xls`, which is a list of indicators of [energy supply and renewable electricity production](Energy%20Indicators.xls) from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should be put into a DataFrame with the variable name of **energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude the footer and header information from the datafile. The first two columns are unneccessary, so you should get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']`
# 
# Convert `Energy Supply` to gigajoules (there are 1,000,000 gigajoules in a petajoule). For all countries which have missing data (e.g. data with "...") make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with numbers and/or parenthesis in their name. Be sure to remove these, 
# 
# e.g. 
# 
# `'Bolivia (Plurinational State of)'` should be `'Bolivia'`, 
# 
# `'Switzerland17'` should be `'Switzerland'`.
# 
# <br>
# 
# Next, load the GDP data from the file `world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# <br>
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `scimagojr-3.xlsx`, which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country, and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# 
# *This function should return a DataFrame with 20 columns and 15 entries.*

# In[6]:


import pandas as pd
import numpy as np
import re



energy = pd.read_excel('Energy Indicators.xls', skiprow=1, skipfooter=1)
energy.drop(['Unnamed: 0', 'Unnamed: 1'], axis = 1, inplace = True)
energy.columns =  ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
energy = energy.dropna()
energy['Country'] = energy['Country'].str.replace('\d+', '')  

en = energy[energy['Country'].str.contains('(', regex=False)]
for sen in en.index:
    s = energy.loc[sen]
    d = s['Country']
    s.loc['Country'] = s['Country'].split('(')[0]

for series in energy.index:
    s = energy.loc[series]
    sx = type(s.loc['Energy Supply'])
    sy = type(s.loc['Energy Supply per Capita'])
    sz = type(s.loc['% Renewable'])
    country = s.loc['Country']

    if (sx == str or sy == str or sz == str):
        s.loc['Energy Supply'] = np.float64(np.nan)
        s.loc['Energy Supply per Capita'] = np.float64(np.nan)
        s.loc['% Renewable'] = np.float64(np.nan)

    if (country == "Iran "):
        s['Country'] = 'Iran'

    if (country == "Republic of Korea"):
        s['Country'] = 'South Korea'

    elif (country == "United States of America"):
        s['Country'] = 'United States'

    elif (country == "United Kingdom of Great Britain and Northern Ireland"):
        s['Country'] = 'United Kingdom'

    elif (country == "China, Hong Kong Special Administrative Region"):
        s['Country'] = 'Hong Kong'

#     energy = energy.drop(8)
energy['Energy Supply'] = energy['Energy Supply']*1000000 
#     energy


GDP = pd.read_csv('world_bank.csv', skiprows=4, skipfooter=0)
GDP = GDP.rename(columns={'Country Name': 'Country'})
columns_to_keep = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
GDP = GDP[columns_to_keep]
GDP[GDP['Country'] == "Korea, Rep."] = 'South Korea'
GDP[GDP['Country'] == "Iran, Islamic Rep."] = 'Iran'
GDP[GDP['Country'] == "Hong Kong SAR, China"] = 'Hong Kong'
#     GDP
ScimEn = pd.read_excel('scimagojr-3.xlsx', skiprow=0, skipfooter=0)
ScimEn = ScimEn[0:15]

#     ScimEn

energy = energy.set_index('Country')
ScimEn = ScimEn.set_index('Country')
GDP = GDP.set_index('Country')
r = pd.merge(energy, GDP, how='inner', left_index=True, right_index=True)
result = pd.merge(r, ScimEn, how='inner', left_index=True, right_index=True)

for n in result.keys()[0:13]:
    for o in result[n]:

        if type(o) == str:
            result[n][o] = np.nan

    result[n] = np.float64(result[n])


# In[21]:


import pandas as pd
import numpy as np
import re

def answer_one():    
    energy = pd.read_excel('Energy Indicators.xls', skiprow=1, skipfooter=1)
    energy.drop(['Unnamed: 0', 'Unnamed: 1'], axis = 1, inplace = True)
    energy.columns =  ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy = energy.dropna()
    energy['Country'] = energy['Country'].str.replace('\d+', '')  

    en = energy[energy['Country'].str.contains('(', regex=False)]
    for sen in en.index:
        s = energy.loc[sen]
        d = s['Country']
        s.loc['Country'] = s['Country'].split('(')[0]

    for series in energy.index:
        s = energy.loc[series]
        sx = type(s.loc['Energy Supply'])
        sy = type(s.loc['Energy Supply per Capita'])
        sz = type(s.loc['% Renewable'])
        country = s.loc['Country']

        if (sx == str or sy == str or sz == str):
            s.loc['Energy Supply'] = np.float64(np.nan)
            s.loc['Energy Supply per Capita'] = np.float64(np.nan)
            s.loc['% Renewable'] = np.float64(np.nan)

        if (country == "Iran "):
            s['Country'] = 'Iran'

        if (country == "Republic of Korea"):
            s['Country'] = 'South Korea'

        elif (country == "United States of America"):
            s['Country'] = 'United States'

        elif (country == "United Kingdom of Great Britain and Northern Ireland"):
            s['Country'] = 'United Kingdom'

        elif (country == "China, Hong Kong Special Administrative Region"):
            s['Country'] = 'Hong Kong'

#     energy = energy.drop(8)
    energy['Energy Supply'] = energy['Energy Supply']*1000000 
#     energy


    GDP = pd.read_csv('world_bank.csv', skiprows=4, skipfooter=0)
    GDP = GDP.rename(columns={'Country Name': 'Country'})
    columns_to_keep = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
    GDP = GDP[columns_to_keep]
    GDP[GDP['Country'] == "Korea, Rep."] = 'South Korea'
    GDP[GDP['Country'] == "Iran, Islamic Rep."] = 'Iran'
    GDP[GDP['Country'] == "Hong Kong SAR, China"] = 'Hong Kong'
#     GDP
    ScimEn = pd.read_excel('scimagojr-3.xlsx', skiprow=0, skipfooter=0)
    ScimEn = ScimEn[0:15]
    
#     ScimEn

    energy = energy.set_index('Country')
    ScimEn = ScimEn.set_index('Country')
    GDP = GDP.set_index('Country')
    r = pd.merge(energy, GDP, how='inner', left_index=True, right_index=True)
    result = pd.merge(r, ScimEn, how='inner', left_index=True, right_index=True)
    
    for n in result.keys()[0:13]:
        for o in result[n]:
            
            if type(o) == str:
                result[n][o] = np.nan
        
        result[n] = np.float64(result[n])
        
    en = result    
    return result

# answer_one()


# ### Question 2 (6.6%)
# The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
# 
# *This function should return a single number.*

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<svg width="800" height="300">\n  <circle cx="150" cy="180" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="blue" />\n  <circle cx="200" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="red" />\n  <circle cx="100" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="green" />\n  <line x1="150" y1="125" x2="300" y2="150" stroke="black" stroke-width="2" fill="black" stroke-dasharray="5,3"/>\n  <text  x="300" y="165" font-family="Verdana" font-size="35">Everything but this!</text>\n</svg>')


# In[7]:


def answer_two():
    rr = pd.merge(energy, GDP, how='inner', left_index=True, right_index=True)
    return len(rr) - 162
# answer_two()


# <br>
# 
# Answer the following questions in the context of only the top 15 countries by Scimagojr Rank (aka the DataFrame returned by `answer_one()`)

# ### Question 3 (6.6%)
# What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
# 
# *This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*

# In[8]:


def answer_three():
    Top15 = result
    Top = Top15.copy()
#     Top.loc['Iran'] = np.nan
#     Top.loc['South Korea'] = np.nan
    Top = Top.apply(avg, axis = 1)
    Top = Top.sort_values('Average',axis = 0, ascending = False, na_position = 'last')
    avgGDP = pd.Series([])
    for series in Top.index:
        s = pd.Series([Top.loc[series]], index = [series])
        avgGDP = avgGDP.append(s)
#     print(avgGDP[0].name)
    return avgGDP

def avg(row):
    data = row[['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    avgGDP = pd.Series({'Average': np.mean(data), 'Average': np.mean(data)})
    
    return avgGDP

# answer_three()


# ### Question 4 (6.6%)
# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
# 
# *This function should return a single number.*

# In[10]:


def answer_four():
    Top15 = result
    Top = Top15.copy()
    columns_to_keep = ['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
    Top = Top[columns_to_keep]
    avgGDP = answer_three()
    country = avgGDP[5].name
    print (country)
    series = Top.loc[country]
    max = 0
    min = 0
    for numbers in series:
        if numbers > max:
            max = numbers
        
        if min == 0:
            min = numbers
        elif numbers < min:
            min = numbers
    return max - min
# answer_four()


# ### Question 5 (6.6%)
# What is the mean `Energy Supply per Capita`?
# 
# *This function should return a single number.*

# In[12]:


def answer_five():
    Top15 = result
    return np.mean(Top15['Energy Supply per Capita'])
# answer_five()


# ### Question 6 (6.6%)
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the percentage.*

# In[15]:


def answer_six():
    Top15 = result
    maximum = np.max(Top15['% Renewable'])
    countryDF = Top15[Top15['% Renewable'] == maximum]
    country = countryDF.index[0]
    res = (country, maximum)
    return res
# answer_six()


# ### Question 7 (6.6%)
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

# In[18]:


def answer_seven():
    Top15 = result
    Top15['Ratio'] = Top15['Self-citations'].div(Top15['Citations'])
    maximum = np.max(Top15['Ratio'])
    countryDF = Top15[Top15['Ratio'] == maximum]
    country = countryDF.index[0]
    res = (country, maximum)
    return res
# answer_seven()


# ### Question 8 (6.6%)
# 
# Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
# What is the third most populous country according to this estimate?
# 
# *This function should return a single string value.*

# In[48]:


import math
def answer_eight():
    Top15 = result
    Top15['Population'] = Top15['Energy Supply'].div(Top15['Energy Supply per Capita'])
    Top15 = Top15.sort_values('Population',axis = 0, ascending = False, na_position = 'last')
    s = Top15.iloc[2].name
    return s
answer_eight()


# ### Question 9 (6.6%)
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the `.corr()` method, (Pearson's correlation).
# 
# *This function should return a single number.*
# 
# *(Optional: Use the built-in function `plot9()` to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)*

# In[22]:


def answer_nine():
    Top15 = result
    Top15 = Top15.copy()
    Top15['Population'] = Top15['Energy Supply'].div(Top15['Energy Supply per Capita'])
    Top15['Document/Capita'] = Top15['Citable documents'].div(Top15['Population'])
    Top15['Energy Supply per Capita'] = np.float64(Top15['Energy Supply per Capita'])
    Top15['Document/Capita'] = np.float64(Top15['Document/Capita'])
    row = ['Energy Supply per Capita', 'Document/Capita']
    Top15 = Top15[row]
    return Top15.corr().loc['Document/Capita']['Energy Supply per Capita']
# answer_nine()


# In[ ]:


def plot9():
    import matplotlib as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


# In[ ]:


# plot9() # Be sure to comment out plot9() before submitting the assignment!


# ### Question 10 (6.6%)
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
# 
# *This function should return a series named `HighRenew` whose index is the country name sorted in ascending order of rank.*

# In[23]:


import math
def answer_ten():
    Top15 = result
    Top15 = Top15.sort_values('Rank',axis = 0, ascending = True, na_position = 'last')
    median = np.median(Top15['% Renewable'])
    for series in Top15.index:
        x = Top15.loc[series, '% Renewable']
        if (x >= median):  
            Top15.loc[series, 'Mean'] = 1
        elif x < median:
            Top15.loc[series, 'Mean'] = 0 
    HighRenew = Top15['Mean']
    for n in HighRenew.index:
        HighRenew[n] = int(math.ceil(HighRenew[n])*10/10)

    
    return HighRenew
# answer_ten()


# ### Question 11 (6.6%)
# Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`*

# In[25]:


def answer_eleven():
    Top15 = result
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    
    s = ContinentDict['Brazil']
    
    for series in Top15.index:
        Top15.loc[series, 'Continent'] = ContinentDict[series]
    Top15 = Top15.reset_index()
    Top15 = Top15.sort_values('Continent',axis = 0, ascending = True, na_position = 'last')
    for x in Top15.index:
        Top15.loc[x, 'Size'] = len(Top15.loc[x])        

    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15 = Top15.set_index('Continent').groupby(level=0)['PopEst'].agg({'size': np.size, 'sum': np.sum, 'mean': np.mean, 'std': np.std})
    Top15 = Top15[['size', 'sum', 'mean', 'std']]
    
    return Top15
# answer_eleven()


# ### Question 12 (6.6%)
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
# 
# *This function should return a __Series__ with a MultiIndex of `Continent`, then the bins for `% Renewable`. Do not include groups with no countries.*

# In[27]:


def answer_twelve():
    Top15 = result
    
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    for series in Top15.index:
        Top15.loc[series, 'Continent'] = ContinentDict[series]
        
    Top15 = Top15.reset_index()
    Top15 = Top15.sort_values('Continent',axis = 0, ascending = True, na_position = 'last')
    Top15['Bins'] = pd.cut(Top15['% Renewable'],5)
    Top15 = Top15.groupby(['Continent', 'Bins']).size()
    return Top15
# answer_twelve()


# ### Question 13 (6.6%)
# Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
# 
# e.g. 317615384.61538464 -> 317,615,384.61538464
# 
# *This function should return a Series `PopEst` whose index is the country name and whose values are the population estimate string.*

# In[32]:


def answer_thirteen():
    Top15 = result
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    
    p = []
    for i in Top15['PopEst']:
        i = str(i)
        s = i.split('.')[0]
        ss = i.split('.')[1]        
        if (len(s) == 8):
            sa = s[len(s)-8:len(s)-7]+s[len(s)-7:len(s)-6]+','+s[len(s)-6:len(s)-5]+s[len(s)-5:len(s)-4]+s[len(s)-4:len(s)-3]+','+s[len(s)-3:len(s)-2]+s[len(s)-2:len(s)-1]+s[len(s)-1:len(s)]+'.'+ss
            
        if (len(s) == 9):
            sa = s[len(s)-9:len(s)-8]+s[len(s)-8:len(s)-7]+s[len(s)-7:len(s)-6]+','+s[len(s)-6:len(s)-5]+s[len(s)-5:len(s)-4]+s[len(s)-4:len(s)-3]+','+s[len(s)-3:len(s)-2]+s[len(s)-2:len(s)-1]+s[len(s)-1:len(s)]+'.'+ss
        
        if (len(s) == 10):
            sa = s[len(s)-10:len(s)-9]+','+s[len(s)-9:len(s)-8]+s[len(s)-8:len(s)-7]+s[len(s)-7:len(s)-6]+','+s[len(s)-6:len(s)-5]+s[len(s)-5:len(s)-4]+s[len(s)-4:len(s)-3]+','+s[len(s)-3:len(s)-2]+s[len(s)-2:len(s)-1]+s[len(s)-1:len(s)]+'.'+ss   
        p.append(sa)
    Top15['String'] = [str(y) for y in p]
    return Top15['String']
# answer_thirteen()


# ### Optional
# 
# Use the built in function `plot_optional()` to see an example visualization.

# In[ ]:


def plot_optional():
    import matplotlib as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    Top15 = answer_one()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' 2014 GDP, and the color corresponds to the continent.")


# In[ ]:


# plot_optional() # Be sure to comment out plot_optional() before submitting the assignment!


# In[ ]:




