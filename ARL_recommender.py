##########################################################################################################
# ASSOCIATION RULE LEARNING RECOMMENDER
##########################################################################################################
# Business Problem: Suggesting products to users at the basket stage
##########################################################################################################
# NOTE:
# Basket information of 3 different users is given below
# ▪ Product ID of User 1: 21987
# ▪ Product ID of User 2: 23235
# ▪ Product ID of User 3: 22747

# Make the most appropriate product recommendation for this basket information.
# (Product recommendations can be one or more).
# Important note: Create decision rules from 2010-2011 Germany customers.

##########################################################################################################
# Importing necessary libraries and modules
# Making necessary adjustments for the representation of the dataset
##########################################################################################################
# !pip install mlxtend
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

##########################################################################################################
# ASSIGNMENT 1: Perform Data Preprocessing
##########################################################################################################
# NOTE: Select 2010-2011 data and preprocess all data.
##########################################################################################################
# To download the dataset:
# https://archive.ics.uci.edu/ml/machine-learning-databases/00502/

df_ = pd.read_csv("C:\Career\VBO DSMLBootcamp-6\Ders_notlari\HAFTA_03\online_retail_II(2010-2011).csv")
df = df_.copy()
df.info()
df.isnull().sum()
df.head()

# Let's drop "Unnamed: 0" column as values are the same with the indices
df.drop(["Unnamed: 0"], axis=1, inplace=True)

df.head()

"""
Head of dataframe for 5 rows:
  
   Invoice StockCode                          Description  Quantity          InvoiceDate  Price  Customer ID         Country
0  536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6  2010-12-01 08:26:00   2.55      17850.0  United Kingdom
1  536365     71053                  WHITE METAL LANTERN         6  2010-12-01 08:26:00   3.39      17850.0  United Kingdom
2  536365    84406B       CREAM CUPID HEARTS COAT HANGER         8  2010-12-01 08:26:00   2.75      17850.0  United Kingdom
3  536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6  2010-12-01 08:26:00   3.39      17850.0  United Kingdom
4  536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6  2010-12-01 08:26:00   3.39      17850.0  United Kingdom

"""

def outlier_thresholds(dataframe, variable):
    """

    Determines the outlier thresholds for a given dataframe and variable.
    It calculates quantile1 (Q1), quantile3 (Q3), interquantile_range (Q3-Q1) values
    according to % 1 quantile values (low) and % 99 quantile values (up).
    Defines threshold values as low_limit and up_limit.

    Parameters
    ----------
    dataframe: dataframe
        The dataframe from which the variable is taken.
    variable: str
        The column name of the given dataframe.

    Returns
    -------
    low_limit: float64
        This is the lowest threshold.
        low_limit = quartile1 - 1.5 * interquantile_range
    up_limit: float64
        This is the upper threshold.
        up_limit = quartile3 + 1.5 * interquantile_range
    Notes
    -------
        quartile1 = dataframe[variable].quantile(0.01)
        quartile3 = dataframe[variable].quantile(0.99)

    """

    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    """

    Changes the low and up limits of the given variable of the dataframe
    according to the thresholds calculated with outlier_thresholds function

    Parameters
    ----------
    dataframe: dataframe
        The dataframe from which the variable is taken.
    variable: str
        The column name of the given dataframe.

    Returns
    -------
    None

    """

    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    """
    Implements below prepocessing transactions to the given dataframe:
        - Drops NA values.
        - Removes the invoices containing "C" from dataframe as they are cancelled transactions.
        - Removes the values of "Quantity" which are less than 0 (zero)
        - Removes the values of "Price" which are less than 0 (zero)
        - Replaces the thresholds of "Quantity" and "Price"  using replace_with_thresholds function.

    Parameters
    ----------
    dataframe: dataframe
        The dataframe which will be preprocessed.

    Returns
    -------
    dataframe: dataframe
        The dataframe which is preprocessed.
    """

    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)
df.info()

"""

BEFORE PREPROCESSING:                               AFTER PREPROCESSING:  

RangeIndex: 541910 entries, 0 to 541909             Int64Index: 397885 entries, 0 to 541909
Data columns (total 9 columns):                     Data columns (total 8 columns):
 #   Column       Non-Null Count   Dtype            #   Column       Non-Null Count   Dtype 
---  ------       --------------   -----            ---  ------       --------------   -----
 0   Unnamed: 0   541910 non-null  int64            0   Invoice      397885 non-null  object
 1   Invoice      541910 non-null  object           1   StockCode    397885 non-null  object
 2   StockCode    541910 non-null  object           2   Description  397885 non-null  object
 3   Description  540456 non-null  object           3   Quantity     397885 non-null  float64
 4   Quantity     541910 non-null  int64            4   InvoiceDate  397885 non-null  object
 5   InvoiceDate  541910 non-null  object           5   Price        397885 non-null  float64
 6   Price        541910 non-null  float64          6   Customer ID  397885 non-null  float64
 7   Customer ID  406830 non-null  float64          7   Country      397885 non-null  object 
 8   Country      541910 non-null  object           

"""

##########################################################################################################
# ASSIGNMENT 2: Generate association rules through Germany customers.
##########################################################################################################
# Select Germany customers
df_grm = df[df['Country'] == "Germany"]
df_grm.head()

"""
Head of Germany customers dataframe for 5 rows:

 Invoice     StockCode                          Description  Quantity          InvoiceDate  Price  Customer ID  Country
1109  536527     22809              SET OF 6 T-LIGHTS SANTA       6.0  2010-12-01 13:04:00   2.95      12662.0  Germany
1110  536527     84347  ROTATING SILVER ANGELS T-LIGHT HLDR       6.0  2010-12-01 13:04:00   2.55      12662.0  Germany
1111  536527     84945   MULTI COLOUR SILVER T-LIGHT HOLDER      12.0  2010-12-01 13:04:00   0.85      12662.0  Germany
1112  536527     22242        5 HOOK HANGER MAGIC TOADSTOOL      12.0  2010-12-01 13:04:00   1.65      12662.0  Germany
1113  536527     22244           3 HOOK HANGER MAGIC GARDEN      12.0  2010-12-01 13:04:00   1.95      12662.0  Germany

"""


def create_invoice_product_df(dataframe, id=False):
    """

    Function to create invoice-product matrix which is the key for ARL recommendation.
    Uses groupby(), sum(), unstack(), fillna() and applymap() functions.

    Parameters
    ----------
    dataframe: dataframe
        The dataframe which will be preprocessed.

    id: bool (default value is False)
        if True: Creates invoice-product matrix by implementing groupby() with "Invoice" and "StockCode" variables.
        if False: Creates invoice-product matrix by implementing groupby() with "Invoice" and "Description" variables.
    Returns
    -------
    None

    """

    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)



def check_id(dataframe, stock_code):
    """
    Function to check the ID of the product and print the name of the product

    Parameters
    ----------
    dataframe: dataframe
        The dataframe which will be preprocessed.
    stock_code: str
        The stock code of the product.
    Returns
    -------
    None

    """

    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, min_support=False, country="Germany"):
    """
    Function to create association rules using Apriori algorithm for a given database and country information.
    Apriori algorithm finds frequent itemsets for Booelan association rules.

    Parameters
    ----------
    dataframe: dataframe
        Dataframe given for creating association rules.
    id: Bool (default value is True)
        if True: Generates "antecedents" and "consequents" according to their IDs.
        if False: Generates "antecedents" and "consequents" according to their names ("Description"
        column of the dataframe)

    country: str
        Country name of the product.

    Returns
    -------
    rules: dataframe
        Association rules generated from frequent itemsets.
        Generated through association_rules function which has "support" metric as default.

    Notes
    -------
    "Apriori algorithm is proposed by R. Agrawal and R. Srikant in 1994 for mining frequent itemsets
    for Boolean association rules. The name of the algorithm is based on the fact that the algorithm
    uses prior knowledge of frequent itemset properties. Apriori employs an iterative approach known as
    a level-wise search, where k-itemsets are used to explore (k+1) itemsets."

    Ref: J. Han, M. Kamber and J. Pei, Data Mining Concepts and Techniques, Morgan Kaufmann,
    Third Edition, p.248, 2012.

    Explanations regarding the association rules:
    Antecedents: First product (itemset).
    Consequents: Second product (itemset) (the one following the first product).
    Antecedent support: Probability of first product appear in the data.
    Consequent support: Probability of first product appear in the data.
    Support: The probability of both products appearing in the data together.
    Confidence: The probability of purchasing the second product when the first product is purchased
    Lift: When the first product is purchased,
    the probability of purchasing the second product increases by "lift" times.
    Leverage; Similar to lift. Gives priority to the products having higher support values.
    Leverage is not prefferd if we have lift. Example: Leverage is the expected frequency of product-1
    without product-2.
    Conviction values are out of the scope of this assignment.

    """

    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    if min_support:
        frequent_itemsets = apriori(dataframe, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


# Creating rules for the customers in Germany with default min_support = 0.01
rules = create_rules(df, country="Germany")

len(rules)  # We have 49612 rules

rules.sort_values("lift", ascending=False).head(10)

"""
Associaton rules generated through min_support = 0.01 

                 antecedents            consequents  antecedent support  consequent support   support  confidence  lift  leverage  conviction
24758         (21989, 21086)         (21987, 21988)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
39030  (21989, 21094, 21987)         (21086, 21988)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
24763         (21987, 21988)         (21989, 21086)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
24762         (21086, 21988)         (21989, 21987)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
24759         (21989, 21987)         (21086, 21988)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
39025  (21094, 21086, 21988)         (21989, 21987)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
39029  (21094, 21987, 21988)         (21989, 21086)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
39026  (21989, 21094, 21086)         (21987, 21988)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
39040         (21989, 21086)  (21094, 21987, 21988)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
39044         (21989, 21987)  (21094, 21086, 21988)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf


"""

# Let' try create_rules function with min_support = 0.04
rules2 = create_rules(df, min_support= 0.04, country="Germany")

len(rules2) # We have 268 rules

rules2.sort_values("lift", ascending=False).head(10)

"""

Associaton rules generated through min_support = 0.04 

        antecedents     consequents  antecedent support  consequent support   support  confidence       lift  leverage  conviction
22          (21094)         (21086)            0.056893            0.052516  0.045952    0.807692  15.379808  0.042964    4.926915
23          (21086)         (21094)            0.052516            0.056893  0.045952    0.875000  15.379808  0.042964    7.544858
155  (85099B, POST)         (20712)            0.063457            0.100656  0.043764    0.689655   6.851574  0.037376    2.897885
158         (20712)  (85099B, POST)            0.100656            0.063457  0.043764    0.434783   6.851574  0.037376    1.656960
160   (POST, 20719)         (20724)            0.115974            0.070022  0.054705    0.471698   6.736439  0.046584    1.760316
165         (20724)   (POST, 20719)            0.070022            0.115974  0.054705    0.781250   6.736439  0.046584    4.041263
6           (20719)         (20724)            0.126915            0.070022  0.059081    0.465517   6.648168  0.050194    1.739959
7           (20724)         (20719)            0.070022            0.126915  0.059081    0.843750   6.648168  0.050194    5.587746
164         (20719)   (POST, 20724)            0.126915            0.065646  0.054705    0.431034   6.566092  0.046373    1.642199
161   (POST, 20724)         (20719)            0.065646            0.126915  0.054705    0.833333   6.566092  0.046373    5.238512


"""

##########################################################################################################
# ASSIGNMENT 3: What are the names of the products whose IDs are given?
# Product ID of User 1: 21987
# Product ID of User 2: 23235
# Product ID of User 1: 22747
##########################################################################################################

product_id_1 = "21987"
check_id(df_grm, product_id_1)   # ['PACK OF 6 SKULL PAPER CUPS']

product_id_2 = "23235"
check_id(df_grm, product_id_2)   # ['STORAGE TIN VINTAGE LEAF']

product_id_3 = "22747"
check_id(df_grm, product_id_3)   # ["POPPY'S PLAYHOUSE BATHROOM"]

##########################################################################################################
# ASSIGNMENT 4: Make product recommendations for the users.
# ASSIGNMENT 5: What are the names of the recommended products?
##########################################################################################################

def arl_recommender(rules_df, product_id, rec_count=1):
    """

    Function to make recommendations according to Association Rule Learning for a given rules dataframe,
    product ID and number of recommendations.

    Parameters
    ----------
    rules_df: dataframe
        Association rules generated with create_rules function.
    product_id: str
        ID of the product.
    rec_count: int
        Number of recommendations.

    Returns
    -------
    recommendation_list: list
        Recommendation list generated through association rule learning.

    """

    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]


##########################################################################################################
# RECOMMENDATIONS FOR USER 1
##########################################################################################################

check_id(df_grm, "21987")  # ['PACK OF 6 SKULL PAPER CUPS']

# Recommending one product
arl_recommender(rules, "21987", 1)  # ['22423']
check_id(df_grm, "22423")  # ['REGENCY CAKESTAND 3 TIER']

# Recommending two products
arl_recommender(rules, "21987", 2)  # ['22423', '21929']
check_id(df_grm, "21929")  # ['JUMBO BAG PINK VINTAGE PAISLEY']

##########################################################################################################
# RECOMMENDATIONS FOR USER 2
##########################################################################################################

check_id(df_grm, "23235")  # ['STORAGE TIN VINTAGE LEAF']

# Recommending one product
arl_recommender(rules, "23235", 1)  # ['22355']
check_id(df_grm, "22356")  # ['CHARLOTTE BAG PINK POLKADOT']

# Recommending two product
arl_recommender(rules, "23235", 2)  # ['22356', '22423']
check_id(df_grm, "22423")  # ['REGENCY CAKESTAND 3 TIER']

##########################################################################################################
# RECOMMENDATIONS FOR USER 3
##########################################################################################################

check_id(df_grm, "22747")  # ["POPPY'S PLAYHOUSE BATHROOM"]

# Recommending one product
arl_recommender(rules, "22747", 1)  # ['22303']
check_id(df_grm, "22303")  # ['COFFEE MUG APPLES DESIGN']

# Recommending two product
arl_recommender(rules, "22747", 2)  # ['22303', '22352']
check_id(df_grm, "22352")  # ['LUNCH BOX WITH CUTLERY RETROSPOT ']

##########################################################################################################
