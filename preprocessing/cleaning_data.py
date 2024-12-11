# code that will be used to preprocess the data you 
# will receive to predict a new price. (fill the NaN values, handle text data, etc...).

#preprocess() that will take a new house's data as input and return those data preprocessed as output.
#If your data doesn't contain the required information, you should return an error to the user.

# Imports

import numpy as np
import pandas as pd
from gower import gower_matrix
import pickle

# Code

class Preprocessor:
    """
    Class to handle loading data, Gower distance calculation, prepping postal/tax data to allow prediction using the model.
    """
    def __init__(self) -> None:
        """
        Create an instance of class Preprocessor.

        : param new_data: list: Data from user input stored in dict.
        """
        self.data = None
        self.new_data = None
        self.distance = None
        self.postalcodes = None
        self.taxdata = None
        self.property_type = None
        self.building_state = None
        self.num_facades = None
        self.num_rooms=  None
        self.training_indices = None

    def load_data(self) -> None:
        """
        Load the data in a pd.DataFrame, clean data and store in attribute. Preparation for distance calculation.
        """
        # Load property data and clean

        data = pd.read_csv('./preprocessing/Clean_data.csv')

        data.drop(columns=["Unnamed: 0", "id"], inplace=True)
        data["Locality"] = data["Locality"].astype("str")
        data["Fully equipped kitchen"] = data["Fully equipped kitchen"].astype("str")
        data["Fireplace"] = data["Fireplace"].astype("str")
        data["Terrace"] = data["Terrace"].astype("str")
        data["Garden"] = data["Garden"].astype("str")
        data["Swimming pool"] = data["Swimming pool"].astype("str")
        data["Furnished"] = data["Furnished"].astype("str")
        data["Number of rooms"] = data["Number of rooms"].astype("int64")
        data["Number of facades"] = data["Number of facades"].astype("int64")

        # Load INS data and tax data to link
        data_insee = pd.read_csv("./preprocessing/INSEE_PostCode.csv", encoding="latin-1")
        subset_columns = data_insee.columns[6:]
        data_insee["PostalCodes"] = data_insee[subset_columns].apply(
            lambda row: row.dropna().tolist(), axis=1
        )
        data_insee.drop(columns=data_insee.columns[6:22], inplace=True)

        # Import and prepare dataset from SPF Finances to join on Code INS
        data_fin = pd.read_csv("./preprocessing/SPF_FIN_Stat.csv", encoding="latin-1")

        # Merge two datasets on Code INS, keeping only records that are present in both tables
        # Rows without postal code, or without financial data are not interesting for final join with data
        data_fin_postcode = pd.merge(data_fin, data_insee, how="inner", on="Code INS")

        # Unpack/explode lists of post codes to obtain dataset with one row of info per postal code
        data_fin_postcode_exploded = data_fin_postcode.explode("PostalCodes")

        # Convert post codes to str type and join data to original dataset
        data_fin_postcode_exploded["PostalCodes"] = (
            data_fin_postcode_exploded["PostalCodes"].astype("int").astype("str")
        )

        # Set postcode - tax data lookup table for input data

        self.taxdata = data_fin_postcode_exploded[['PostalCodes', 'Revenu moyen par déclaration']]
        self.taxdata.rename(
            columns={"Revenu moyen par déclaration": "Mean_income_taxunit"},
            inplace=True,
        )

        # Get list of available post codes

        self.postalcodes = self.taxdata['PostalCodes'].to_list()

        # Continue prepping data for distance calculation

        data_postcodes = pd.merge(
            data_fin_postcode_exploded,
            data,
            how="inner",
            left_on="PostalCodes",
            right_on="Locality",
        )

        # Drop duplicate columns, rename columns
        data_postcodes.drop(columns="Entités administratives_x", inplace=True)
        data_postcodes.drop(columns="Locality", inplace=True)
        data_postcodes.drop(columns="Code INS", inplace=True)
        data_postcodes.rename(
            columns={"Entités administratives_y": "Locality"}, inplace=True
        )
        data_postcodes.rename(
            columns={"Nombre d'habitants": "N_Inhabitants"}, inplace=True
        )
        data_postcodes.rename(
            columns={"Revenu total net imposable": "Tot_taxable_income"}, inplace=True
        )
        data_postcodes.rename(
            columns={"Revenu moyen par déclaration": "Mean_income_taxunit"},
            inplace=True,
        )
        data_postcodes.rename(
            columns={"Revenu médian par déclaration": "Median_income_taxunit"},
            inplace=True,
        )
        data_postcodes.rename(
            columns={"Revenu moyen par habitant": "Mean_income_inhabitant"},
            inplace=True,
        )
        data_postcodes.rename(
            columns={"Indice de richesse": "Wealth_index"}, inplace=True
        )
        data_postcodes.rename(columns={"Impôt d'Etat": "State_tax"}, inplace=True)
        data_postcodes.rename(
            columns={"Taxes communales et d'agglomération": "Local_tax"}, inplace=True
        )
        data_postcodes.rename(columns={"Impôt total": "Tot_tax"}, inplace=True)
        data_postcodes.rename(columns={"Langue": "Language"}, inplace=True)
        data_postcodes.rename(columns={"Région": "Region"}, inplace=True)
        data_postcodes.rename(columns={"Arrondissement": "District"}, inplace=True)

        # clean data types
        columns_to_convert = [
            "N_Inhabitants",
            "Tot_taxable_income",
            "State_tax",
            "Local_tax",
            "Tot_tax",
        ]
        data_postcodes[columns_to_convert] = (
            data_postcodes[columns_to_convert]
            .apply(lambda col: col.str.replace(".", "", regex=False))
            .astype(float)
        )
        data_postcodes["N_Inhabitants"] = data_postcodes["N_Inhabitants"].astype(int)
        data_postcodes["Wealth_index"] = data_postcodes["Wealth_index"].astype(float)

        # Subset data to set price range and store in attribute

        subset_price_datapostcodes = data_postcodes[(data_postcodes['Price'] >= 200000) & (data_postcodes['Price'] <= 600000)]
        subset_columns_datapostcodes = subset_price_datapostcodes[[
        "Mean_income_taxunit",
        "Subtype of property",
        "State of the building",
        "Surface area of the plot of land",
        "Number of rooms",
        "Living Area",
        "Number of facades",
    ]]

        self.data = subset_columns_datapostcodes

        # Set lists for selectors

        self.property_type = list(data_postcodes['Subtype of property'].unique())
        self.building_state = list(data_postcodes['State of the building'].unique())
        self.num_facades = list(data_postcodes['Number of facades'].sort_values().unique().astype('int32'))
        self.num_rooms = list(data_postcodes['Number of rooms'].sort_values().unique().astype('int32'))

    def preprocess(self, new_data:dict) -> np.ndarray:
        """
        Prepare an np.ndarray of Gower distances to training dataset used in the model
        """

        self.new_data = pd.DataFrame(new_data)

        # obtain the tax information for given postal code ; drop postal code

        self.new_data['PostalCodes'] = self.new_data['PostalCodes'].astype('str')
        self.new_data['Mean_income_tax_unit'] = self.taxdata[self.taxdata['PostalCodes'] == self.new_data['PostalCodes'][0]]['Mean_income_taxunit']
        self.new_data.drop(columns= ['PostalCodes'], inplace= True)

        # Ensure datatypes

        self.new_data['Subtype of property'] = self.new_data['Subtype of property'].astype('str')
        self.new_data['State of the building'] = self.new_data['State of the building'].astype('str')
        self.new_data['Number of facades'] = self.new_data['Number of facades'].astype('int64')
        self.new_data['Number of rooms'] = self.new_data['Number of rooms'].astype('int64')
        self.new_data['Surface area of the plot of land'] = self.new_data['Surface area of the plot of land'].astype('float')
        self.new_data['Living Area'] = self.new_data['Living Area'].astype('float')

        # Call function to calc Gower distance

        gower_dist = self.gower_calc()
        self.distance = gower_dist
        return gower_dist


    def gower_calc(self) -> np.ndarray:
        """
        Calculate Gower distance of new datapoint to points used in training set of the model to allow prediction.

        : returns: np.ndarray: A np.ndarray of Gower distances permitting to predict property price
        """

        # Load array of indices and select rows of training data from original dataset

        self.training_indices = pickle.load(open('./preprocessing/training_indices.pkl', 'rb'))
        selected_rows = self.data.iloc[self.training_indices]

        # Add row of new data to dataframe

        data_to_dist = pd.concat([selected_rows, self.new_data], ignore_index=True).values

        # Calculate Gower distance and retrieve distances for the new datapoint

        gower_mat = gower_matrix(data_to_dist)
        return gower_mat[-1]

