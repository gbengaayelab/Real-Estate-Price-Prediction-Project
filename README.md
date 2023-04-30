README

# Housing Price Prediction Project

This project aims to predict the prices of houses using Decision Tree Classifier.

## Prerequisites

This project requires the following software:

- Python 3.x
- Jupyter Notebook

This project also requires the following Python libraries:

- pandas
- scikit-learn
- matplotlib

## Installation

1. Clone this repository to your local machine using `https://github.com/gbengaayelab/Real-Estate-Price-Prediction-Project.git`
2. Install the required Python libraries by running the following command: `pip install pandas scikit-learn matplotlib`


Sure, here's the updated metadata section with the additional notes:

## Metadata

The dataset used in this project contains information about various houses including their prices and features. The dataset has a total of 545 entries with 13 columns. The columns are as follows:

- **price:** Price of the house (in millions)
- **area:** Area of the house (in square feet)
- **bedrooms:** Number of bedrooms
- **bathrooms:** Number of bathrooms
- **stories:** Number of stories
- **mainroad:** Type of road leading to the house (yes/no)
- **guestroom:** Presence of guest room (yes/no)
- **basement:** Presence of basement (yes/no)
- **hotwaterheating:** Provision for hot water heating (yes/no)
- **airconditioning:** Presence of air conditioning (yes/no)
- **parking:** Number of parking spots
- **prefarea:** Preferred location (yes/no)
- **furnishingstatus:** Furnishing status of the house (unfurnished = 0, semi-furnished = 1, furnished = 2)

Please note that the `furnishingstatus` column has been encoded as follows: `0` indicates `unfurnished`, `1` indicates `semi-furnished`, and `2` indicates `furnished`. When using this dataset to build a predictive model, it is important to keep this in mind.


## Usage

1. Load the dataset: 
   - The dataset used in this project is the `Housing.csv` file. 
   - The `Housing.csv` file contains information about the houses such as area, number of bedrooms, number of bathrooms, stories, main road availability, guest room availability, basement availability, hot water heating availability, air conditioning availability, parking availability, preference area availability, and furnishing status. 
   - The `price` column is the output column which we are trying to predict. 

2. Split the dataset into input and output datasets:

   ```python
   X = df.drop(columns=['price']) # input dataset
   y = df['price'] # output dataset
   ```

3. Split the dataset into training and testing datasets:

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

4. Train the Decision Tree Classifier model:

   ```python
   model = DecisionTreeClassifier()
   model.fit(X_train, y_train)
   ```

5. Visualize the decision tree using `tree.export_graphviz`:

   ```python
   tree.export_graphviz(model, out_file='housing_prediction.dot',
                        feature_names=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'], 
                        class_names= sorted(y.astype(str).unique()), 
                        label = 'all',
                        rounded = True,
                        filled = True)
   ```

6. Make predictions:

   ```python
   input_data = [ [7420, 6, 4, 3, 1, 3, 0, 0, 1, 2, 1, 2], [3000, 3, 1, 2, 1, 1, 0, 1, 2, 1, 1, 1], [7330, 2, 2, 1, 1, 0, 0, 1, 2, 1, 1, 1] ]
   predictions = model.predict(input_data)
   ```

7. Visualize the predicted prices:

   ```python
   predicted_prices = [prediction for prediction in predictions]
   fig, ax = plt.subplots()
   ax.bar(['Search 1', 'Search 2', 'Search 3'], predicted_prices)
   ax.set_xlabel('Input')
   ax.set_ylabel('Predicted Price in Millions ')
   ax.set_title('Predicted Prices for Three Search Points')
   plt.show()
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.