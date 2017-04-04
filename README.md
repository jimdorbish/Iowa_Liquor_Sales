# Iowa_Liquor_Sales
For the sale and distribution of goods, the most difficult part is making sure the supply chain mirrors the consumer demand. Arriving to a store with empty shelves or owning a store that has product on the shelf too long are products of an innaccurate supply chain model. This is a great real world situation to allow machine learning to do the heavy lifting and creating an accurate sales projection model. Using the first quarter of sales data for 2016 we can predict the sales for the full 2016 sales giving us a much more accurate model than just using the total sales for 2015. 


The State of Iowa has released all state owned liquor stores’ pertinent sales data in the form of a reduced .CSV spreadsheet, which can be found here.  This information included local information like: zip code, county name/number, store number, and date. The sales information included is: product name, cost of bottle for the state, cost of bottle retail price, bottles sold, and liters sold. 


There are categories of superfluous information, redundant data, and messy data types in the .CSV that underwent transformation. After loading the .CSV into a dataframe with pandas, unnecessary columns and dollar signs were removed, dates were converted to date-time format, null values were dropped, and any object typed integers were transformed into int type integers. Sub group tables and derived units were also made. Firstly stores that opened and/or closed during 2015-2016 were dropped. Also store number organized tables that sorted 2015/2016 first quarter sales and 2015 as a whole. Derived units that were created were “Total Sales” ,“Margin”, and “Price per Liter”. Total sales is the sum of all product sales per store over the allotted date time period.


Looking at total sales and dates, the liquor sold in the first quarter is proportional to full year sales. The models that could best predict these linear relationships are a linear regression model or an elastic net model. The linear regression model with yield a more direct interpretation of the sales variables, whereas with an elastic net model the cost penalties of an overly complex model can be varied as well as the alpha values to see what yields a higher score. Also with the adjustable parameters available with elastic net, the data can be cross-validated with a 5 fold cross. First the models were trained on the 2015 1st quarter totals sales to full 2015 total sales. The elastic net function was passed penalty values .5, 1, and 5. 


The results of the two models trained on the 2015  dataset and tested on the 2016 dataset show a model score of 98.40% for linear regression and 98.39% model fit for elastic net. Using the elastic net model 2016 total sales is predicted to be $278,358,955.68. These results were broken down per store for margin growth and sales growth for each year. 


![alt text](https://github.com/jayghez/Iowa_Liquor_Sales/blob/master/final_results_Iowa.png)
