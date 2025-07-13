---
title: Examining iPhone Listings on eBay in Excel
date: 2025-06-05 14:30:00 -0500
toc: true
toc_sticky: true
categories: [EXCEL , DEMO]
tags: [excel , macros, demo]
comments:
  provider: "utterances"
  utterances:
    theme: "github-dark" # "github-dark"
    issue_term: "pathname"
    label: "comment" # Optional - must be existing label.
---


## Data Utilized

*Dataset Source:* [UltimateAppleSamsungEcom 🍏📱💻 Dataset](https://www.kaggle.com/datasets/kashishparmar02/ultimateapplesamsungecom-dataset) `> iphone_ebay.csv`


**Purpose:** This spreadsheet was created as an exercise to demonstrate potential uses for a variety of Excel formulas. 

>**Remark.** This workbook presents as a tool to be used to make informed decisions when buying an iPhone on eBay, but offers little to no real world utility due to the limitations of the underlying dataset and the intention behind this demonstration. 
{: .notice--warning}

### Workbook Overview 


[Excel File Link](https://www.dropbox.com/scl/fi/1d8skubpagoes6f1c1l39/iphone_ebay_listings_2025.xlsm?rlkey=avoer36sjmd3x3cf9djw1ai8b&st=n3caw974&dl=0){: .btn .btn--primary}


Tab names are in **bold** text. 

* **Listings Overview**  
	* Simple stats: 
		* Price Range 
		* Average Price 
		* Median Price 
		* Number of Listings
	* A "table" that has the columns: 
		* Listing Title 
		* Price 
		* Condition 
		* Variant 

* **Listing Stats**
	* Pivot Tables 
		* `pvtPrices`
			* Pivot Chart: `PriceChart`
		* `pvtListings`
			* Pivot Chart: `CondChart`

* **Master-Table**
	* Listing Title
	* Price
	* Condition
	* Variant 
	* Model (included for the pivot tables)


### Named Ranges 

| **Title** | **Reference**                          |
| --------- | -------------------------------------- |
| Condition | `'Listings Overview'!$C$9`             |
| iPhone13  | `'Listings Overview'!$K$17:$K$22`      |
| iPhone14  | `'Listings Overview'!$L$17:$L$22`      |
| iPhone15  | `Listings Overview'!$M$17:$M$22`       |
| Model     | `'Listings Overview'!$A$4`             |
| Price     | `'Listings Overview'!$B$23:$B$1048576` |
| Variant   | `'Listings Overview'!$B$9`             |


## The Contents of Each Tab


### **Listings Overview Tab** 

![image-center](/assets/images/2025-06-05_Excel-Formula-Refresher/ListOverview_2025-06-05-18.19.24.png){: .align-center}


The **Listings Overview** tab's purpose is to display some details about the listings included in the underlying dataset. 

The user can select the iPhone model using one of the three black icons that appear in the top left. They can also select the iPhone *variant* and *condition* in cells `B9` and `C9` respectively. 

Based on the options selected the price range, average price, median price, and number of listings values will update accordingly.

The three black icons are linked to a hidden drop-down list in cell `A4` by use of macros. When clicked, the a macro will run, changing the values in the drop-down list in cell `B9` (iPhone variant) accordingly.  

>**Example.**
>
> 1. The black 13 icon is clicked
> 2. The selected item in the drop-down list in `A4` becomes 'iPhone13' which references the named range *iPhone13*.
> 3. The choice of variants in the drop down list in cell `B9` then changes accordingly to the variants of iPhone 13s available ('Mini', 'Base' , 'Pro', and 'Pro Max').
>
>This is done with the formula `=INDIRECT($A$4)` in cell `B9` which again, is a reference to the named range that was selected when the black icon was clicked. If the 14 icon were clicked, the choices in cell `B9` would change to 'Base', 'Plus', 'Pro', and 'Pro Max'.
{: .notice--info}

### Listings Stats Tab

![image-center](/assets/images/2025-06-05_Excel-Formula-Refresher/ListStats2025-06-06_11.57.51.png){: .align-center}


This tab contains two pivot tables, `pvtPrices` and `pvtListings`, which are each accompanied by pivot charts `PriceChart` and `CondChart` respectively. 

The pivot table and pivot chart combination are both controlled by three slicers which allow the user to select which iPhone model, variant, and condition they want displayed. 

#### `pvtPrices`Fields 
* Filters
	* Model 
	* Variant 
* Rows
	* Condition
* Values
	* Minimum Price
	* Average Price
	* Maximum Price
	* Std Dev of Price

#### `pvtListings`Fields 

* Filters
	* Model 
	* Variant 
* Rows
	* Condition
* Values 
	* % of Listings 

> **Remark.**   A [custom list](https://support.microsoft.com/en-us/office/sort-data-using-a-custom-list-cba3d67a-c5cb-406f-9b14-a02205834d72) is used to sort the conditions. The "new" conditions ("Brand New", "Open Box") are first in ascending order, i.e. they take on the lower values. 
{: .notice--info}


#### `PricesChart`

This pivot chart is a cluster column chart which is linked to the pivot table `pvtPrices`. For each condition of phone available, it displays 3 columns on the chart for the minimum price, average price,  maximum price, and standard deviation. 
#### `CondChart`

This pivot chart is a pie chart which is linked to the pivot table `pvtListings`. Each section of the piechart corresponds to the available iPhone listing conditions. The "Condition" slicer can be used to filter which conditions appear on either  chart. 

*Reminder:* All three slicers are linked to both pivot charts. 




## Formula Explanations 

### Preliminary definitions 

[Named ranges](https://support.microsoft.com/en-us/office/create-a-named-range-in-excel-adee78ff-bcf0-4283-8c29-83304ca0c29d) are used in this spreadsheet to make formulas more readable and to create a dynamic list in cell `B9` in the **[Listings Overview tab](#listings-overview-tab)** . 

Named ranges referenced in this section : 


| **Title** | **Reference**                          |
| --------- | -------------------------------------- |
| Condition | `'Listings Overview'!$C$9`             |
| iPhone13  | `'Listings Overview'!$K$17:$K$22`      |
| iPhone14  | `'Listings Overview'!$L$17:$L$22`      |
| iPhone15  | `Listings Overview'!$M$17:$M$22`       |
| Model     | `'Listings Overview'!$A$4`             |
| Price     | `'Listings Overview'!$B$23:$B$1048576` |
| Variant   | `'Listings Overview'!$B$9`             |

**Price** - 
This contains the contents of column B in the **[Listings Overview tab](#listings-overview-tab)**

**Model** - (list in the hidden cell `A4`)

The three iPhone models included in this spreadsheet 
* "iPhone13"
* "iPhone14"
* "iPhone15"


**Variant** - (data validation list in cell `B9`) * Changes based on which model is selected in cell `A4`
*Content:*

>   **IF** `A4` = "iPhone13" , **THEN** `B9` = {"Mini" ,"Base" , "Pro" , "Pro Max"} 
    
>   **IF** `A4` = ("iPhone14" **OR** "iPhone15") , **THEN** `B9` =  {"Base" , "Plus", "Pro" , "Pro Max"} 
				

**Condition** (data validation list in cell `C9`)
Indicates what condition the iPhone is in, e.g. used or brand new. These are predefined by eBay. 
*Content:* 
"Brand New" , "New (Other)" , "Open Box",  "Excellent - Refurbished", "Very Good - Refurbished", "Good - Refurbished" , "Parts Only" , "Pre-Owned" 



### Listings Overview Tab

#### Drop-down lists and buttons 



>**Example.** When you click the button 13 to select the iPhone 13 model. A macro that is linked to that button selects the "iPhone13" option in the (hidden) drop-down list in `A4`
>
>*  In cell `B9` there is a visible drop down list that contains the iPhone variant
>
>* The formula that is responsible for this drop-down list is `=INDIRECT($A$4)`
>
>* This drop-down list is dynamic and changes based on which iPhone model is selected in cell `A4` 
{: .notice--info}


#### The stats 

Below are the formulas that populate the section with basic statistics in this tab. 

* Price Range:  `="$"&MIN(Price)&" to $"&MAX(Price)`
	* The `&` operator concatenates the values returned by `MIN` and `MAX`
* Average Price: `=AVERAGE(Price)`
* Median Price: `=MEDIAN(Price)`
* Number of listings: `=COUNTA(Price)`

#### The overview "table"

```vb
=IFS(Condition="Any", FILTER(MasterTbl[[Listing Title]:[Variant]], (SUBSTITUTE(MasterTbl[Model]," ","")=Model) * (MasterTbl[Variant]=Variant)), Condition<>"Any", FILTER(MasterTbl[[Listing Title]:[Variant]], (SUBSTITUTE(MasterTbl[Model]," ","")=Model) * (MasterTbl[Condition]=Condition) * (MasterTbl[Variant]=Variant))
)
```


**Formula explanation:** If *"Any"* is selected as the iPhone condition in cell `C9` then the table filters listings by just the iPhone model and variant; regardless of the condition of the phone. 

If any other iPhone condition is selected in `C9`, then the table filters by both iPhone model, variant, and condition.

Functions used: [IFS](https://support.microsoft.com/en-us/office/ifs-function-36329a26-37b2-467c-972b-4a39bd951d45),  [FILTER](https://support.microsoft.com/en-au/office/filter-function-f4f7cb66-82eb-4767-8f7c-4877ad80c759) ,  & [SUBSTITUTE](https://support.microsoft.com/en-au/office/substitute-function-6434944e-a904-4336-a9b0-1e58df3bc332) 

*`IFS` formula syntax:*
```vb
IFS(logical_test1, value_if_true1, [logical_test2, value_if_true2], [logical_test3, value_if_true3],…)
```



**Formula breakdown:** 

1. *Argument 1 `(logical_test1)`*: If "Any" is selected from the phone condition drop-down list in cell `C9`, execute Argument 2. 

2. *Argument 2* `(value_if_true1)` 
	- `FILTER`: Will return the listing name, price, condition, and variant from the table `MaterTbl` (that resides in the **Master-Table** tab) under the conditions that the iPhone model matches the one selected with the black icons and that the variant matches the one selected in cell `B9`.
		* The `SUBSTITUTE` function removes all spaces from the given iPhone model in column E so that it can match the formatting of the `Model` named range.
		* The `*` operator functions as the logical operator/function `AND` within the criteria argument in the `FILTER` function. 
			* i.e. the row corresponding to a listing from `MasterTbl` is returned if and only if it has the desired model and variant. 

3. *Argument 3* `(logical_test2)`: If the condition selected in cell `C9` is not *"Any“*, then execute argument 4.

4. *Argument 4* `(value_if_true2)`: If a listing from the table `MasterTbl` is of the condition that matches the one selected in cell `C9` and has the variant selected in cell `B9`, that listing is returned into the overview table. 
	* Again, the `*` operator functions as the logical operator/function `AND` within the criteria argument in the `FILTER` function.
		* The row corresponding to a listing is returned if and only if it has the selected model, variant, and condition.

### Master-Table tab 

This tab was created to house the table that would act as the data source for both the **[Listings Overview tab](#listings-overview-tab)** and **[Listings Stats](#listings-stats-tab)** tabs. 

It differs from the data in the `iphone_ebay.csv` file in that I added two columns to make the dataset more useful for my project. The columns were *Variant (Column D)* and *Model (Column E)*. In the following subsections I go over the formulas that populate the cells in these two columns. I also go over the formulas I used to deal with eBay listings that were selling multiple iPhones. 

#### The Price Column (B)

Some of the prices in the dataset were simply listed as "price ranges" in the sense that they would be in  the following format : `$200to$500`. 

 In reality the listings represented that had these ranges were selling more than one storage configuration of a given iPhone model. 
	 E.g. if it is a listing for an iPhone 13 Pro Max, the listing may be selling iPhone 13 Pro Maxs at 128gb , 256gb, and 512gb of internal storage.

 In that case, the minimum value of our "price range" would represent the price of the 128gb model and the maximum value of our price range would represent the price of the 512gb model. With the price of the 256gb model being unknown to us, falling somewhere between the minimum value and the maximum value. 

To deal with these listings I simply took the average of the minimum and maximum value of the price range for simplicity. This unfortunately makes our data a little less reliable. 

The following formula was used to return the values in the price column. The values returned were *pasted as values* into the price column. 

**Averaging of Price Ranges** 


```vb
=IFS(ISTEXT(E2),AVERAGE(NUMBERVALUE(LEFT(E2,7)),NUMBERVALUE(RIGHT(E2,7))),ISNUMBER(E2),E2)
```



**Formula explanation:** If the price is a price range (i.e. a string of text of the form `$Xto$Y`), it takes the average of the maximum value and minimum value. If it is simply a single numeric price it just returns that price. 

Functions used: [IFS](https://support.microsoft.com/en-us/office/ifs-function-36329a26-37b2-467c-972b-4a39bd951d45), [ISTEXT](https://support.microsoft.com/en-au/office/is-functions-0f2d7971-6019-40a0-a171-f2d869135665), [AVERAGE](https://support.microsoft.com/en-us/office/average-function-047bac88-d466-426c-a32b-8f33eb960cf6), [NUMBERVALUE](https://support.microsoft.com/en-us/office/numbervalue-function-1b05c8cf-2bfa-4437-af70-596c7ea7d879), [LEFT](https://support.microsoft.com/en-us/office/left-function-d5897bf6-91f5-4bf8-853a-b63d7de09681), [RIGHT](https://support.microsoft.com/en-us/office/right-function-112699e2-fe7e-4cbf-8d1e-0e8db3884f26), and [ISNUMBER](https://support.microsoft.com/en-us/office/is-functions-0f2d7971-6019-40a0-a171-f2d869135665) 

Cell `E2` contains the price for the listing given by our original dataset. 

**Formula breakdown:** 
1. *Argument1* `ISTEXT(E2)`: If the price in cell E2 is a string of text (i.e. a price range), then `ISTEXT` will return TRUE 

2. *Argument2* `AVERAGE`: If Argument1 returns TRUE then we take the average of the minimum value and maximum value of the "price range" 
	1. Both the functions `LEFT` and `RIGHT` return 7 characters because an iPhone is going to be at most somewhere in the thousands of dollars.  1000.00 has 7 characters, including the decimal point. 
	2. `NUMBERVALUE` converts the strings extracted by the `LEFT` and `RIGHT` functions into the number format so that `AVERAGE` can use them in it's calculation. 
3. *Argument3* `ISNUMBER(E2)`, if our price is formatted as a number in Excel. I.e. it's just a number and not a price range represented by a string of text,  this argument returns the value `TRUE`.
4. *Argument4*: If Argument3 is `TRUE`, then `IFS` simply returns the value in cell `E2`. 

>If we had a price that was of the form *$X* but formatted as a number in excel then the 2nd portion of the IFS function would take the average of that single price, which would be equal to the price (i.e. there would be no change in value).
{: .notice--info}

#### Variant (Column D)

This column's purpose is to discern between iPhone variants of a particular model . The formula populating this column is slightly modified based on the iPhone model. 

To do this, I filtered the table by model using column E and then the following formulas were used in column D.  


**iPhone13** 

 ```vb
 =IFNA(IFS(ISNUMBER(FIND("Pro Max",$A2)),"Pro Max", AND(ISNUMBER(FIND("Pro",$A2)),ISNUMBER(FIND("Pro Max",$A2))=FALSE),"Pro",ISNUMBER(FIND("Mini",$A2)),"Mini"),"Base")
```

**iPhone14 & iPhone 15**

```vb
=IFNA(IFS(ISNUMBER(FIND("Pro Max",A227)),"Pro Max", AND(ISNUMBER(FIND("Pro",A227)),ISNUMBER(FIND("Pro Max",A227))=FALSE),"Pro",ISNUMBER(FIND("Plus",A227)),"Plus"),"Base")
```

Below I provide an explanation for the iPhone13 version of this formula. 

```vb
=IFNA(IFS(ISNUMBER(FIND("Pro Max",$A2)),"Pro Max", AND(ISNUMBER(FIND("Pro",$A2)),ISNUMBER(FIND("Pro Max",$A2))=FALSE),"Pro",ISNUMBER(FIND("Mini",$A2)),"Mini"),"Base")
```

**Formula explanation:** The basic idea is if any of the strings associated with the variants of given iPhone model are found in the listing title then the formula will return that string as the variant. 
	e.g. If "Pro Max" is found in the listing title the formula will return "Pro Max" as the variant assuming the listing is for an iPhone 13 Pro Max 
Functions used: [IFNA](https://support.microsoft.com/en-us/office/ifna-function-6626c961-a569-42fc-a49d-79b4951fd461) ,  [IFS](https://support.microsoft.com/en-us/office/ifs-function-36329a26-37b2-467c-972b-4a39bd951d45) , [ISNUMBER](https://support.microsoft.com/en-us/office/is-functions-0f2d7971-6019-40a0-a171-f2d869135665) , [FIND](https://support.microsoft.com/en-us/office/find-function-06213f91-b5be-4544-8b0b-2fd5a775436f)

`IFNA(value, value_if_na_error)`

`IFS(condition1, value_if_true1, [logical_test2, value_if_true2], [logical_test3, value_if_true3],…)`

**Formula breakdown:** 

1. *Formula 1:*  `IFNA( ... , "Base")` will return "Base" as the variant if none of the other strings ("Mini", "Pro", "Pro Max") are found in the listing name
2. *Formula 2* `IFS`:
	1. *Argument1* `(logical_test1)`: If `FIND` finds the string "Pro Max" in the listing name (cell `A2`) then it returns a number marking the position of the *P* in "Pro Max". In this case, `ISNUMBER` will return `TRUE` as the position of *P* will be always be a number. 
	2. *Argument2* `(value_if_true1)`: If  the test is true. IFS returns "Pro Max" as the iPhone variant 
	3. *Argument3* `(logical_test2)`: If the string "Pro" is found in the listing name and it's not a substring of "Pro Max" then `AND` will return true 
	4. *Argument4* `(value_if_true2)`: If Condition2 is true, then `IFS` will return "Pro" as the iPhone variant
	5. *Argument5* `(logical_test3)`: If `FIND` finds the string "Mini" in the listing name (cell `A2`) then it returns a number marking the position of the *M* in "Mini". In this case, `ISNUMBER` will return `TRUE` as the position of *M* will  always be a number. 
	6. *Argument6* `(value_if_true3)`: If Condition3 is true, then `IFS` will return "Mini" as the iPhone variant. 

#### Model (Column E) 

Formula used in Column E (Model) : `=TRIM(MID(A2,FIND("iPhone",A2),9))`

**Formula explanation:** The formula looks for the starting point of the word "iPhone" and then prints nine characters from the starting point,. Nine characters because "iPhone 13" is nine characters long including the space. `TRIM` is used to remove any trailing spaces. 

>**Example.** `A2` = 'Apple iPhone 13  128GB / 256GB - Verizon - Locked'
The *i* in 'iPhone' in cell `A2` is the 7th character in the cell
So the formula starts at position 7 and then prints nine characters starting from that position. This returns "iPhone 13" in this case. 
{: .notice--info}	
Functions used: [TRIM](https://support.microsoft.com/en-us/office/trim-function-410388fa-c5df-49c6-b16c-9e5630b479f9) , [MID](https://support.microsoft.com/en-au/office/mid-function-2eba57be-0c05-4bdc-bf81-5ecf4421eB9a) , and [FIND](https://support.microsoft.com/en-us/office/find-function-06213f91-b5be-4544-8b0b-2fd5a775436f)
{: .notice--info}
**Formula breakdown:**

`TRIM`
To remove any trailing spaces from the text returned from `MID` and `FIND` 

`MID`
Returns nine characters from cell `A2`, starting from the 'i' in 'iPhone'

`FIND`
Locates the starting point of 'iPhone' in the cell `A2`



>**Remark.** In part two of looking at this data set, I will explore price prediction models, using an updated version of this dataset. 
{: .notice--info}


