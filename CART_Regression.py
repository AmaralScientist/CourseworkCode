##
 #  This program uses the classification and regression tree (CART) algorithm to build
 #  a decision tree for regression problems. Input files are pre-processed data sets 
 #  that must be formatted such that the target variable for prediction is labelled
 #  "Response" and located in the last column of the data frame. 
 #  Ten percent of a data set is removed as a validation set and the same validation set 
 #  is used for all experiments. Five-fold cross-validation is performed and a tree
 #  is tuned to determine the optimal number of data points remaining in a bin when
 #  branching is stopped. Summary statistics are output to a file. The data set is again 
 #  split for five-fold cross validation and a regression tree with no cutoff for 
 #  the number of data points remaining in a bin is built as well as a regression tree
 #  using the optimized value of the tuned hyperparameter; the performances of both
 #  are compared. The summary statistics are output to a separate file.
 #  @author Michelle Amaral
 #  @version 1.0
##

import sys,re,os 
# Load pandas
import pandas as pd
# Load numpy
import numpy as np
import random
from collections import Counter
import math
from operator import itemgetter
sys.setrecursionlimit(10**6) 


# Receive the name of input file from the command line
input_file_name = sys.argv[1]       

# This method accesses a data frame and splits its row numbers into k 
# lists of row numbers, each corresponding to k unique test folds
# for use with k fold cross-validation
def split_for_cross_validation( data_frame, number_of_folds ):
    total_number_of_instances = data_frame.shape[0]
    
    # Calculate the number of data instances per fold 
    number_instances_per_fold = math.floor( total_number_of_instances / number_of_folds )

    # Create empty list to hold k number of test data sets
    list_of_data_sets = []
    
    # Create empty list to hold row numbers that are available for selection 
    eligible_row_numbers = []    
    # Append the row numbers to the list, then randomly shuffle
    for k in range(0, total_number_of_instances):
        eligible_row_numbers.append ( k )
    random.shuffle( eligible_row_numbers )
        
    # Create empty list to hold the final row numbers for each test fold
    list_of_row_numbers_for_test_folds = []
    
    for k_index in range(0, number_of_folds):
        # Empty list to store the row numbers for current fold
        temp_row_numbers_for_this_fold = []
        # Initialize counter variable
        counter = 0                
        while counter < number_instances_per_fold:
            # Randomly select the row number of an eligible instance 
            index_of_row_number_to_grab = random.randrange(len(eligible_row_numbers))
            # Access the actual row number from original data frame
            row_number_to_grab = eligible_row_numbers[index_of_row_number_to_grab]
            # Append this row number to list
            temp_row_numbers_for_this_fold.append(row_number_to_grab)
            # Remove this row number from list of eligible row numbers
            eligible_row_numbers.pop(index_of_row_number_to_grab)
            # Increment counter variable by 1
            counter = counter + 1
        # Append all row numbers to final list
        list_of_row_numbers_for_test_folds.append( temp_row_numbers_for_this_fold )

    return( list_of_row_numbers_for_test_folds )


# This method sorts the values of an attribute and bins the
# values into 10 equal-sized bins. Returns a list of the bin values
def determine_bins( column_as_list, response_as_list ):  
    sorted_column_as_list, sorted_response_as_list = 
      (list(t) for t in zip(*sorted(zip(column_as_list, 
        response_as_list))))
    binned_attribute_values, bin_edges = 
      pd.qcut(sorted_column_as_list, q = 10, retbins=True, duplicates='drop')
    return( bin_edges )
    

# This method determines the best split for the values of an
# attribute, defined as the split that returns the lowest mean 
# squared error. The attribute values are split into 10
# equal-sized bins, with each bin edge being evaluated for 
# mean squared error. 
def calculate_MSE_for_attribute( df, column_name ):
    # Determine index of the Response column
    column_names = df.columns.values.tolist()
    for column_name_index in range( 0, len(column_names) ):
        if column_names[column_name_index] == "Response":
            response_column_index = column_name_index

    # Convert the Response column to a list
    response_as_list = df.iloc[:,response_column_index].tolist()
        
    # Convert the attribute column to a list
    column_index = df.columns.get_loc( str(column_name) )
    column_as_list = df.iloc[:,column_index].tolist()

    # Calculate bins of equal width
    bin_edges = determine_bins( column_as_list, response_as_list )
    # Remove first and last bin values since they will be 
    # open to all values less than or greater than
    utilizable_bins = []
    for bin_edge_index in range( 1, (len(bin_edges) - 1) ):
        utilizable_bins.append(bin_edges[bin_edge_index])
    # If there are no more possible bins, return to CART_algorithm
    if len(utilizable_bins) == 0:
        return( 1, 1, 1, 1 )

    # Empty list to store the MSE for each possible split
    MSE_for_all_bins = []
    average_response_value_before_split_for_all_bins = []
    average_response_value_after_split_for_all_bins = []

    # Iterate over each bin edge value, grouping the instances above edge
    # and less than or equal to edge
    for bin_edge_index in range( 0, len(utilizable_bins) ):
        # Iterate over each attribute value, splitting into those attribute 
        # values above split and less than or equal to split
        response_value_before_split = []
        response_value_after_split = []
        for attribute_value_index in range( 0, len(column_as_list) ):
            if column_as_list[attribute_value_index] <= utilizable_bins[bin_edge_index]:
                response_value_before_split.append(response_as_list[attribute_value_index])
            elif column_as_list[attribute_value_index] > utilizable_bins[bin_edge_index]:
                response_value_after_split.append(response_as_list[attribute_value_index])

        # Empty list to hold each squared error
        MSE_values = []

        # Calculate the mean of the response variables prior or equal to split
        average_response_value_before_split = np.mean( response_value_before_split )
        average_response_value_before_split_for_all_bins.append( average_response_value_before_split )
        for response_value_before_split_index in range( 0, len(response_value_before_split) ):
            # Subtract each item in response_value_before_split list and square the difference
            mean_squared_difference = 
              ( average_response_value_before_split - response_value_before_split[response_value_before_split_index] ) ** 2
            MSE_values.append( mean_squared_difference )

        # Calculate the mean of the response variables after split       
        average_response_value_after_split = np.mean( response_value_after_split )
        average_response_value_after_split_for_all_bins.append( average_response_value_after_split )
        for response_value_after_split_index in range( 0, len(response_value_after_split) ):
            # Subtract each item in response_value_before_split list and square the difference
            mean_squared_difference = 
              ( average_response_value_after_split - response_value_after_split[response_value_after_split_index] ) ** 2
            MSE_values.append( mean_squared_difference )

        # Calculate the sum of the squared errors
        sum_of_MSE = sum( MSE_values )
        averaged_MSE = sum_of_MSE / len(MSE_values)
        # Append this value to the overall list
        MSE_for_all_bins.append( averaged_MSE ) 

    # Determine the lowest MSE and corresponding split point for this single attribute    
    # Find the position of the lowest MSE

    # Begin by setting the first MSE in the list to the lowest value then 
    # iterate through the remaining values in the list to see if they are lower
    lowest_MSE = MSE_for_all_bins[0]
    index_of_lowest_MSE = 0
    for MSE_for_all_bins_index in range( 1, len(MSE_for_all_bins) ):
        if MSE_for_all_bins[MSE_for_all_bins_index] < lowest_MSE:
            lowest_MSE = MSE_for_all_bins[MSE_for_all_bins_index]
            index_of_lowest_MSE = MSE_for_all_bins_index

    split_point = index_of_lowest_MSE
    # Find the value of that split
    split_point_value = utilizable_bins[split_point]
    average_response_value_before_split = 
      average_response_value_before_split_for_all_bins[split_point]
    average_response_value_after_split = 
      average_response_value_after_split_for_all_bins[split_point]
    
    return( lowest_MSE, split_point_value, average_response_value_before_split, 
      average_response_value_after_split )


# This method determines which attribute has the smallest 
# MSE and returns the column number of that attribute 
def find_attribute_smallest_MSE( MSE_for_all_attributes ):    
    smallest_MSE = MSE_for_all_attributes[0]
    column_number = 0
    for MSE_index in range( 1, len(MSE_for_all_attributes) ):
        if MSE_for_all_attributes[MSE_index] < smallest_MSE:
            smallest_MSE = MSE_for_all_attributes[MSE_index]
            column_number = MSE_index
    return(column_number)


# This method calculates the average value of the response 
# for a given data frame 
def calculate_mean_response( df ):
    # Determine index of the Response column
    column_names = df.columns.values.tolist()
    for column_index in range( 0, len(column_names) ):
        if column_names[column_index] == "Response":
            response_column_index = column_index

    # Convert the Response column to a list
    response_as_list = df.iloc[:,response_column_index].tolist()
    # Calculate the mean response value
    mean_response_value = np.mean( response_as_list )
    return( mean_response_value )


# This method implements the CART algorithm for regression. It does not provide 
# tuning of the number of data points remaining in a bin when branching stops
def CART_algorithm( df, mean_response_for_node, attributes_split_upon, 
  attribute_best_split_point ):    
    # Determine index of the Response column
    column_names = df.columns.values.tolist()
    for column_index in range( 0, len(column_names) ):
        if column_names[column_index] == "Response":
            response_column_index = column_index
	
	# Extract the names of the attributes
    column_names = df_original.columns.values.tolist()
    list_of_attributes = []
    for column_names_index in range(0, (len(column_names)-1)):
        list_of_attributes.append(column_names[column_names_index])
	
	
    df_instances = df.shape[0]

    if df_instances == 0 :
        # return the average of the mean_response_for_node node
        return ( Node( mean_response_for_node ) )

    # If only one instance remains, return tree
    if df_instances == 1:
        # Assign the averaged value
        tree = Node( calculate_mean_response( df ) )
        return ( tree )

    # If the remaining instances are of the same response variable, return tree
    response_as_list = df.iloc[:, response_column_index].tolist()
    values_in_response = list( set( response_as_list ) )      
    if len(values_in_response) == 1:
        # Assign that value
        tree = Node( calculate_mean_response( df ) )
        return ( tree )

    # Determine MSE for attributes
    name_of_the_attribute = []
    MSE_for_all_attributes = []    
    best_location_of_splits = []
    average_response_value_before_split_all_attributes = []
    average_response_value_after_split_all_attributes = []
    
    column_index = 0
    while( column_index < (len(list_of_attributes))-1 ):
        # If the values in the attribute column are the same,
        # remove that attribute from consideration
        if np.var((df.loc[:,list_of_attributes[column_index]].values)) == 0:
            del list_of_attributes[column_index]
        # Call method to determine the mean squared error and the location of the best split
        an_attribute_MSE, location_of_best_split, average_response_value_before_split, 
          average_response_value_after_split = 
            calculate_MSE_for_attribute( df, list_of_attributes[column_index] )
        if an_attribute_MSE == 1 and location_of_best_split == 1 and 
          average_response_value_before_split == 1 and average_response_value_after_split == 1:
            del list_of_attributes[column_index]
        else:            
            name_of_the_attribute.append( list_of_attributes[column_index] )
            MSE_for_all_attributes.append( an_attribute_MSE )
            best_location_of_splits.append( location_of_best_split )
            average_response_value_before_split_all_attributes.append( average_response_value_before_split ) 
            average_response_value_after_split_all_attributes.append( average_response_value_after_split )
            column_index = column_index + 1

    if not MSE_for_all_attributes:
        tree = Node( calculate_mean_response( df ) )
        return ( tree )

    resulting_column_indices = []    
    for names_index in range( 0, len(name_of_the_attribute) ):
        location = df.columns.get_loc( name_of_the_attribute[names_index] )
        resulting_column_indices.append( location )

    # Determine attribute with smallest MSE
    if len( MSE_for_all_attributes ) == 1:
        column_index_smallest_MSE = 0
        column_index_of_smallest_MSE_mapped_to_df = 
          resulting_column_indices[column_index_smallest_MSE]
        best_split_value = best_location_of_splits[0]
        average_response_before_split = 
          average_response_value_before_split_all_attributes[0]
        average_response_after_split = 
          average_response_value_after_split_all_attributes[0]

    else:    
        column_index_smallest_MSE = find_attribute_smallest_MSE( MSE_for_all_attributes )
        column_index_of_smallest_MSE_mapped_to_df = 
          resulting_column_indices[column_index_smallest_MSE]
        best_split_value = best_location_of_splits[column_index_smallest_MSE]
        average_response_before_split = 
          average_response_value_before_split_all_attributes[column_index_smallest_MSE]
        average_response_after_split = 
          average_response_value_after_split_all_attributes[column_index_smallest_MSE]


    name_of_column = str( df.columns[column_index_of_smallest_MSE_mapped_to_df] )
    attributes_split_upon.append( name_of_column )
    attribute_best_split_point.append( best_split_value )
    length_of_attributes_split_upon = len(attributes_split_upon)

    # If algorithm starts to split on the same attribute with the same split value
    # over and over, return the tree
    if attributes_split_upon[(length_of_attributes_split_upon - 3)] == 
      attributes_split_upon[(length_of_attributes_split_upon - 1)] and 
      attribute_best_split_point[(length_of_attributes_split_upon - 3)] == 
      attribute_best_split_point[(length_of_attributes_split_upon - 1)]:                
        tree = Node( calculate_mean_response( df ) )
        return ( tree )

    temp_df_before_split = 
      df.loc[df.iloc[:,column_index_of_smallest_MSE_mapped_to_df] <= best_split_value]
    temp_df_after_split = 
      df.loc[df.iloc[:,column_index_of_smallest_MSE_mapped_to_df] > best_split_value]

    # Assign attribute, attribute values, and the most popular class
    # value from attribute that had the largest gain ratio
    tree = Node( calculate_mean_response( df ) )
    tree.attribute = name_of_column

    split_dfs = [temp_df_before_split, temp_df_after_split]

    # Split the data set by response values greater than and less than or equal to mean            
    for split_dfs_index in range( 0, len(split_dfs) ):
        temp_df = split_dfs[split_dfs_index]
        if temp_df.shape[0] == 0:
            tree = ( Node( mean_response_for_node ) )
            return(tree)
        else:
            mean_response_for_node = calculate_mean_response( temp_df )
            subtree = CART_algorithm ( temp_df, mean_response_for_node, 
              attributes_split_upon, attribute_best_split_point )
            # Store this dataframe in the node
            subtree.instances_labeled = temp_df    
            if split_dfs_index == 0:
                name_of_value = best_split_value
                tree.children["LessThan"] = subtree
            else:
                name_of_value = best_split_value
                tree.children["GreaterThan"] = subtree
    return ( tree )

# This method implements the CART algorithm for regression and takes an extra parameter
# to designate the number of data points remaining in a bin when branching stops
def CART_algorithm_for_prepruning( df, mean_response_for_node, attributes_split_upon, 
  attribute_best_split_point, this_threshold ):
    # Determine index of the Response column
    column_names = df.columns.values.tolist()
    for column_index in range( 0, len(column_names) ):
        if column_names[column_index] == "Response":
            response_column_index = column_index
            
    # Extract the names of the attributes
    column_names = df_original.columns.values.tolist()
    list_of_attributes = []
    for column_names_index in range(0, (len(column_names)-1)):
        list_of_attributes.append(column_names[column_names_index])

    df_instances = df.shape[0]
    if this_threshold == 0:
        if df_instances == 0 :
            # return the average of the mean_response_for_node node
            return ( Node( mean_response_for_node ) )
    elif this_threshold != 0:
        if df_instances == 0 :
            return ( Node( mean_response_for_node ) )        
        elif df_instances <= this_threshold:
            tree = Node( calculate_mean_response( df ) )
            return( tree )

    # If only one instance remains, return tree
    if df_instances == 1:
        # Assign the averaged value
        tree = Node( calculate_mean_response( df ) )
        return ( tree )

    # If the remaining instances are of the same response variable, return tree
    response_as_list = df.iloc[:, response_column_index].tolist()
    values_in_response = list( set( response_as_list ) )      
    if len(values_in_response) == 1:
        # Assign that value
        tree = Node( calculate_mean_response( df ) )
        return ( tree )

    # Determine MSE for attributes
    name_of_the_attribute = []
    MSE_for_all_attributes = []    
    best_location_of_splits = []
    average_response_value_before_split_all_attributes = []
    average_response_value_after_split_all_attributes = []

    column_index = 0
    while( column_index < (len(list_of_attributes))-1 ):
        # If the values in the attribute column are the same, 
        # remove that attribute from consideration
        if np.var((df.loc[:,list_of_attributes[column_index]].values)) == 0:
            del list_of_attributes[column_index]
        # Call method to determine the mean squared error and the location of the best split
        an_attribute_MSE, location_of_best_split, average_response_value_before_split, 
          average_response_value_after_split = 
          calculate_MSE_for_attribute( df, list_of_attributes[column_index] )
        if an_attribute_MSE == 1 and location_of_best_split == 1 and 
          average_response_value_before_split == 1 and 
          average_response_value_after_split == 1:
            del list_of_attributes[column_index]
        else:            
            name_of_the_attribute.append( list_of_attributes[column_index] )
            MSE_for_all_attributes.append( an_attribute_MSE )
            best_location_of_splits.append( location_of_best_split )
            average_response_value_before_split_all_attributes.append( average_response_value_before_split ) 
            average_response_value_after_split_all_attributes.append( average_response_value_after_split )
            column_index = column_index + 1

    if not MSE_for_all_attributes:
        tree = Node( calculate_mean_response( df ) )
        return ( tree )

    resulting_column_indices = []    
    for names_index in range( 0, len(name_of_the_attribute) ):
        location = df.columns.get_loc( name_of_the_attribute[names_index] )
        resulting_column_indices.append( location )

    # Determine attribute with smallest MSE
    if len( MSE_for_all_attributes ) == 1:
        column_index_smallest_MSE = 0
        column_index_of_smallest_MSE_mapped_to_df = 
          resulting_column_indices[column_index_smallest_MSE]
        best_split_value = best_location_of_splits[0]
        average_response_before_split = 
          average_response_value_before_split_all_attributes[0]
        average_response_after_split = 
          average_response_value_after_split_all_attributes[0]

    else:    
        column_index_smallest_MSE = find_attribute_smallest_MSE( MSE_for_all_attributes )
        column_index_of_smallest_MSE_mapped_to_df = 
          resulting_column_indices[column_index_smallest_MSE]
        best_split_value = best_location_of_splits[column_index_smallest_MSE]
        average_response_before_split = 
          average_response_value_before_split_all_attributes[column_index_smallest_MSE]
        average_response_after_split = 
          average_response_value_after_split_all_attributes[column_index_smallest_MSE]

    name_of_column = str( df.columns[column_index_of_smallest_MSE_mapped_to_df] )
    attributes_split_upon.append( name_of_column )
    attribute_best_split_point.append( best_split_value )
    length_of_attributes_split_upon = len(attributes_split_upon)

    # If algorithm starts to split on the same attribute with the same split value
    # over and over, return the tree
    if attributes_split_upon[(length_of_attributes_split_upon - 3)] == 
      attributes_split_upon[(length_of_attributes_split_upon - 1)] and 
      attribute_best_split_point[(length_of_attributes_split_upon - 3)] == 
      attribute_best_split_point[(length_of_attributes_split_upon - 1)]:                
        tree = Node( calculate_mean_response( df ) )
        return ( tree )

    temp_df_before_split = df.loc[df.iloc[:,column_index_of_smallest_MSE_mapped_to_df] <= best_split_value]
    temp_df_after_split = df.loc[df.iloc[:,column_index_of_smallest_MSE_mapped_to_df] > best_split_value]

    # Assign attribute, attribute values, and the most popular class
    # value from attribute that had the largest gain ratio
    tree = Node( calculate_mean_response( df ) )
    tree.attribute = name_of_column

    split_dfs = [temp_df_before_split, temp_df_after_split]

    # Split the data set by response values greater than and less than or equal to mean            
    for split_dfs_index in range( 0, len(split_dfs) ):
        temp_df = split_dfs[split_dfs_index]
        if temp_df.shape[0] == 0:
            tree = ( Node( mean_response_for_node ) )
            return(tree)
        else:
            mean_response_for_node = calculate_mean_response( temp_df )
            subtree = CART_algorithm_for_prepruning ( temp_df, mean_response_for_node, 
              attributes_split_upon, attribute_best_split_point, this_threshold )
            # Store this dataframe in the node
            subtree.instances_labeled = temp_df    
            if split_dfs_index == 0:
                name_of_value = best_split_value
                tree.children["LessThan"] = subtree
            else:
                name_of_value = best_split_value
                tree.children["GreaterThan"] = subtree
    return ( tree )


# This method calculates the prediction accuracy of the decision tree on 
# the test set of data
def calculate_accuracy( trained_tree, test_df ):    
    # Determine index of the Class column
    column_names = test_df.columns.values.tolist()
    for column_index in range( 0, len(column_names) ):
        if column_names[column_index] == "Response":
            class_column_index = column_index
    
    number_instances = test_df.shape[0]
    MSE_for_accuracy = [0]
    # Obtain the prediction for each test data set instance and evaluate 
    # whether it equals the actual 'Class' value or not
    for test_index in range(0, number_instances):
        this_prediction = predict( trained_tree, test_df.iloc[test_index,] )
        mean_squared_difference = 
          ( test_df.iloc[test_index,class_column_index] - this_prediction ) ** 2
        MSE_for_accuracy.append( mean_squared_difference )
    total_MSE = sum(MSE_for_accuracy)
    averaged_MSE = total_MSE / len(MSE_for_accuracy)
    return ( averaged_MSE )


# This method predicts the response value for a test instance. 
def predict( node, test_instance ):
    # If there are no more children, return the response value for that node
    if len( node.children ) == 0:
        return ( node.label )
    # If there are child nodes, recurse on predict method
    else:
        # Obtain value for this attribute from test instance
        attribute_value = test_instance[node.attribute]

        if attribute_value > node.label:
            this_prediction = predict(node.children["GreaterThan"], test_instance)
            return ( this_prediction )

        elif attribute_value <= node.label:
            this_prediction = predict(node.children["LessThan"], test_instance)
            return ( this_prediction )

# This class builds a node object for use in building decision tree
class Node:
    def __init__(self, label):
        self.attribute = None
        self.label = label
        self.children = {}
        self.instances_labeled = []
        
                
################################################
############## Main Driver #####################
################################################      

# Load input file
df_original = pd.read_csv(input_file_name, header=[0], sep='\t')

# Parse input file name for output file
split_input_path = input_file_name.strip().split("/")
split_input_file_name = split_input_path[7].split("_")
output_file_name_list = []
words_to_drop_from_name = ['clean']
for split_index in range(0, len(split_input_file_name)):
    if words_to_drop_from_name[0] not in split_input_file_name[split_index]:
        output_file_name_list.append(split_input_file_name[split_index])
output_file_name_final = '_'.join(output_file_name_list)

# Write results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_3/Outputs/" + 
  output_file_name_final + "_output_Regression_Tree_Tuning", 'w') as tuneOutputFile:
    tuneOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    tuneOutputFile.write( "\nType of problem: Regression\n" )
    
    print( "\nData set: ", output_file_name_final, "\n" )
    print( "\nType of problem: Regression\n" )

    # Store 10% of the data as a validation set
    number_of_instances = df_original.shape[0]
    number_of_instances_for_validation_set = math.floor(0.1 * number_of_instances)
    validation_subset = 
      df_original.sample(n=number_of_instances_for_validation_set, axis = 0)
    row_numbers_validation = validation_subset.index.tolist()
    df_subset = df_original.drop( row_numbers_validation )
    validation_data_frame = validation_subset.reset_index( drop=True )
    df = df_subset.reset_index( drop=True )

    #########################################################################
    # Build regression tree, tuning for the number of points remaining in bin
    #########################################################################

    # Split data set into 5 folds for cross-validation
    test_folds_row_numbers = split_for_cross_validation(df, 5)
    
    # Tune 
    tuneOutputFile.write( "\n******************************************\n" )
    tuneOutputFile.write( "\nCART Regression, tuning the hyperparameter\n" )
    tuneOutputFile.write( "\n******************************************\n" )
    
    # Create 2-dimensional arrays to store MSE calculations
    MSE_array_validation = np.zeros((5, 7))
    MSE_array_test = np.zeros((5, 7))
    
    for index_lists in range(0, len(test_folds_row_numbers)):
        tuneOutputFile.write( "\nWhen fold " + str(index_lists) + " is the test set:\n" )
        # Obtain row numbers for test fold
        df_row_list = test_folds_row_numbers[index_lists]
        # Obtain test data frame using row numbers for test fold
        test_data_frame = df.iloc[df_row_list,]
        test_data_frame = test_data_frame.reset_index( drop=True )
        # Obtain train data frame by dropping row numbers for test fold
        train_data_frame = df.drop( df.index[df_row_list] )
        train_data_frame = train_data_frame.reset_index( drop=True )

        mean_response_for_node = calculate_mean_response( train_data_frame )    

        # Initialize empty lists to prevent recursing on the same attribute 
        # over and over inside the method implementing the CART algorithm 
        attributes_split_upon = [0, 0]
        attribute_best_split_point = [0, 0]
            	
    	# Set threshold values for number of data points in a bin when branching stops
        list_of_threshold_values = [0, 25, 50, 75, 100, 150, 200]
        for threshold_value_index in range(0, len(list_of_threshold_values)):
            tuneOutputFile.write( "\nFor " + 
              str(list_of_threshold_values[threshold_value_index]) + 
              " data points in partition:\n" )
            print( "\nFor", str(list_of_threshold_values[threshold_value_index]), 
              "data points in partition:\n" )
            this_threshold = list_of_threshold_values[threshold_value_index]
			
            # Call CART algorithm to build tree; tune here        
            tuned_tree = CART_algorithm_for_prepruning( train_data_frame, 
              mean_response_for_node, attributes_split_upon, attribute_best_split_point, 
              this_threshold )
            
            # Calculate MSE for validation
            MSE_measure_with_tuning_validation = 
              calculate_accuracy( tuned_tree, validation_data_frame )
            MSE_array_validation[index_lists, threshold_value_index] = 
              MSE_measure_with_tuning_validation           
            print("MSE_measure_with_tuning_validation: ", MSE_measure_with_tuning_validation)
            tuneOutputFile.write( "\tMSE, validation: " + 
              str(round( MSE_measure_with_tuning_validation, 5 )) + "\n" )
            
            # Calculate MSE for test
            MSE_measure_with_tuning_test = calculate_accuracy( tuned_tree, test_data_frame )
            MSE_array_test[index_lists, threshold_value_index] = 
              MSE_measure_with_tuning_test            
            print("MSE_measure_with_tuning_test: ", MSE_measure_with_tuning_test)
            tuneOutputFile.write( "\tMSE, test: " + 
              str(round( MSE_measure_with_tuning_test, 5 )) + "\n" )
        
        print("\nMSE_array_validation: ", MSE_array_validation)        
        print("\nMSE_array_test: ", MSE_array_test)


    tuneOutputFile.write( "\n**********************************************\n" )
    tuneOutputFile.write( "\nSummary Statistics, Validation Set Performance\n" )
    tuneOutputFile.write( "\n**********************************************\n" )


    tuneOutputFile.write( "\n\nHyperparameter values for tuning: " + 
      str(list_of_threshold_values) + "\n" )
	# Calculate average validation MSE for each threshold 
    final_average_MSE_for_thresholds_validation = MSE_array_validation.mean(axis = 0)
    print("\nfinal_average_MSE_for_thresholds_validation: ", 
      final_average_MSE_for_thresholds_validation)
    tuneOutputFile.write( "\nAverage MSE over 5 fold cross validation: " + 
      str(final_average_MSE_for_thresholds_validation) + "\n" )
    final_std_MSE_for_thresholds_validation = MSE_array_validation.std(axis = 0)
    print("\n\final_std_MSE_for_thresholds_validation: ", 
      final_std_MSE_for_thresholds_validation)
    tuneOutputFile.write( "\nStandard deviation of MSE over 5 fold cross validation: " + 
      str(final_std_MSE_for_thresholds_validation) + "\n" )
	
    tuneOutputFile.write( "\n**********************************************\n" )
    tuneOutputFile.write( "\n  Summary Statistics, Test Set Performance\n" )
    tuneOutputFile.write( "\n**********************************************\n" )
	
    tuneOutputFile.write( "\n\nHyperparameter values for tuning: " + 
      str(list_of_threshold_values) + "\n" )
	# Calculate average test MSE for each threshold 
    final_average_MSE_for_thresholds_test = MSE_array_test.mean(axis = 0)
    print("final_average_MSE_for_thresholds_test: ", final_average_MSE_for_thresholds_test)
    tuneOutputFile.write( "\nAverage MSE over 5 fold cross validation: " + 
      str(final_average_MSE_for_thresholds_test) + "\n" )
    final_std_MSE_for_thresholds_test = MSE_array_test.std(axis = 0)
    print("\nfinal_std_MSE_for_thresholds_validation: ", 
      final_std_MSE_for_thresholds_validation)
    tuneOutputFile.write( "\nStandard deviation of MSE over 5 fold cross validation: " + 
      str(final_std_MSE_for_thresholds_validation) + "\n\n" )	
	
# Close this output file
tuneOutputFile.close()	

#########################################################################
# Build regression tree without tuning and build regression tree with
# optimized number of data points remaining in bin, compare the two.
#########################################################################

# Find best threshold and use it to compare with untuned tree
best_threshold_index = min(enumerate(final_average_MSE_for_thresholds_test), 
  key=itemgetter(1))[0] 
print("best_threshold_index: ", best_threshold_index)
best_threshold_value = list_of_threshold_values[best_threshold_index]
print("best_threshold_value: ", best_threshold_value)
    
    
# Split data set into 5 folds for cross-validation
test_folds_row_numbers = split_for_cross_validation(df, 5)


# Prepare output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_3/Outputs/" + 
  output_file_name_final + "_output_Regression_Tree_Optimal_Hyperparameter", 'w') as tunedKOutputFile:
    tunedKOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    tunedKOutputFile.write( "\nType of problem: Regression\n" )
    
    tunedKOutputFile.write( "\n**************************************************************\n" )
    tunedKOutputFile.write( "\nCART Regression with optimal hyperparameter = " + 
      str( best_threshold_value ) + " data points\n" )
    tunedKOutputFile.write( "\n**************************************************************\n" )


    print( "\nData set: ", output_file_name_final, "\n" )
    print( "\nType of problem: Regression\n" )
    
    MSE_no_tuning = []
    MSE_with_tuning = []

    for index_lists in range(0, len(test_folds_row_numbers)):
        #print("When fold", index_lists, "is the test set:")
        tunedKOutputFile.write( "\nWhen fold " + str(index_lists) + " is the test set:\n" )
        # Obtain row numbers for test fold
        print('index lists: ', index_lists)
        df_row_list = test_folds_row_numbers[index_lists]
        # Obtain test data frame using row numbers for test fold
        test_data_frame = df.iloc[df_row_list,]
        test_data_frame = test_data_frame.reset_index( drop=True )
        # Obtain train data frame by dropping row numbers for test fold
        train_data_frame = df.drop( df.index[df_row_list] )
        train_data_frame = train_data_frame.reset_index( drop=True )

        mean_response_for_node = calculate_mean_response( train_data_frame )    

        # Initialize empty lists to prevent recursing on the same attribute 
        # over and over inside the method implementing the CART algorithm 
        attributes_split_upon = [0, 0]
        attribute_best_split_point = [0, 0]        

        # Call CART algorithm to build tree; no tuning here
        tree = CART_algorithm( train_data_frame, mean_response_for_node, 
          attributes_split_upon, attribute_best_split_point )
        MSE_measure_without_tuning = calculate_accuracy( tree, test_data_frame )
        MSE_no_tuning.append(MSE_measure_without_tuning)
        tunedKOutputFile.write( "\tMSE, no hyperparameter: " + 
          str(round( MSE_measure_without_tuning, 5 )) + "\n" )

        #Build a tree with best tuning parameter
        tuned_tree = CART_algorithm_for_prepruning( validation_data_frame, 
          mean_response_for_node, attributes_split_upon, attribute_best_split_point, 
            best_threshold_value )
        # Calculate accuracy of tuned tree
        MSE_measure_with_tuning = calculate_accuracy( tuned_tree, test_data_frame )
        MSE_with_tuning.append( MSE_measure_with_tuning )
        tunedKOutputFile.write( "\tMSE, with hyperparameter: " + 
          str(round( MSE_measure_with_tuning, 5 )) + "\n" )

    print("MSE_no_tuning: ", MSE_no_tuning)
    print("MSE_with_tuning: ", MSE_with_tuning)

    average_MSE_no_tuning = np.mean( MSE_no_tuning )
    tunedKOutputFile.write( "\n\nAverage MSE, no tuned hyperparameter: " + 
      str(round( average_MSE_no_tuning, 5 )) + "\n" )
    print("average_MSE_no_tuning: ", average_MSE_no_tuning)
    std_MSE_no_tuning = np.std( MSE_no_tuning )
    print("\std_MSE_no_tuning: ", std_MSE_no_tuning)
    tunedKOutputFile.write( "Standard deviation of MSE over 5 fold cross validation: " + 
      str(std_MSE_no_tuning) + "\n\n" )	
	
    
    average_MSE_with_tuning = np.mean( MSE_with_tuning )
    tunedKOutputFile.write( "\nAverage MSE, with tuned hyperparameter: " + 
      str(round( average_MSE_with_tuning, 5 )) + "\n" )
    print("average_MSE_with_tuning: ", average_MSE_with_tuning)
    std_MSE_with_tuning = np.std( MSE_with_tuning )
    print("\nfinal_std_MSE_for_thresholds_validation: ", std_MSE_with_tuning)
    tunedKOutputFile.write( "Standard deviation of MSE over 5 fold cross validation: " + 
      str(std_MSE_with_tuning) + "\n\n" )
    
# Close this output file
tunedKOutputFile.close()


