##
 #  This program uses the iterative dichotomiser (ID3) algorithm to build
 #  a decision tree for classification problems. Input files are pre-processed data sets 
 #  that must be formatted such that the target variable for prediction is labelled
 #  "Class" and located in the last column of the data frame. 
 #  Ten percent of a data set is removed as a validation set and the same validation set 
 #  is used for all experiments. Five-fold cross-validation is performed. Summary 
 #  statistics are output to a file.  
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
    # Create empty list to hold the final row numbers for each test fold
    list_of_row_numbers_for_test_folds = []
    list_of_row_numbers_for_test_folds_validation = []
    
    number_instances = data_frame.shape[0]
    # Calculate number of instances needed for validation set
    number_instances_validation = math.floor(number_instances * 0.1)
    # Calculate number of instances needed for cross-validation
    available_instances = number_instances - number_instances_validation

    # Calculate the number of data instances per fold 
    number_instances_per_fold = math.floor(available_instances / number_of_folds)

    # Create empty list to hold k number of test data sets
    list_of_data_sets = []
    
    # Create empty list to hold row numbers corresponding to each class
    list_of_class_row_numbers = []

    # Create empty list to hold proportion of each class
    number_instances_of_each_class = []
    
    number_instances_of_each_class_for_validation = []

    # Determine the number of instances of each class
    class_breakdown = data_frame.groupby(["Class"])["Class"].count()

    # Determine the row numbers in data frame that correspond to each class
    for class_index in range(0, len(class_breakdown)):
        # Create a temporary data frame containing instances from a given class
        temp_data_frame = data_frame.loc[data_frame['Class'] == 
          class_breakdown.index.values[class_index]]
        #print("temp_data_frame: ", temp_data_frame)
        # Determine the actual row numbers of those instances
        row_numbers_for_class = list(temp_data_frame.index.values.astype(int))
        # Append these row numbers to list
        list_of_class_row_numbers.append(row_numbers_for_class)
        # Calculate the ratio class instances:number total instancess in big data set
        composition_ratio = len(row_numbers_for_class) / number_instances
        # Calculate the number of instances needed for each fold
        number_instances_of_this_class_needed = 
          math.floor(number_instances_per_fold * composition_ratio)      
        #print("number_instances_of_this_class_needed: ", 
          number_instances_of_this_class_needed)
        rounded_number_instances_of_this_class_needed = 
          round(number_instances_of_this_class_needed, 0)
        number_instances_of_each_class.append( rounded_number_instances_of_this_class_needed)
        
        # Calculate the ratio class instances:number total instancess in big data set
        # Calculate the number of instances needed for each fold
        number_instances_of_this_class_needed_validation = 
          math.floor(number_instances_validation * composition_ratio)      
        rounded_number_instances_of_this_class_needed_validation = 
          round(number_instances_of_this_class_needed_validation, 0)
        number_instances_of_each_class_for_validation.append( rounded_number_instances_of_this_class_needed_validation)
        
    # Obtain row numbers for cross validation
    for k_index in range(0, number_of_folds):
        #print("\n\n\nThis is FOLD: ", k_index)
        # Empty list to store the row numbers for current fold
        temp_row_numbers_for_this_fold = []
        # Grab the row numbers needed for each class to be represented
        for class_index in range(0, len(list_of_class_row_numbers)):
            #print("\nThis is CLASS: ", class_index)            
            # The number of instances needed from given class
            number_instances_needed = number_instances_of_each_class[class_index]
            #print("number instances needed from this class: ", number_instances_needed)
            # Access eligible row numbers from given class
            row_numbers_of_interest = list_of_class_row_numbers[class_index]
            # Initialize counter variable
            counter = 0
            while counter < number_instances_needed:
                # Randomly select the index of the eligible row number for given class
                index_of_row_number_to_grab = random.randrange(len(row_numbers_of_interest))
                # Access the actual row number from original data frame
                row_number_to_grab = row_numbers_of_interest[index_of_row_number_to_grab]
                # Append this row number to list
                temp_row_numbers_for_this_fold.append(row_number_to_grab)
                # Remove this row number from list of eligible row numbers
                row_numbers_of_interest.pop(index_of_row_number_to_grab)
                # Increment counter variable by 1
                counter = counter + 1
                #print(counter)
        # Append all row numbers to final list
        list_of_row_numbers_for_test_folds.append( temp_row_numbers_for_this_fold )        
        
    # Obtain instances for validation set
    temp_row_numbers_for_validation = []
    for class_index in range(0, len(list_of_class_row_numbers)):
        #print("\nThis is CLASS: ", class_index)            
        # The number of instances needed from given class
        number_instances_needed_validation = 
          number_instances_of_each_class_for_validation[class_index]
        # Access eligible row numbers from given class
        row_numbers_of_interest = list_of_class_row_numbers[class_index]
        # Initialize counter variable
        counter = 0
        while counter < number_instances_needed_validation:
            # Randomly select the index of the eligible row number for given class
            index_of_row_number_to_grab = random.randrange(len(row_numbers_of_interest))
            # Access the actual row number from original data frame
            row_number_to_grab = row_numbers_of_interest[index_of_row_number_to_grab]
            # Append this row number to list
            temp_row_numbers_for_validation.append(row_number_to_grab)
            # Remove this row number from list of eligible row numbers
            row_numbers_of_interest.pop(index_of_row_number_to_grab)
            # Increment counter variable by 1
            counter = counter + 1
            #print(counter)
    # Append all row numbers to final list
    list_of_row_numbers_for_test_folds_validation.append( temp_row_numbers_for_validation )    

    return( list_of_row_numbers_for_test_folds, list_of_row_numbers_for_test_folds_validation )


# This method determines the identities of the unique values of an 
# attribute and the number of times it appears
def get_one_attribute_counts( df, column_number ):
    # Convert column to a list
    attrubite_as_list = df.iloc[:, column_number].tolist()
    # Obtain the unique names of the values
    values_in_class = list( set( attrubite_as_list ) )
    # Create empty list to store the unique value names
    list_of_value_names = []
    # Create empty list to store the number of times each 
    # value name is in the list
    list_of_value_counts = []
    # Count the number of times each value name  
    # appears in column
    for value_index in range( 0, len(values_in_class) ):
        counter_i = 0    
        for list_index in range( 0, len(attrubite_as_list) ):
            if attrubite_as_list[list_index] == values_in_class[value_index]:
                counter_i = counter_i + 1
        list_of_value_names.append( values_in_class[value_index] )
        list_of_value_counts.append( counter_i )
    # Return a list of the names of the attribute values and
    # a list of the number of times each value appears
    return( list_of_value_names, list_of_value_counts )


# This method calls the get_one_attribute_counts method above and 
# returns the most common value in the 'Class' column
def most_popular_class_value( df ):
    # Determine index of the Class column
    column_names = df.columns.values.tolist()
    for column_index in range( 0, len(column_names) ):
        if column_names[column_index] == "Class":
            class_column_index = column_index

    # Calculate entropy of the data set or subset
    list_of_class_names, list_of_class_counts = 
      get_one_attribute_counts( df, class_column_index )

    largest_count = list_of_class_counts[0]
    column_number = 0
    for count_index in range( 0, len(list_of_class_counts) ):
        if list_of_class_counts[count_index] > largest_count:
            largest_count = list_of_class_counts[count_index]
            column_number = count_index
    most_common_class = list_of_class_names[column_number]
    return( most_common_class )


# This method calculates the entropy for a data set
def calculate_data_set_entropy( list_of_class_counts ):
    number_of_instances = sum( list_of_class_counts )
    # Initialize a variable to track the total entropy
    entropy_of_data_set = 0
    for k in range( 0, len(list_of_class_counts) ):
        this_fraction_data_set = ( list_of_class_counts[k] )/number_of_instances
        if this_fraction_data_set == 0:
            continue
        else:
            this_entropy_component = 
              -( this_fraction_data_set*math.log2(this_fraction_data_set) )
            entropy_of_data_set = entropy_of_data_set + this_entropy_component
    return( entropy_of_data_set )


# This method performs the actual calculation of entropy for one attribute.
# It is called by the calculate_entropy_for_attribute method below.
def calculate_entropies_in_attribute( counts_one_value_by_class, number_of_instances ):
    # Determine total number of instances
    sum_counts_one_value_by_class = sum( counts_one_value_by_class )
    # Initialize variable to track the total entropy
    running_info_content = 0
    for class_counts_index in range( 0, len(counts_one_value_by_class) ):
        if counts_one_value_by_class[class_counts_index] == 0:
            continue        
        else:            
            # Obtain the fractional component of one value in the attribute
            fractional_component = ( counts_one_value_by_class[class_counts_index] ) / 
              sum_counts_one_value_by_class
            # Calculate entropy
            result_of_this_calculation = 
              -( fractional_component * math.log2(fractional_component) )
            # Add the entropy value to the running sum of value entropies
            running_info_content = running_info_content + result_of_this_calculation
    fraction_times_info_content = 
      running_info_content * ( sum_counts_one_value_by_class / number_of_instances )
    return( fraction_times_info_content )


# This method calculates the entropy in one attribute
def calculate_entropy_for_attribute( df, column_index, list_of_class_names ):
    list_of_names = []
    list_of_counts = []

    # Determine index of the Class column
    column_names = df.columns.values.tolist()
    for k in range( 0, len(column_names) ):
        if column_names[k] == "Class":
            class_column_index = k    

    # Convert the column to a list
    column_as_list = df.iloc[:,column_index].tolist()
    # Obtain the unique value names
    values_in_column = list( set(column_as_list) )

    # Count the number of times each value appears in column
    for value_index in range( 0, len(values_in_column) ):
        counter_i = 0    
        for list_index in range(0, len(column_as_list)):
            if column_as_list[list_index] == values_in_column[value_index]:
                counter_i = counter_i + 1
        list_of_names.append( values_in_column[value_index] )
        list_of_counts.append( counter_i )

    entropy_of_each_feature_value = []
    # For one of the values in one of the feature lists
    for names_index in range( 0, len(list_of_names) ):
        counts_one_value_by_class = []    
        # "yes" or "no"
        for class_summary_index in range( 0, len(list_of_class_names) ):
            counter = 0
            # Iterate over all rows
            number_of_instances = df.shape[0]
            for instance_index in range( 0, number_of_instances ):
                if df.iloc[instance_index, column_index] == 
                  list_of_names[names_index] and df.iloc[instance_index, class_column_index] == 
                  list_of_class_names[class_summary_index]:
                    counter = counter + 1
            counts_one_value_by_class.append( counter )
        # Call the calculate_entropies_in_attribute method to perform the entropy calculation
        fraction_times_info_content = 
          calculate_entropies_in_attribute( counts_one_value_by_class, number_of_instances )
        entropy_of_each_feature_value.append( fraction_times_info_content )
    # Add entropy for each feature value to obtain entropy of entire attribute
    entropy_for_attribute = sum( entropy_of_each_feature_value )
    return( entropy_for_attribute )


# This method calculates the information value of a feature
def information_value( df, column_index ):
    list_of_names = []
    list_of_counts = []

    number_of_instances = df.shape[0]

    # Convert the feature column to a list
    column_as_list = df.iloc[:,column_index].tolist()

    # Obtain the unique feature value names
    values_in_column = list( set(column_as_list) )
    # Count the number of times each feature value appears in column
    for value_index in range( 0, len(values_in_column) ):
        counter_i = 0    
        for list_index in range( 0, len(column_as_list) ):
            if column_as_list[list_index] == values_in_column[value_index]:
                 counter_i = counter_i + 1
        list_of_names.append( values_in_column[value_index] )
        list_of_counts.append( counter_i )

    # Initialize a variable to track the total information value 
    information_value = 0    
    for k in range( 0, len(list_of_counts) ):
        # Calculate the information value
        information_value = 
          information_value + -( (list_of_counts[k] / number_of_instances) * 
          math.log2(list_of_counts[k] / number_of_instances) )
    return( information_value )


# This method determines which attribute has the largest gain 
# ratio and returns the column number of that attribute 
def find_attribute_largest_gain_ratio( gain_for_all_attributes ):    
    largest_gain = gain_for_all_attributes[0]
    column_number = 0
    for gain_index in range( 0, len(gain_for_all_attributes) ):
        if gain_for_all_attributes[gain_index] > largest_gain:
            largest_gain = gain_for_all_attributes[gain_index]
            column_number = gain_index
    return( column_number )


# This method implements the ID3 algorithm
def ID3_algorithm( df, most_common_class_for_node, list_of_attributes ):

    # Determine index of the Class column
    column_names = df.columns.values.tolist()
    for column_index in range( 0, len(column_names) ):
        if column_names[column_index] == "Class":
            class_column_index = column_index

    # If there are no more instances, return node
    df_instances = df.shape[0]
    if df_instances == 0:
        return ( Node( most_common_class_for_node ) )

    # If only one instance remains, return tree
    if df_instances == 1:
        tree = Node( most_popular_class_value( df ) )
        return ( tree )

    # If the remaining instances are of the same class, return tree
    class_as_list = df.iloc[:, class_column_index].tolist()
    values_in_class = list( set( class_as_list ) )        
    if len( values_in_class ) == 1:
        tree = Node( most_popular_class_value( df ) )
        return ( tree )

    # Calculate entropy of the data set or subset
    list_of_class_names, list_of_class_counts = 
      get_one_attribute_counts( df, class_column_index )
    entropy_of_data_set = calculate_data_set_entropy( list_of_class_counts )

    # Determine the location of available attributes 
    location_of_attribute = []
    for i in range( 0, len(list_of_attributes) ):
        attribute_name = str( list_of_attributes[i] )
        location_of_attribute.append( df.columns.get_loc(attribute_name) )
    # Determine the gain ratio for attributes
    gain_ratio_for_all_attributes = []
    for column_index in range( 0, len(location_of_attribute) ):
        an_attribute_entropy = 
          calculate_entropy_for_attribute( df, location_of_attribute[column_index], 
          list_of_class_names )
        an_attribute_gain = entropy_of_data_set - an_attribute_entropy
        an_information_value = information_value( df, location_of_attribute[column_index] )
        if an_information_value == 0:
            gain_ratio = 0
        else: 
            gain_ratio = an_attribute_gain / an_information_value
        gain_ratio_for_all_attributes.append( gain_ratio )
    if not gain_ratio_for_all_attributes:
        tree = Node( most_popular_class_value( df ) )
        return ( tree )
    
    # Determine attribute with largest gain ratio
    if len( gain_ratio_for_all_attributes ) == 1:
        column_index_largest_gain = column_index
        attribute_index_largest_gain = location_of_attribute[column_index]
    else:
        column_index_largest_gain = 
          find_attribute_largest_gain_ratio( gain_ratio_for_all_attributes )    
        attribute_index_largest_gain = location_of_attribute[column_index_largest_gain]

    # Assign attribute_index_largest_gain to root node here
    temp_name_of_column = str( df.columns[attribute_index_largest_gain] )    
    # Determine the values and counts of the attribute with the largest gain ratio
    decision_node_names, decision_node_values = 
      get_one_attribute_counts( df, attribute_index_largest_gain )

    # Assign attribute, attribute values, and the most popular class
    # value from attribute that had the largest gain ratio
    tree = Node( most_popular_class_value( df ) )
    tree.attribute = temp_name_of_column
    tree.attribute_values = decision_node_names

    # Split the data set based upon the attribute with largest gain ratio
    for i in range( 0, len(decision_node_values) ):        
        temp_df = df.loc[df.iloc[:,attribute_index_largest_gain] == decision_node_names[i]]
        # Recurse on ID3 algorithm
        subtree = 
          ID3_algorithm(temp_df, most_popular_class_value( temp_df ), list_of_attributes)        
        # Store the sub (split) data frame
        subtree.subsets = temp_df

        # Store the name of the attribute value
        name_of_value = decision_node_names[i]
        tree.children[name_of_value] = subtree
    return ( tree )

# This class builds a node object for use in building decision tree
class Node:
    def __init__(self, label):
        self.attribute = None
        self.attribute_values = []
        self.label = label
        self.children = {}
        self.subsets = []
        
# This method calculates the prediction accuracy of 
# the decision tree on the test set of data
def calculate_accuracy( trained_tree, test_df ):
    # Determine index of the Class column
    column_names = test_df.columns.values.tolist()
    for column_index in range( 0, len(column_names) ):
        if column_names[column_index] == "Class":
            class_column_index = column_index

    # Initialize a counter variable to track the 
    # number of predictions that are correct
    counter_correct_predictions = 0
    number_instances = test_df.shape[0]

    # Obtain the prediction for each test data set instance and 
    # check if it equals the actual 'Class' value or not
    for test_index in range(0, number_instances):
    	# Current instance being tested
        instance_being_tested = test_df.iloc[test_index,]
        # Store the correct label for this instance
        correct_label = test_df.iloc[test_index, class_column_index]
        # Predict the class value and store it
        predicted_label = make_prediction( trained_tree, instance_being_tested )
        # Check whether the labels are equal, if so increase the counter by 1
        if predicted_label == correct_label:
            # Increment the counter
            counter_correct_predictions = counter_correct_predictions + 1    
    accuracy_measure = ( counter_correct_predictions / number_instances )
    return ( accuracy_measure )


# This method predicts the response value for a test instance.
def make_prediction( node, test_instance ):
    # If node doesn't have any children, return the node's class value
    if len( node.children ) == 0:
        return ( node.label )
    # If there are child nodes, recurse on predict method
    else:
        # Obtain attribute value for this test instance
        this_attribute_value = test_instance[node.attribute]        
        if this_attribute_value in node.children:
            this_prediction = 
              make_prediction(node.children[this_attribute_value], test_instance)
            return ( this_prediction )
        else:
            class_result = []
            for these_attribute_values in node.attribute_values:
                this_attribute_instances = node.children[these_attribute_values].subsets
                class_values_of_the_instances = this_attribute_instances["Class"].values

                for class_values_index in range(0, len(class_values_of_the_instances)):
                    class_result.append(class_values_of_the_instances[class_values_index])                
            # Determine the most common class
            count_classes = Counter(class_result)
            most_common_class_and_count = count_classes.most_common(1)
            most_common_class_only = most_common_class_and_count[0][0]
            return( most_common_class_only )

        
    
################################################
############## Main Driver #####################
################################################   

# Load input file
df = pd.read_csv( input_file_name, header=[0], sep='\t' )

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
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_3/Outputs/" + output_file_name_final + "_output_Classification_Tree", 'w') as tuneOutputFile:
    tuneOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    tuneOutputFile.write( "\nType of problem: Classification\n" )
    
    print( "\nData set: ", output_file_name_final, "\n" )
    print( "\nType of problem: Classification\n" )
    
    tuneOutputFile.write( "\n***********************\n" )
    tuneOutputFile.write( "\n ID3 Classification\n" )
    tuneOutputFile.write( "\n***********************\n" )


    # Extract the names of the attributes
    column_names = df.columns.values.tolist()
    column_names_df = []
    for column_names_index in range(0, (len(column_names)-1)):
        column_names_df.append(column_names[column_names_index])

    # Create empty lists to store accuracy data
    accuracy_no_pruning = []
    accuracy_with_pruning = []

    # Split data set into 5 folds for cross-validation
    test_folds_row_numbers, validation_set_row_numbers = split_for_cross_validation(df, 5)

    # Obtain validation data frame using row numbers for test fold
    
    for index_lists in range(0, len(validation_set_row_numbers)): 
        val_row_list = validation_set_row_numbers[index_lists]   
        validation_subset = df.iloc[val_row_list]
        validation_data_frame = validation_subset.reset_index( drop=True )

    for index_lists in range(0, len(test_folds_row_numbers)):
        print("  When fold", index_lists, "is the test set:")
        tuneOutputFile.write( "\nWhen fold " + str(index_lists) + " is the test set:\n" )
        # Obtain row numbers for test fold
        df_row_list = test_folds_row_numbers[index_lists]
        # Obtain test data frame using row numbers for test fold
        test_data_frame = df.iloc[df_row_list,]
        test_data_frame = test_data_frame.reset_index( drop=True )
        # Obtain train data frame by dropping row numbers for test fold
        train_data_frame = df.drop( df.index[df_row_list] )
        train_data_frame = train_data_frame.reset_index( drop=True )
        
        most_common_class_for_node = most_popular_class_value( train_data_frame )

        # Determine accuracy of an unpruned decision tree
        tree = ID3_algorithm( train_data_frame, most_common_class_for_node, column_names_df ) 
        accuracy_measure_without_pruning = calculate_accuracy( tree, test_data_frame )
        accuracy_no_pruning.append( accuracy_measure_without_pruning )
        print("accuracy_no_pruning: ", accuracy_no_pruning)
        tuneOutputFile.write( "\tAccuracy, no pruning: " + 
          str(round( accuracy_measure_without_pruning, 5 )) + "\n" )
        
    average_accuracy_no_tuning = np.mean( accuracy_no_pruning )
    tuneOutputFile.write( "\n\nAverage accuracy, no pruning: " + 
      str(round( average_accuracy_no_tuning, 5 )) + "\n" )
    print("average_accuracy_no_tuning: ", average_accuracy_no_tuning)
    std_accuracy_no_tuning = np.std( accuracy_no_pruning )
    print("\std_accuracy_no_tuning: ", std_accuracy_no_tuning)
    tuneOutputFile.write( "Standard deviation of accuracy over 5 fold cross validation: " + 
      str(std_accuracy_no_tuning) + "\n\n" )	
	






