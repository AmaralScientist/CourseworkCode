##
 #  This program uses the k nearest neighbors (KNN) algorithm, the condensed KNN
 #  algorithm, and the edited KNN algorithm to train and test machine learning 
 #  classifiers. Five-fold cross-validation is performed and a KNN, condensed KNN, 
 #  and edited KNN classifier are each tuned to determine the optimal value of k
 #  between k = 1 to k = 12. Summary statistics are output to separate files. The
 #  data set is again split for five-fold cross validation and the optimal value
 #  of k for each respective algorithm is used to train and test a classifier. The
 #  summary statistics for these experiments are output to separate files. 
 #  The input is a pre-processed data set organized in a pandas dataframe such that
 #  the target class is located in the first column and the attributes in the 
 #  remaining columns.
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

# Receive the name of input file from the command line
input_file_name = sys.argv[1]
        
# This method accesses a data frame and splits its row numbers into k 
# lists of row numbers, each corresponding to k unique test folds
# for use with k fold cross-validation
def split_for_cross_validation(data_frame, number_of_folds):
    # Create empty list to hold the final row numbers for each test fold
    list_of_row_numbers_for_test_folds = []

    # Calculate the number of data instances per fold 
    number_instances_per_fold = math.floor(data_frame.shape[0] / number_of_folds)

    # Create empty list to hold k number of test data sets
    list_of_data_sets = []

    # Create empty list to hold row numbers corresponding to each class
    list_of_class_row_numbers = []

    # Create empty list to hold proportion of each class
    number_instances_of_each_class = []

    # Determine the number of instances of each class
    class_breakdown = data_frame.groupby(["Class"])["Class"].count()

    # Determine the row numbers in data frame that correspond to each class
    for class_index in range(0, len(class_breakdown)):
        # Create a temporary data frame containing instances from a given class
        temp_data_frame = data_frame.loc[data_frame['Class'] == 
          class_breakdown.index.values[class_index]]
        # Determine the actual row numbers of those instances
        row_numbers_for_class = list(temp_data_frame.index.values.astype(int))
        # Append these row numbers to list
        list_of_class_row_numbers.append(row_numbers_for_class)
        # Calculate the ratio class instances:number total instancess in big data set
        composition_ratio = len(row_numbers_for_class) / data_frame.shape[0]
        # Calculate the number of instances needed for each fold
        number_instances_of_this_class_needed = 
          number_instances_per_fold * composition_ratio        
        rounded_number_instances_of_this_class_needed = 
          round(number_instances_of_this_class_needed, 0)
        number_instances_of_each_class.append( rounded_number_instances_of_this_class_needed)

    # In each fold, maintain the same ratio of the classes as in the full data set
    for k_index in range(0, number_of_folds):
        # Create empty list to store the row numbers for current fold
        temp_row_numbers_for_this_fold = []
        # Grab the row numbers needed for each class to be represented
        for class_index in range(0, len(list_of_class_row_numbers)):         
            # The number of instances needed from given class
            number_instances_needed = number_instances_of_each_class[class_index]
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
        # Append all row numbers to final list
        list_of_row_numbers_for_test_folds.append( temp_row_numbers_for_this_fold )
    
    # Return the unique row numbers for all 5 test sets
    return( list_of_row_numbers_for_test_folds )

# This method implements the KNN algorithm by calculating Euclidean distances,
# identifying the k nearest neighbors, predicting a class based on the classes
# of the nearest neighbors, evaluates the accuracy of the predictions, and 
# returns that measure. 
def knn_algorithm_for_tuning( temp_test_data_frame, temp_train_data_frame, k_index ):    
    # Determine the number of rows in the test data frame
    number_test_instances = temp_test_data_frame.shape[0]
    # Determine the number of rows in the train data frame
    number_train_instances = temp_train_data_frame.shape[0]
    # Create temporary numpy array to store results
    temp_df_for_test_results = np.zeros( shape=( number_test_instances, 5 ) )
    # Create dataframe only consisting of attributes for test data
    temp_test_df_attributes_only = temp_test_data_frame.drop(["Class"], axis=1)
    
    # Initialize a variable to count the correctly classified instances
    correct_class_counter = 0
    # Initialize a variable to count the incorrectly classified instances
    incorrect_class_counter = 0
    for test_row_index in range( 0, number_test_instances ):
        # Create temporary numpy array to store euclidean distances
        temp_df_for_distances = np.zeros( shape=( number_train_instances, 2 ) )
        # Create dataframe only consisting of attributes for train data
        temp_train_df_attributes_only = temp_train_data_frame.drop(["Class"], axis=1)
        temp_train_class_column = temp_train_data_frame["Class"]
        number_train_columns = temp_test_df_attributes_only.shape[1]

        # Calculate Euclidean distance between test instance and training instances
        for train_row_index in range( 0, number_train_instances ):
        	# Initialize a variable to sum the squared distance of each feature
            squared_distance = 0
            # Increment over each feature to obtain the difference between 
            # each test and train instance
            for feature_index in range( 0, number_train_columns ):
                squared_distance = squared_distance + 
                  ( temp_test_df_attributes_only.iloc[test_row_index, feature_index] - 
                  temp_train_df_attributes_only.iloc[train_row_index, feature_index] )** 2               
            # Calculate the square root of the sum of the differences
            euclidean_distance = np.sqrt( squared_distance )
            # Store the Euclidean distances and the location of that training instance 
            temp_df_for_distances[train_row_index, 0] = euclidean_distance
            temp_df_for_distances[train_row_index, 1] = train_row_index
        # Sort the numpy array containing the distances
        sorted_temp_df_for_distances = 
          temp_df_for_distances[temp_df_for_distances[:,0].argsort()]
        # Grab the k neighbors that are closest to the training instance
        k_nearest_neighbors = sorted_temp_df_for_distances[0:k_index,]
        
        # Store the classes of these neighbors
        correct_class = []
        for j in range(0, k_index):            
            class_row_number = int(k_nearest_neighbors[j, 1])
            correct_class.append(temp_train_data_frame.iloc[class_row_number, 0])
        # Count the number of each class from the neighbors
        k_classes = Counter(correct_class)
                
        # If all neighbors are of the same class, call the prediction        
        if len(k_classes) == 1:
            for key in k_classes:
                this_class_predicted = key
        # If the neighbors represent more than one class, find most frequent
        else:
            # Create empty list to hold the number of each class
            counts_of_neighbors = []
            # Create empty list to hold the names of each class
            class_name_of_neighbors = []
            # Append counter output to lists
            for thing in k_classes:
                counts_of_neighbors.append(k_classes[thing])
                class_name_of_neighbors.append(thing)
            # Initialize variable to track the most frequent class
            largest_count = 0
            # Empty list to hold class counts in case there is a tie
            tied_neighbor_counts = []
            # Empty list to hold class names in case there is a tie
            tied_neighbor_names = []
            # Iterate over the counts of the classes to find the largest
            for j in range(0, len(counts_of_neighbors)):
                if counts_of_neighbors[j] > largest_count:
                    largest_count = counts_of_neighbors[j]
                    name_of_class = class_name_of_neighbors[j]
                # If there is a tie, store name and count of the class
                elif counts_of_neighbors[j] == largest_count:
                    tied_neighbor_counts.append(counts_of_neighbors[j])
                    tied_neighbor_names.append(class_name_of_neighbors[j])
            # If there was not a tie call the predicted class
            if len(counts_of_neighbors) == 0:
                this_class_predicted = name_of_class
            # There was a tie, break it randomly
            else:
                tied_neighbor_counts.append(largest_count)
                tied_neighbor_names.append(name_of_class)                
                select_one_class_index = random.randrange(len(tied_neighbor_names))
                this_class_predicted = tied_neighbor_names[select_one_class_index]    

        # Compare predicted to actual
        correct_class = temp_test_data_frame.iloc[test_row_index, 0]

        if this_class_predicted == correct_class:
            correct_class_counter = correct_class_counter + 1
        else:
            incorrect_class_counter = incorrect_class_counter + 1

    #print("\tCorrectly classified instances: ", correct_class_counter)
    #print("\tIncorrectly classified instances: ", incorrect_class_counter)
    
    # Calculate the accuracy for this data subset
    performance = correct_class_counter / (correct_class_counter + incorrect_class_counter)
    # Return the performance value
    return( performance )


# This method accepts a training data set and implements the condensed nearest 
# neighbors algorithm to produce a smaller training data set 
def condensed_nearest_neighbor( temp_train_data_frame ):
    # Determine the number of rows in the train data frame
    number_train_instances = temp_train_data_frame.shape[0]
    # Create empty list to hold row numbers of the original training data set (X)
    original_training_set = []    
    # Append the row numbers to the list, then randomly shuffle
    for k in range(0, number_train_instances):
        original_training_set.append ( k )
    random.shuffle( original_training_set )

    # Create empty list to hold row numbers that will be contained in
    # the condensed training set (Z)
    condensed_training_set = []    
    # Move row number of the first instance to the condensed training set
    remove_one_training_example = original_training_set.pop( 0 )
    condensed_training_set.append( remove_one_training_example )

    # Perform 1NN until a full pass is made over the original training data set
    for p in range(0, len(original_training_set)):
        # Grab next training example from original training data set
        remove_one_training_example = original_training_set.pop( 0 )        
        # Create a temporary array to hold the Euclidean distances and the row number
        temp_df_for_distances = np.zeros( shape=( len( condensed_training_set ), 2 ) )
        # Create dataframe only consisting of attributes for train data
        temp_train_df_attributes_only = temp_train_data_frame.drop(["Class"], axis=1)
        number_train_columns = temp_train_df_attributes_only.shape[1]

        # Calculate the Euclidean distances between the example and the instances
        # contained in the condensed training data set (Z)
        for m in range(0, len(condensed_training_set)):            
            squared_distance = 0
            for feature_index in range( 0, number_train_columns ):
                squared_distance = squared_distance + 
                    ( temp_train_df_attributes_only.iloc[remove_one_training_example, feature_index] - 
                    temp_train_df_attributes_only.iloc[condensed_training_set[m], feature_index]  ) ** 2               
            euclidean_distance = np.sqrt( squared_distance )
            
            temp_df_for_distances[m, 0] = euclidean_distance
            temp_df_for_distances[m, 1] = condensed_training_set[m]
        # Sort the array by distance 
        sorted_temp_df_for_distances = 
          temp_df_for_distances[temp_df_for_distances[:,0].argsort()]
        # k = 1 so remove the row of the array with the smallest distance 
        k_nearest_neighbors = sorted_temp_df_for_distances[0,]   
        # Extract the row number corresponding to that instance in the data frame  
        class_row_number = int(k_nearest_neighbors[1,])
        # Extract the class of that instance
        predicted_class = temp_train_data_frame.iloc[class_row_number, 0]
        # Compare predicted class to actual class of the instance in question
        correct_class = temp_train_data_frame.iloc[remove_one_training_example, 0]

        # If the predicted and correct classes are not equal, append the 
        # row number of that instance to the condensed training set.
        # Otherwise, add it back to the original training data set
        if predicted_class != correct_class:
            condensed_training_set.append(remove_one_training_example)
        else:
            original_training_set.append(remove_one_training_example)

    # Initialize a variable to track the point at which no more instances
    # are being added to the condensed training data set
    instance_not_added_to_Z = 0
    # Continue performing 1NN until a complete pass is made over the 
    # original training data and no instances are added to the condensed set
    while ( len(original_training_set) != 0 ) and 
      instance_not_added_to_Z < len(original_training_set):        
        # Grab next training example from original training data set
        remove_one_training_example = original_training_set.pop( 0 )

        # Create a temporary array to hold the Euclidean distances and the row number
        temp_df_for_distances = np.zeros( shape=( len( condensed_training_set ), 2 ) )
        # Create dataframe only consisting of attributes for train data
        temp_train_df_attributes_only = temp_train_data_frame.drop(["Class"], axis=1)
        # Calculate the Euclidean distances between the example and the instances
        # contained in the condensed training data set (Z)
        for m in range(0, len(condensed_training_set)):
            squared_distance = 0
            for feature_index in range( 0, number_train_columns ):
                squared_distance = squared_distance + 
                  ( temp_train_df_attributes_only.iloc[remove_one_training_example, feature_index] - 
                    temp_train_df_attributes_only.iloc[condensed_training_set[m], feature_index] ) ** 2               
            euclidean_distance = np.sqrt( squared_distance )
            
            temp_df_for_distances[m, 0] = euclidean_distance
            temp_df_for_distances[m, 1] = condensed_training_set[m]

        # Sort the array by distance
        sorted_temp_df_for_distances = 
          temp_df_for_distances[temp_df_for_distances[:,0].argsort()]
        # k = 1 so remove the row of the array with the smallest distance
        k_nearest_neighbors = sorted_temp_df_for_distances[0,]
        # Extract the row number corresponding to that instance in the data frame 
        class_row_number = int(k_nearest_neighbors[1,])
        # Extract the class of that instance
        predicted_class = temp_train_data_frame.iloc[class_row_number, 0]
        # Compare predicted class to actual class of the instance in question
        correct_class = temp_train_data_frame.iloc[remove_one_training_example, 0]

        # If the predicted and correct classes are not equal, append the 
        # row number of that instance to the condensed training set.
        # Otherwise, add it back to the original training data set
        if predicted_class != correct_class:
            condensed_training_set.append(remove_one_training_example)
            # Reset indicator variable to 0 since the instance was added
            instance_not_added_to_Z = 0
        else:
            original_training_set.append(remove_one_training_example)
            # Instance was not added so increment by 1
            instance_not_added_to_Z = instance_not_added_to_Z + 1
    # Evaluate differences between original and condensed training data sets
    print("\tInstances before condensing: ", number_train_instances)
    print("\tInstances after condensing: ", len(condensed_training_set))
    reduction_in_number_of_instances = ( ( number_train_instances - 
      len(condensed_training_set) ) / number_train_instances )
    percent_reduction = reduction_in_number_of_instances * 100
    print("\tPercent reduction: ", round( percent_reduction, 2 ), "%\n")

    condensed_training_data_set = temp_train_data_frame.iloc[condensed_training_set,]
    return( condensed_training_data_set, reduction_in_number_of_instances )


# This method accepts a training data set and implements the edited 
# nearest neighbors algorithm to produce a smaller training data set 
def edited_nearest_neighbor( temp_train_data_frame ):
    # Determine the number of rows in the train data frame
    number_train_instances = temp_train_data_frame.shape[0]
    # Create empty list to hold row numbers of the original training data set
    original_training_set = []    
    # Append the row numbers to the list, then randomly shuffle
    for k in range(0, number_train_instances):
        original_training_set.append ( k )
    random.shuffle( original_training_set )
    
    number_train_columns = temp_train_data_frame.shape[1]

    # Perform k=3 nearest neighbors until a full pass
    # is made over the original training data set
    for p in range(0, number_train_instances):
        # Grab one training example from original training data set
        remove_one_training_example = original_training_set.pop( 0 )        
        # Create a temporary array to hold the Euclidean distances and the row number
        temp_df_for_distances = np.zeros( shape=( len( original_training_set ), 2 ) )
        # Create dataframe only consisting of attributes for train data
        temp_train_df_attributes_only = temp_train_data_frame.drop(["Class"], axis=1)
        number_train_columns = temp_train_df_attributes_only.shape[1]

        # Calculate the Euclidean distances between the example instance and
        # the remaining instances contained in the original training data set
        for m in range(0, len(original_training_set)):
            squared_distance = 0
            for feature_index in range( 0, number_train_columns ):
                squared_distance = squared_distance + 
                    ( temp_train_df_attributes_only.iloc[remove_one_training_example, feature_index] - 
                    temp_train_df_attributes_only.iloc[original_training_set[m], feature_index] ) ** 2               
            euclidean_distance = np.sqrt( squared_distance )
            
            temp_df_for_distances[m, 0] = euclidean_distance
            temp_df_for_distances[m, 1] = original_training_set[m]
        # Sort the temporary array by distance 
        sorted_temp_df_for_distances = 
          temp_df_for_distances[temp_df_for_distances[:,0].argsort()]
        # Take the three instances with the smallest distance
        k_nearest_neighbors = sorted_temp_df_for_distances[0:3,]
        # Obtain the class for these instances
        correct_class = []
        for j in range(0, 3):            
            class_row_number = int(k_nearest_neighbors[j, 1])
            correct_class.append(temp_train_data_frame.iloc[class_row_number, 0])
        k_classes = Counter(correct_class)
        predicted_class = max(k_classes, key=lambda key: k_classes[key])
        # Compare predicted class to actual class of the instance in question
        correct_class = temp_train_data_frame.iloc[remove_one_training_example, 0]

        # If the predicted and correct classes are equal, append the 
        # row number of that instance back to the original training set.
        # Otherwise, remove instance
        if predicted_class == correct_class:
            original_training_set.append(remove_one_training_example)
        else:
            remove_one_training_example = None 

    # Initialize a variable to track the point at which no more 
    # instances are being added back to the original training data set 
    instance_not_removed = 0

    while ( instance_not_removed < len(original_training_set) ):        
        # Grab next training example from original training data set
        remove_one_training_example = original_training_set.pop( 0 )        
        # Create a temporary array to hold the Euclidean distances and the row number
        temp_df_for_distances = np.zeros( shape=( len( original_training_set ), 2 ) )
        # Create dataframe only consisting of attributes for train data
        temp_train_df_attributes_only = temp_train_data_frame.drop(["Class"], axis=1)

        # Calculate the Euclidean distances between the example 
        # and the instances contained in the condensed training data set
        for m in range(0, len(original_training_set)):
            squared_distance = 0
            for feature_index in range( 0, number_train_columns ):
                squared_distance = squared_distance + 
                  ( temp_train_df_attributes_only.iloc[remove_one_training_example, feature_index] - 
                    temp_train_df_attributes_only.iloc[original_training_set[m], feature_index] ) ** 2               
            euclidean_distance = np.sqrt( squared_distance )
                        
            temp_df_for_distances[m, 0] = euclidean_distance
            temp_df_for_distances[m, 1] = original_training_set[m]
        # Sort the array by distance 
        sorted_temp_df_for_distances = 
          temp_df_for_distances[temp_df_for_distances[:,0].argsort()]
        # Take the three instances with the smallest distance
        k_nearest_neighbors = sorted_temp_df_for_distances[0:3,]
        # Obtain the class for these instances
        correct_class = []
        for j in range(0, 3):            
            class_row_number = int(k_nearest_neighbors[j, 1])
            correct_class.append(temp_train_data_frame.iloc[class_row_number, 0])
        k_classes = Counter(correct_class)
        predicted_class = max(k_classes, key=lambda key: k_classes[key])
        # Compare predicted class to actual class of the instance in question
        correct_class = temp_train_data_frame.iloc[remove_one_training_example, 0]

        # If the predicted and correct classes are equal, append the 
        # row number of that instance back to the original training set.
        # Otherwise, remove instance
        if predicted_class == correct_class:
            original_training_set.append(remove_one_training_example)
            instance_not_removed = instance_not_removed + 1
        else:
            remove_one_training_example = None 
            instance_not_removed = 0

    # Evaluate differences between original and condensed training data sets
    print("\tInstances before editing: ", number_train_instances)
    print("\tInstances after editing: ", len(original_training_set))
    reduction_in_number_of_instances = ( number_train_instances - 
      len(original_training_set) ) / number_train_instances
    percent_reduction = reduction_in_number_of_instances * 100
    print("\tPercent reduction: ", round(percent_reduction, 2), "%\n")

    edited_train_data_frame = temp_train_data_frame.iloc[original_training_set,]   
    return( edited_train_data_frame, reduction_in_number_of_instances )


        
################################################
############## Main Driver #####################
################################################

# Load input file
temp_df = pd.read_csv(input_file_name, header=[0], sep='\t')

# Parse input file name for output file
split_input_path = input_file_name.strip().split("/")
split_input_file_name = split_input_path[7].split("_")
output_file_name_list = []
words_to_drop_from_name = ['clean']
for split_index in range(0, len(split_input_file_name)):
    if words_to_drop_from_name[0] not in split_input_file_name[split_index]:
        output_file_name_list.append(split_input_file_name[split_index])
output_file_name_final = '_'.join(output_file_name_list)

################################################
# Perform KNN, tuning for the k parameter
################################################

# Write results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_KNN_Classifier_Tune_K", 'w') as tuneOutputFile:
    tuneOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    tuneOutputFile.write( "\nType of problem: Classification\n" )

    # Split data set into 5 folds 
    test_folds_row_numbers = split_for_cross_validation(temp_df, 5)

    # Tune the nearest neighbors parameter k
    tuneOutputFile.write( "\n************************\n" )
    tuneOutputFile.write( "\nTuning the parameter k\n" )
    tuneOutputFile.write( "\n************************\n" )

    # Create an empty list to store accuracy measure for each k
    performance_for_this_k = []
    # Create an empty list to track the values of k
    list_of_k_values = []
    for k_index in range(1, 13):
        # Store the value of k for this run
        list_of_k_values.append( k_index )
        # Output value of k for this run
        tuneOutputFile.write( "\nFor k = " + str(k_index) + ":\n" )
        # Create empty list to store accuracy measure for each of the 5 folds
        list_of_performances = []
        for index_lists in range(0, len(test_folds_row_numbers)):
            tuneOutputFile.write( "  When fold "+str(index_lists)+" is the test set:\n" )
            # Obtain row numbers for test fold
            temp_df_row_list = test_folds_row_numbers[index_lists]
            # Obtain test data frame using row numbers for test fold
            temp_test_data_frame = temp_df.iloc[temp_df_row_list,]
            temp_test_data_frame = temp_test_data_frame.reset_index( drop=True )
            # Obtain train data frame by dropping row numbers for test fold
            temp_train_data_frame = temp_df.drop( temp_df.index[temp_df_row_list] )
            temp_train_data_frame = temp_train_data_frame.reset_index( drop=True )

            # Perform the knn algorithm
            this_fold_performance = knn_algorithm_for_tuning( temp_test_data_frame, 
              temp_train_data_frame, k_index )
            # Store the accuracy measure
            list_of_performances.append( this_fold_performance )
            # Express the accuracy as a percentage
            this_fold_percent_accuracy = this_fold_performance * 100

            tuneOutputFile.write( "\tClassification accuracy: " + 
              str(round( this_fold_percent_accuracy, 2 )) + "%\n" )
        # Calculate the average of the 5 accuracy performances   
        average_of_5_performances = np.mean( list_of_performances )
        performance_for_this_k.append( average_of_5_performances )
        # Express the average accuracy as a percentage
        percent_accuracy = average_of_5_performances * 100
        # Calculate the standard deviation for the 5 accuracy measures
        stdev_accuracy = np.std( list_of_performances )
        stdev_accuracy_as_percent = stdev_accuracy * 100
        tuneOutputFile.write( "\nAverage classification accuracy: " + 
          str(round( percent_accuracy, 2 )) + "% for k equals " + str(k_index) + "\n" )
        tuneOutputFile.write( "Standard deviation: " + 
          str(round( stdev_accuracy_as_percent, 2 )) + "%\n" )
    # Write the raw average accuracy measures for each k
    tuneOutputFile.write( "\nSummary of raw average accuracy over all k: " + 
      str(performance_for_this_k) + "\n" )
    
    # Determine k value associated with highest accuracy
    max_accuracy = 0
    optimal_k_for_knn = 0
    for accuracy_index in range( 0, len(performance_for_this_k) ):
        if performance_for_this_k[ accuracy_index ] > max_accuracy:
            max_accuracy = performance_for_this_k[ accuracy_index ]
            optimal_k_for_knn = list_of_k_values[ accuracy_index ]
    tuneOutputFile.write( "\nHighest accuracy over all k: " + str(max_accuracy) + "\n" )
    tuneOutputFile.write( "Optimal value of k parameter: " + str(optimal_k_for_knn)+"\n")

# Close this output file
tuneOutputFile.close()


####################################################
# Perform Condensed KNN, tuning for the k parameter
####################################################

# Write condensed nearest neighbors results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_Condensed_KNN_Classifier_Tune_K", 'w') as condensedOutputFile:
    condensedOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    condensedOutputFile.write( "\nType of problem: Classification\n" )
    
    condensedOutputFile.write( "\n****************************\n" )
    condensedOutputFile.write( "\nCondensed Nearest Neighbors\n" )
    condensedOutputFile.write( "\n  Tuning the parameter k\n" )
    condensedOutputFile.write( "\n****************************\n" )

    # Create empty list to store accuracy measure for each k
    condensed_performance_for_this_k = []
    # Create an empty list to track the values of k
    list_of_k_values = []
    for k_index in range(1, 13):
        # Track the value of k for this run
        list_of_k_values.append( k_index )
        # Write this value of k to the output file
        condensedOutputFile.write( "\nFor k = " + str(k_index) + ":\n" )
        # Create empty list to store the reduction in size between original
        # data set and condensed data set
        list_of_reduction_in_data_size = []
        # Create empty list to store accuracy measure for each of the 5 folds
        condensed_list_of_performances = []
        for index_lists in range( 0, len(test_folds_row_numbers) ):
            condensedOutputFile.write( "  When fold " + str(index_lists) + 
              " is the test set:\n" )
            # Obtain row numbers for test fold
            temp_df_row_list = test_folds_row_numbers[index_lists]
            # Obtain test data frame using row numbers for test fold
            temp_test_data_frame = temp_df.iloc[temp_df_row_list,]
            temp_test_data_frame = temp_test_data_frame.reset_index( drop=True )
            # Obtain train data frame by dropping row numbers for test fold
            temp_train_data_frame = temp_df.drop( temp_df.index[temp_df_row_list] )
            temp_train_data_frame = temp_train_data_frame.reset_index( drop=True )
            
            # Call the condensed nearest neighbors method in an attempt to reduce the data size
            condensed_train_data_frame, instance_reduction = 
              condensed_nearest_neighbor( temp_train_data_frame )
            list_of_reduction_in_data_size.append( instance_reduction )
            
            # Call the knn algorithm to determine the nearest neighbors
            this_fold_performance = knn_algorithm_for_tuning( temp_test_data_frame, 
              condensed_train_data_frame, k_index )
            condensed_list_of_performances.append( this_fold_performance )
            # Express the accuracy measure as a percentage
            this_fold_percent_accuracy = this_fold_performance * 100

            condensedOutputFile.write( "\tClassification accuracy: " + 
              str(round( this_fold_percent_accuracy, 2 )) + "%\n" )
            condensedOutputFile.write( "\tPercent reduction in number of instances: " + 
              str(round( this_fold_percent_accuracy, 2 )) + "%\n" )
        # Calculate the average reduction in the size of the data set
        average_of_5_instance_reductions = np.mean(list_of_reduction_in_data_size)    
        percent_reduction_of_instances = average_of_5_instance_reductions * 100
        # Calculate the standard deviation of the reduction in the size of the data set
        stdev_reduction_of_instances = np.std( list_of_reduction_in_data_size )
        stdev_reduction_as_percent = stdev_reduction_of_instances * 100 
        
        condensedOutputFile.write( "\nAverage reduction in number of instances: " + 
          str(round( percent_reduction_of_instances, 2 )) + "%\n" )
        condensedOutputFile.write( "Standard deviation: " + 
          str(round( stdev_reduction_as_percent, 2 )) + "%\n" )
        
        # Calculate the average accuracy of the classifier
        average_of_5_performances = np.mean( condensed_list_of_performances )
        condensed_performance_for_this_k.append( average_of_5_performances )
        percent_accuracy = average_of_5_performances * 100
        # Calculate the standard deviation of the average accuracy
        stdev_accuracy = np.std( condensed_list_of_performances )
        stdev_accuracy_as_percent = stdev_accuracy * 100
        
        condensedOutputFile.write( "\nAverage classification accuracy: " + 
          str(round( percent_accuracy, 2 )) + "%\n" )
        condensedOutputFile.write( "Standard deviation: " + 
          str(round( stdev_accuracy_as_percent, 2 )) + "%\n" )
    # Write the raw average accuracy measures for each k
    condensedOutputFile.write( "\nSummary of raw average accuracy over all k: " + 
      str(condensed_performance_for_this_k) + "\n" ) 
    
    # Determine k value associated with highest accuracy
    max_accuracy = 0
    condensed_optimal_k = 0
    for accuracy_index in range( 0, len(condensed_performance_for_this_k) ):
        if condensed_performance_for_this_k[ accuracy_index ] > max_accuracy:
            max_accuracy = condensed_performance_for_this_k[ accuracy_index ]
            condensed_optimal_k = list_of_k_values[ accuracy_index ]
    condensedOutputFile.write( "\nHighest accuracy over all k: " + 
      str(max_accuracy) + "\n" )
    condensedOutputFile.write( "Optimal value of k parameter: " + 
      str(condensed_optimal_k) + "\n" )  

# Close this output file
condensedOutputFile.close()


################################################
# Perform Edited KNN, tuning for the k parameter
################################################

# Write condensed nearest neighbors results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_Edited_KNN_Classifier_Tune_K", 'w') as editedOutputFile:
    editedOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    editedOutputFile.write( "\nType of problem: Classification\n" )
    
    editedOutputFile.write( "\n************************\n" )
    editedOutputFile.write( "\nEdited Nearest Neighbors\n" )
    editedOutputFile.write( "\n  Tuning the parameter k\n" )
    editedOutputFile.write( "\n************************\n" )


    # Create empty list to store accuracy measure for each k
    edited_performance_for_this_k = []
    for k_index in range(1, 13):
        editedOutputFile.write( "\nFor k = " + str(k_index) + ":\n" )
        # Create empty list to store the reduction in size between original
        # data set and edited data set
        list_of_reduction_in_data_size = []
        # Create empty list to store accuracy measure for each of the 5 folds
        edited_list_of_performances = []
        for index_lists in range( 0, len(test_folds_row_numbers) ):
            editedOutputFile.write( "  When fold " + str(index_lists) + 
              " is the test set:\n" )
            # Obtain row numbers for test fold
            temp_df_row_list = test_folds_row_numbers[index_lists]
            # Obtain test data frame using row numbers for test fold
            temp_test_data_frame = temp_df.iloc[temp_df_row_list,]
            temp_test_data_frame = temp_test_data_frame.reset_index( drop=True )
            # Obtain train data frame by dropping row numbers for test fold
            temp_train_data_frame = temp_df.drop( temp_df.index[temp_df_row_list] )
            temp_train_data_frame = temp_train_data_frame.reset_index( drop=True )
            
            # Call the edited nearest neighbors method in an attempt to
            # reduce the data set size 
            edited_train_data_frame, instance_reduction = 
              edited_nearest_neighbor( temp_train_data_frame )
            list_of_reduction_in_data_size.append( instance_reduction )

            # Call the knn algorithm; accuracy of the classifier will be returned 
            this_fold_performance = knn_algorithm_for_tuning( temp_test_data_frame, 
              edited_train_data_frame, k_index )
            edited_list_of_performances.append( this_fold_performance )
            this_fold_percent_accuracy = this_fold_performance * 100

            editedOutputFile.write( "\tClassification accuracy: " + 
              str(round( this_fold_percent_accuracy, 2 )) + "%\n" )
            editedOutputFile.write( "\tPercent reduction in number of instances: " + 
              str(round( this_fold_percent_accuracy, 2 )) + "%\n" )
        # Calculate the average reduction in size of the data set
        average_of_5_instance_reductions = np.mean( list_of_reduction_in_data_size )
        percent_reduction_of_instances = average_of_5_instance_reductions * 100
        # Calculate the standard deviation of the data size reductions
        stdev_reduction_of_instances = np.std( list_of_reduction_in_data_size )
        stdev_reduction_as_percent = stdev_reduction_of_instances * 100
        
        editedOutputFile.write( "\nAverage reduction in number of instances: " + 
          str(round( percent_reduction_of_instances, 2 )) + "%\n" )
        editedOutputFile.write( "Standard deviation: " + 
          str(round( stdev_reduction_as_percent, 2 )) + "%\n" )

        # Calculate the average accuracy of the classifier
        average_of_5_performances = np.mean( edited_list_of_performances )
        edited_performance_for_this_k.append( average_of_5_performances )
        percent_accuracy = average_of_5_performances * 100
        # Calculate the standard deviation of the accuracy measures
        stdev_accuracy = np.std( edited_list_of_performances )
        stdev_accuracy_as_percent = stdev_accuracy * 100
        
        editedOutputFile.write( "\nAverage classification accuracy: " + 
          str(round( percent_accuracy, 2 )) + "%\n" )
        editedOutputFile.write( "Standard deviation: " + 
          str(round( stdev_accuracy_as_percent, 2 )) + "%\n" )
    # Write the raw average accuracy measures for each k
    editedOutputFile.write( "\nSummary of raw average accuracy over all k: " + 
      str(edited_performance_for_this_k) + "\n" ) 
    
    # Determine k value associated with highest accuracy
    max_accuracy = 0
    edited_optimal_k = 0
    for accuracy_index in range( 0, len(edited_performance_for_this_k) ):
        if edited_performance_for_this_k[ accuracy_index ] > max_accuracy:
            max_accuracy = edited_performance_for_this_k[ accuracy_index ]
            edited_optimal_k = list_of_k_values[ accuracy_index ]
    editedOutputFile.write( "\nHighest accuracy over all k: " + str(max_accuracy) + "\n" )
    editedOutputFile.write( "Optimal value of k parameter: " + str(edited_optimal_k) + "\n" )  

# Close this output file
editedOutputFile.close()


####################################################
# Perform KNN using optimal k parameter from tuning
####################################################

# Split data set into 5 folds 
optimal_k_test_folds_row_numbers = split_for_cross_validation(temp_df, 5)

# Prepare output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_KNN_Classifier_Optimal_K", 'w') as tunedKOutputFile:
    tunedKOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    tunedKOutputFile.write( "\nType of problem: Classification\n" )
    
    tunedKOutputFile.write( "\n************************\n" )
    tunedKOutputFile.write( "\nKNN with optimal k = " + str( optimal_k_for_knn ) + "\n" )
    tunedKOutputFile.write( "\n************************\n" )

    # Create empty list to store accuracy measure for each of the 5 folds
    optimal_k_list_of_performances = []
    for index_lists in range(0, len(optimal_k_test_folds_row_numbers)):
        tunedKOutputFile.write( "  When fold " + str(index_lists) + " is the test set:\n")
        # Obtain row numbers for test fold
        temp_df_row_list = optimal_k_test_folds_row_numbers[index_lists]
        # Obtain test data frame using row numbers for test fold
        temp_test_data_frame = temp_df.iloc[temp_df_row_list,]
        temp_test_data_frame = temp_test_data_frame.reset_index( drop=True )
        # Obtain train data frame by dropping row numbers for test fold
        temp_train_data_frame = temp_df.drop( temp_df.index[temp_df_row_list] )
        temp_train_data_frame = temp_train_data_frame.reset_index( drop=True )

        # Perform the knn algorithm with optimal k parameter
        this_fold_performance = knn_algorithm_for_tuning( temp_test_data_frame, 
          temp_train_data_frame, optimal_k_for_knn )
        optimal_k_list_of_performances.append( this_fold_performance )
        this_fold_percent_accuracy = this_fold_performance * 100            
        tunedKOutputFile.write( "\tClassification accuracy: " + 
          str(round( this_fold_percent_accuracy, 2 )) + "%\n" )
    # Calculate average accuracy from the five folds
    average_of_5_performances = np.mean( optimal_k_list_of_performances )
    percent_accuracy = average_of_5_performances * 100
    # Calculate standard deviation for the average accuracy measure
    stdev_accuracy = np.std( optimal_k_list_of_performances )
    stdev_accuracy_as_percent = stdev_accuracy * 100

    tunedKOutputFile.write( "\nAverage classification accuracy: " + 
      str(round( percent_accuracy, 2 )) + "% for k equals " + 
        str(optimal_k_for_knn) + "\n" )
    tunedKOutputFile.write( "Standard deviation: " + 
      str(round( stdev_accuracy_as_percent, 2 )) + "%\n" )

# Close this output file
tunedKOutputFile.close()


##############################################################
# Perform Condensed KNN using optimal k parameter from tuning
##############################################################

# Write condensed nearest neighbors results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_Condensed_KNN_Classifier_Optimal_K", 'w') as optimalKCondensedOutputFile:
    optimalKCondensedOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    optimalKCondensedOutputFile.write( "\nType of problem: Classification\n" )
    
    optimalKCondensedOutputFile.write( "\n********************************\n" )
    optimalKCondensedOutputFile.write( "\nCondensed KNN with optimal k = " + 
      str( condensed_optimal_k ) + "\n" )
    optimalKCondensedOutputFile.write( "\n********************************\n" )

    # Create empty list to store the reduction in size between original
    # data set and condensed data set
    list_of_reduction_in_data_size_optimal_k = []
    # Create empty list to store accuracy measure for each of the 5 folds
    condensed_list_of_performances_optimal_k = []
    for index_lists in range( 0, len(optimal_k_test_folds_row_numbers) ):
        optimalKCondensedOutputFile.write( "  When fold " + str(index_lists) + 
          " is the test set:\n" )
        # Obtain row numbers for test fold
        temp_df_row_list = optimal_k_test_folds_row_numbers[index_lists]
        # Obtain test data frame using row numbers for test fold
        temp_test_data_frame = temp_df.iloc[temp_df_row_list,]
        temp_test_data_frame = temp_test_data_frame.reset_index( drop=True )
        # Obtain train data frame by dropping row numbers for test fold
        temp_train_data_frame = temp_df.drop( temp_df.index[temp_df_row_list] )
        temp_train_data_frame = temp_train_data_frame.reset_index( drop=True )

        condensed_train_data_frame, instance_reduction = 
          condensed_nearest_neighbor( temp_train_data_frame )
        list_of_reduction_in_data_size_optimal_k.append( instance_reduction )
        percent_reduction_in_data_set_size = instance_reduction * 100

        this_fold_performance = knn_algorithm_for_tuning( temp_test_data_frame, 
          condensed_train_data_frame, condensed_optimal_k )
        condensed_list_of_performances_optimal_k.append( this_fold_performance )
        this_fold_percent_accuracy = this_fold_performance * 100

        optimalKCondensedOutputFile.write( "\tClassification accuracy: " + 
          str(round( this_fold_percent_accuracy, 2 )) + "%\n" )
        optimalKCondensedOutputFile.write( "\tPercent reduction in number of instances: " + 
          str(round( percent_reduction_in_data_set_size, 2 )) + "%\n" )
    
    # Calculate the average reduction in the size of the data set     
    average_of_5_instance_reductions = np.mean(list_of_reduction_in_data_size_optimal_k)    
    percent_reduction_of_instances = average_of_5_instance_reductions * 100
    # Calculate the standard deviation for the reduction in the data set size
    stdev_reduction_of_instances = np.std( list_of_reduction_in_data_size_optimal_k )
    stdev_reduction_as_percent = stdev_reduction_of_instances * 100 
        
    optimalKCondensedOutputFile.write( "\nAverage reduction in number of instances: " + 
      str(round( percent_reduction_of_instances, 2 )) + "%\n" )
    optimalKCondensedOutputFile.write( "Standard deviation: " + 
      str(round( stdev_reduction_as_percent, 2 )) + "%\n" )
     
    # Calculate the average accuracy of the classifier
    average_of_5_performances = np.mean( condensed_list_of_performances_optimal_k )
    percent_accuracy = average_of_5_performances * 100
    # Calculate the standard deviation of the accuracy measures
    stdev_accuracy = np.std( condensed_list_of_performances )
    stdev_accuracy_as_percent = stdev_accuracy * 100
        
    optimalKCondensedOutputFile.write( "\nAverage classification accuracy: " + 
      str(round( percent_accuracy, 2 )) + "%\n" )
    optimalKCondensedOutputFile.write( "Standard deviation: " + 
      str(round( stdev_accuracy_as_percent, 2 )) + "%\n" )

# Close this output file
optimalKCondensedOutputFile.close()


###########################################################
# Perform Edited KNN using optimal k parameter from tuning
###########################################################

# Write edited nearest neighbors results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_Edited_KNN_Classifier_Optimal_K", 'w') as optimalKEditedOutputFile:
    optimalKEditedOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    optimalKEditedOutputFile.write( "\nType of problem: Classification\n" )
    
    optimalKEditedOutputFile.write( "\n************************\n" )
    optimalKEditedOutputFile.write( "\nEdited KNN with optimal k = " + 
      str( edited_optimal_k ) + "\n" )
    optimalKEditedOutputFile.write( "\n************************\n" )

    # Create empty list to store the reduction in size between original
    # data set and edited data set
    list_of_reduction_in_data_size = []
    # Create empty list to store accuracy measure for each of the 5 folds
    edited_list_of_performances = []
    for index_lists in range( 0, len(test_folds_row_numbers) ):
        optimalKEditedOutputFile.write( "  When fold " + str(index_lists) + 
          " is the test set:\n" )
        # Obtain row numbers for test fold
        temp_df_row_list = test_folds_row_numbers[index_lists]
        # Obtain test data frame using row numbers for test fold
        temp_test_data_frame = temp_df.iloc[temp_df_row_list,]
        temp_test_data_frame = temp_test_data_frame.reset_index( drop=True )
        # Obtain train data frame by dropping row numbers for test fold
        temp_train_data_frame = temp_df.drop( temp_df.index[temp_df_row_list] )
        temp_train_data_frame = temp_train_data_frame.reset_index( drop=True )
        
        # Call the edited nearest neighbor method in an attempt to reduce the data set size
        edited_train_data_frame, instance_reduction = 
          edited_nearest_neighbor( temp_train_data_frame )
        list_of_reduction_in_data_size.append( instance_reduction )
        percent_reduction_in_data_set_size = instance_reduction * 100
        
        # Call the knn method; will return the accuracy of the classifier
        this_fold_performance = knn_algorithm_for_tuning( temp_test_data_frame, 
          edited_train_data_frame, edited_optimal_k )
        edited_list_of_performances.append( this_fold_performance )
        this_fold_percent_accuracy = this_fold_performance * 100

        optimalKEditedOutputFile.write( "\tClassification accuracy: " + 
          str(round( this_fold_percent_accuracy, 2 )) + "%\n" )
        optimalKEditedOutputFile.write( "\tPercent reduction in number of instances: " + 
          str(round( percent_reduction_in_data_set_size, 2 )) + "%\n" )
    # Calculate the average reduction in the size of the data set
    average_of_5_instance_reductions = np.mean( list_of_reduction_in_data_size )
    percent_reduction_of_instances = average_of_5_instance_reductions * 100
    # Calculate the standard deviation of the reduction in data set size
    stdev_reduction_of_instances = np.std( list_of_reduction_in_data_size )
    stdev_reduction_as_percent = stdev_reduction_of_instances * 100
        
    optimalKEditedOutputFile.write( "\nAverage reduction in number of instances: " + 
      str(round( percent_reduction_of_instances, 2 )) + "%\n" )
    optimalKEditedOutputFile.write( "Standard deviation: " + 
      str(round( stdev_reduction_as_percent, 2 )) + "%\n" )
    
    # Calculate the average accuracy of the classifier
    average_of_5_performances = np.mean( edited_list_of_performances )
    edited_performance_for_this_k.append( average_of_5_performances )
    percent_accuracy = average_of_5_performances * 100
    # Calculate the standard deviation of the accuracy measures
    stdev_accuracy = np.std( edited_list_of_performances )
    stdev_accuracy_as_percent = stdev_accuracy * 100
        
    optimalKEditedOutputFile.write( "\nAverage classification accuracy: " + 
      str(round( percent_accuracy, 2 )) + "%\n" )
    optimalKEditedOutputFile.write( "Standard deviation: " + 
      str(round( stdev_accuracy_as_percent, 2 )) + "%\n" )
    
# Close this output file
optimalKEditedOutputFile.close()

