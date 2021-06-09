##
 #  This program uses the k nearest neighbors (KNN) algorithm to train and test 
 #  a machine learning regressor, one that applies kernel smoothing and  
 #  one that does not. Five-fold cross-validation is performed and a
 #  KNN regressor is tuned to determine the optimal value of k between k = 1 to
 #  k = 12. Summary statistics are output to a file. The data set is again split 
 #  for five-fold cross validation and the optimal value of k is used to train
 #  and test a regressor. The summary statistics are output to a separate file.
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

# This method implements the KNN algorithm by calculating Euclidean distances,
# identifying the k nearest neighbors, calculating a weighted average of their
# target values through the use of a Gaussian kernel smoother and using that 
# value as the predicted value for test instances. Calculates MSE to evaluate 
# the predictions.
def knn_algorithm_for_tuning( temp_test_data_frame, temp_train_data_frame, k_index ):    
    # Determine the number of rows in the test data frame
    number_test_instances = temp_test_data_frame.shape[0]
    # Determine the number of rows in the train data frame
    number_train_instances = temp_train_data_frame.shape[0]
    # Create temporary numpy array to store results
    temp_df_for_test_results = np.zeros( shape=( number_test_instances, 5 ) )
    # Create dataframe only consisting of attributes for test data
    temp_test_df_attributes_only = temp_test_data_frame.drop(["PRP"], axis=1)
    # Create empty list to hold error value for each instance
    mse_for_data_set = []
    number_train_columns = temp_test_df_attributes_only.shape[1]

    for test_row_index in range( 0, number_test_instances ):
        # Create temporary numpy array to store euclidean distances
        temp_df_for_distances = np.zeros( shape=( number_train_instances, 2 ) )
        # Create dataframe only consisting of attributes for train data
        temp_train_df_attributes_only = temp_train_data_frame.drop(["PRP"], axis=1)
        temp_train_target_column = temp_train_data_frame["PRP"]
        # Calculate Euclidean distance between test instance and training instances
        for train_row_index in range( 0, number_train_instances ):
            squared_distance = 0
            for feature_index in range( 0, number_train_columns ):
                squared_distance = squared_distance + 
                  ( temp_test_df_attributes_only.iloc[test_row_index, feature_index] - 
                    temp_train_df_attributes_only.iloc[train_row_index, feature_index] ) ** 2               
            euclidean_distance = np.sqrt( squared_distance )
            
            temp_df_for_distances[train_row_index, 0] = euclidean_distance
            temp_df_for_distances[train_row_index, 1] = train_row_index
            
        sorted_temp_df_for_distances = 
          temp_df_for_distances[temp_df_for_distances[:,0].argsort()]
        k_nearest_neighbors = sorted_temp_df_for_distances[0:k_index,]
        # Grab the target values for each of the k neighbors
        target_value = []
        for target_value_index in range(0, k_index):            
            target_row_number = int(k_nearest_neighbors[target_value_index, 1])
            target_value.append(temp_train_data_frame.iloc[target_row_number, 0])
        
        # Grab the distances for each of the k neighbors
        nearest_distances = []
        for nearest_distance_index in range(0, k_index):
            euc_distance = k_nearest_neighbors[nearest_distance_index, 0]
            nearest_distances.append(euc_distance) 
            
        # Locate the largest distance for use in Gaussian calculation
        distance_for_h = max(nearest_distances)
        
        # Create empty vector to hold the weighted distances
        weighted_distances = []
        # Create empty vector to hold all of the weights calculated 
        # using Gaussian function
        all_weights = []
        
        # Calculate weights
        for distance_index in range( 0, len(nearest_distances) ):
            if distance_for_h != 0:
                weight = 
                  math.exp(-((nearest_distances[distance_index] ** 2) / (distance_for_h*2)))
                all_weights.append(weight)
                weighted_target_value = target_value[distance_index] * weight           
                weighted_distances.append(weighted_target_value)
            # If the bin size (distance_for_h) is equal to 0, don't calculate Gaussian
            else:
                weight = 1
                all_weights.append(weight)
                weighted_target_value = target_value[distance_index] * weight           
                weighted_distances.append(weighted_target_value)
                        
        average_of_weighted_distances = sum(weighted_distances) / sum(all_weights) 
        
        # Compare predicted to actual
        test_instance_value = temp_test_data_frame.iloc[test_row_index, 0]
        
        # Calculate error for this instance
        # Subtract the predicted value from the correct value
        difference_between_values = average_of_weighted_distances - test_instance_value
        # Square this distance
        squared_difference = difference_between_values ** 2
        # Append to the running list of errors
        mse_for_data_set.append(squared_difference)
    total_mse_for_data_set = sum( mse_for_data_set ) / len( mse_for_data_set )
    return( total_mse_for_data_set )

# This method implements the KNN algorithm by calculating Euclidean distances,
# identifying the k nearest neighbors, and averaging their target values to arrive
# at the predicted value for test instances. This method does not use a kernel 
# smoother. Calculates MSE to evaluate the predictions.
def knn_algorithm_no_smoothing( temp_test_data_frame, temp_train_data_frame, k_index ):    
    # Determine the number of rows in the test data frame
    number_test_instances = temp_test_data_frame.shape[0]
    # Determine the number of rows in the train data frame
    number_train_instances = temp_train_data_frame.shape[0]
    # Create temporary numpy array to store results
    temp_df_for_test_results = np.zeros( shape=( number_test_instances, 5 ) )
    # Create dataframe only consisting of attributes for test data
    temp_test_df_attributes_only = temp_test_data_frame.drop(["PRP"], axis=1)
    # Create empty list to hold error value for each instance
    mse_for_data_set = []
    number_train_columns = temp_test_df_attributes_only.shape[1]

    for test_row_index in range( 0, number_test_instances ):
        # Create temporary numpy array to store euclidean distances
        temp_df_for_distances = np.zeros( shape=( number_train_instances, 2 ) )
        # Create dataframe only consisting of attributes for train data
        temp_train_df_attributes_only = temp_train_data_frame.drop(["PRP"], axis=1)
        temp_train_target_column = temp_train_data_frame["PRP"]
        # Calculate Euclidean distance between test instance and training instances
        for train_row_index in range( 0, number_train_instances ):
            squared_distance = 0
            for feature_index in range( 0, number_train_columns ):
                squared_distance = squared_distance + 
                  ( temp_test_df_attributes_only.iloc[test_row_index, feature_index] - 
                    temp_train_df_attributes_only.iloc[train_row_index, feature_index] ) ** 2               
            euclidean_distance = np.sqrt( squared_distance )
            
            temp_df_for_distances[train_row_index, 0] = euclidean_distance
            temp_df_for_distances[train_row_index, 1] = train_row_index
            
        sorted_temp_df_for_distances = 
          temp_df_for_distances[temp_df_for_distances[:,0].argsort()]
        k_nearest_neighbors = sorted_temp_df_for_distances[0:k_index,]
        # Grab the target values for each of the k neighbors
        target_value = []
        for target_value_index in range(0, k_index):            
            target_row_number = int(k_nearest_neighbors[target_value_index, 1])
            target_value.append(temp_train_data_frame.iloc[target_row_number, 0])
        average_target_value = np.mean( target_value )
        
        # Compare predicted to actual
        test_instance_value = temp_test_data_frame.iloc[test_row_index, 0]
        
        # Calculate error for this instance
        # Subtract the predicted value from the correct value
        difference_between_values = average_target_value - test_instance_value
        # Square this distance
        squared_difference = difference_between_values ** 2
        # Append to the running list of errors
        mse_for_data_set.append(squared_difference)
    total_mse_for_data_set = sum( mse_for_data_set ) / len( mse_for_data_set )
    return( total_mse_for_data_set )


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

# For computer hardware data set:
temp_df = temp_df.drop("ERP", axis = 1)

###############################################################
# Perform KNN with kernel smoother, tuning for the k parameter
###############################################################

# Write results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_KNN_Regressor_Tune_K", 'w') as tuneOutputFile:
    tuneOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    tuneOutputFile.write( "\nType of problem: Regression\n" )

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
            tuneOutputFile.write( "  When fold " + str(index_lists) + 
              " is the test set:\n" )
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
            mean_squared_error = this_fold_performance

            tuneOutputFile.write( "\tMean squared error for this fold: " + 
              str(round( mean_squared_error, 4 )) + "\n" )
        
        
        # Calculate the average of the 5 mean squared errors   
        average_of_5_performances = np.mean( list_of_performances )
        performance_for_this_k.append( average_of_5_performances )
        average_mean_squared_error_for_this_k = average_of_5_performances
        # Calculate the standard deviation for the 5 accuracy measures
        stdev_accuracy = np.std( list_of_performances )
        tuneOutputFile.write( "\nAverage mean squared error: " + 
          str(round( average_mean_squared_error_for_this_k, 2 )) + 
          " for k equals " + str(k_index) + "\n" )
        tuneOutputFile.write( "Standard deviation: " + 
          str(round( stdev_accuracy, 2 )) + "\n" )
        
    # Write the raw average accuracy measures for each k
    tuneOutputFile.write( "\nSummary of raw MSE over all k: " + 
      str(performance_for_this_k) + "\n" )
    
    
    # Determine k value associated with lowest MSE
    max_accuracy = 10000
    optimal_k_for_knn = 0
    for accuracy_index in range( 0, len(performance_for_this_k) ):
        if performance_for_this_k[ accuracy_index ] < max_accuracy:
            max_accuracy = performance_for_this_k[ accuracy_index ]
            optimal_k_for_knn = list_of_k_values[ accuracy_index ]
    tuneOutputFile.write( "\nLowest MSE over all k: " + str(max_accuracy) + "\n" )
    tuneOutputFile.write( "Optimal value of k parameter: " + 
      str(optimal_k_for_knn) + "\n" )

# Close this output file
tuneOutputFile.close()


##################################################################
# Perform KNN without kernel smoother, tuning for the k parameter
##################################################################

# Write results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_KNN_Regressor_Tune_K_no_smoothing", 'w') as tuneNoSmoothingOutputFile:
    tuneNoSmoothingOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    tuneNoSmoothingOutputFile.write( "\nType of problem: Regression\n" )

    # Split data set into 5 folds 
    test_folds_row_numbers = split_for_cross_validation(temp_df, 5)

    # Tune the nearest neighbors parameter k
    tuneNoSmoothingOutputFile.write( "\n************************\n" )
    tuneNoSmoothingOutputFile.write( "\nTuning the parameter k\n" )
    tuneNoSmoothingOutputFile.write( "\n************************\n" )


    # Create an empty list to store accuracy measure for each k
    performance_for_this_k = []
    # Create an empty list to track the values of k
    list_of_k_values = []

    for k_index in range(1, 13):
        # Store the value of k for this run
        list_of_k_values.append( k_index )
        # Output value of k for this run
        tuneNoSmoothingOutputFile.write( "\nFor k = " + str(k_index) + ":\n" )
        # Create empty list to store accuracy measure for each of the 5 folds
        list_of_performances = []
        for index_lists in range(0, len(test_folds_row_numbers)):
            tuneNoSmoothingOutputFile.write( "  When fold " + 
              str(index_lists) + " is the test set:\n" )
            # Obtain row numbers for test fold
            temp_df_row_list = test_folds_row_numbers[index_lists]
            # Obtain test data frame using row numbers for test fold
            temp_test_data_frame = temp_df.iloc[temp_df_row_list,]
            temp_test_data_frame = temp_test_data_frame.reset_index( drop=True )
            # Obtain train data frame by dropping row numbers for test fold
            temp_train_data_frame = temp_df.drop( temp_df.index[temp_df_row_list] )
            temp_train_data_frame = temp_train_data_frame.reset_index( drop=True )

            # Perform the knn algorithm
            this_fold_performance = knn_algorithm_no_smoothing( temp_test_data_frame, 
              temp_train_data_frame, k_index )
            # Store the accuracy measure
            list_of_performances.append( this_fold_performance )
            mean_squared_error = this_fold_performance

            tuneNoSmoothingOutputFile.write( "\tMean squared error for this fold: " + 
              str(round( mean_squared_error, 4 )) + "\n" )
        
        
        # Calculate the average of the 5 mean squared errors   
        average_of_5_performances = np.mean( list_of_performances )
        performance_for_this_k.append( average_of_5_performances )
        average_mean_squared_error_for_this_k = average_of_5_performances
        # Calculate the standard deviation for the 5 accuracy measures
        stdev_accuracy = np.std( list_of_performances )
        tuneNoSmoothingOutputFile.write( "\nAverage mean squared error: " + 
          str(round( average_mean_squared_error_for_this_k, 2 )) + " for k equals " + 
          str(k_index) + "\n" )
        tuneNoSmoothingOutputFile.write( "Standard deviation: " + 
          str(round( stdev_accuracy, 2 )) + "\n" )
        
    # Write the raw average accuracy measures for each k
    tuneNoSmoothingOutputFile.write( "\nSummary of raw MSE over all k: " + 
      str(performance_for_this_k) + "\n" )
    
    
    # Determine k value associated with lowest MSE
    max_accuracy = 10000
    optimal_k_for_knn_no_smooth = 0
    for accuracy_index in range( 0, len(performance_for_this_k) ):
        if performance_for_this_k[ accuracy_index ] < max_accuracy:
            max_accuracy = performance_for_this_k[ accuracy_index ]
            optimal_k_for_knn_no_smooth = list_of_k_values[ accuracy_index ]
    tuneNoSmoothingOutputFile.write( "\nLowest MSE over all k: " + 
      str(max_accuracy) + "\n" )
    tuneNoSmoothingOutputFile.write( "Optimal value of k parameter: " + 
      str(optimal_k_for_knn_no_smooth) + "\n" )

# Close this output file
tuneNoSmoothingOutputFile.close()

    
##########################################################################
# Perform KNN with kernel smoothing using optimal k parameter from tuning
##########################################################################

# Split data set into 5 folds 
optimal_k_test_folds_row_numbers = split_for_cross_validation(temp_df, 5)

# Prepare output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_KNN_Regressor_Optimal_K", 'w') as tunedKOutputFile:
    tunedKOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    tunedKOutputFile.write( "\nType of problem: Regression\n" )
    
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
        mean_squared_error = this_fold_performance           
        tunedKOutputFile.write( "\tMean squared error for this fold: " + 
          str(round( mean_squared_error, 4 )) + "\n" )
    # Calculate average accuracy from the five folds
    average_of_5_performances = np.mean( optimal_k_list_of_performances )
    #percent_accuracy = average_of_5_performances * 100
    # Calculate standard deviation for the average accuracy measure
    stdev_accuracy = np.std( optimal_k_list_of_performances )

    tunedKOutputFile.write( "\nAverage mean squared error: " + 
      str(round( average_of_5_performances, 4 )) + " for k equals " + 
      str(optimal_k_for_knn) + "\n" )
    tunedKOutputFile.write( "Standard deviation: " + 
      str(round( stdev_accuracy, 4 )) + "\n" )

# Close this output file
tunedKOutputFile.close()
  
#############################################################################
# Perform KNN with no kernel smoothing using optimal k parameter from tuning
#############################################################################

# Prepare output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_2/Outputs/" + 
  output_file_name_final + "_output_KNN_Regressor_Optimal_K_no_smoothing", 'w') as tunedKnoSmoothingOutputFile:
    tunedKnoSmoothingOutputFile.write( "\nData set: " + output_file_name_final + "\n" )
    tunedKnoSmoothingOutputFile.write( "\nType of problem: Regression\n" )
    
    tunedKnoSmoothingOutputFile.write( "\n************************\n" )
    tunedKnoSmoothingOutputFile.write( "\nKNN with optimal k = " + 
      str( optimal_k_for_knn_no_smooth ) + "\n" )
    tunedKnoSmoothingOutputFile.write( "\n************************\n" )

    # Create empty list to store accuracy measure for each of the 5 folds
    optimal_k_list_of_performances = []
    for index_lists in range(0, len(optimal_k_test_folds_row_numbers)):
        tunedKnoSmoothingOutputFile.write( "  When fold " + str(index_lists) + 
          " is the test set:\n" )
        # Obtain row numbers for test fold
        temp_df_row_list = optimal_k_test_folds_row_numbers[index_lists]
        # Obtain test data frame using row numbers for test fold
        temp_test_data_frame = temp_df.iloc[temp_df_row_list,]
        temp_test_data_frame = temp_test_data_frame.reset_index( drop=True )
        # Obtain train data frame by dropping row numbers for test fold
        temp_train_data_frame = temp_df.drop( temp_df.index[temp_df_row_list] )
        temp_train_data_frame = temp_train_data_frame.reset_index( drop=True )

        # Perform the knn algorithm with optimal k parameter
        this_fold_performance = knn_algorithm_no_smoothing( temp_test_data_frame, 
          temp_train_data_frame, optimal_k_for_knn_no_smooth )
        optimal_k_list_of_performances.append( this_fold_performance )
        mean_squared_error = this_fold_performance           
        tunedKnoSmoothingOutputFile.write( "\tMean squared error for this fold: " + 
          str(round( mean_squared_error, 4 )) + "\n" )
    # Calculate average accuracy from the five folds
    average_of_5_performances = np.mean( optimal_k_list_of_performances )
    #percent_accuracy = average_of_5_performances * 100
    # Calculate standard deviation for the average accuracy measure
    stdev_accuracy = np.std( optimal_k_list_of_performances )

    tunedKnoSmoothingOutputFile.write( "\nAverage mean squared error: " + 
      str(round( average_of_5_performances, 4 )) + " for k equals " + 
      str(optimal_k_for_knn_no_smooth) + "\n" )
    tunedKnoSmoothingOutputFile.write( "Standard deviation: " + 
      str(round( stdev_accuracy, 4 )) + "\n" )

# Close this output file
tunedKnoSmoothingOutputFile.close()
  
    
    
