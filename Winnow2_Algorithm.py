##
 #  This program uses the Winnow2 algorithm to train and test a machine learning 
 #  classifier. The input is a pre-processed data set organized in a pandas dataframe. 
 #  The data set is randomly split into a training data set (2/3 of the number of   
 #  instances) and a test data set (1/3 of the number of instances). The learned model 
 #  and its summary statistics are written to a txt file. 
 #  @author Michelle Amaral
 #  @version 1.0
##

import sys,re,os

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Load counter
from collections import Counter 

# Receive the name of input file from the command line
input_file_name = sys.argv[1]
# Receive the starting index of the classes from the command line
feature_index_start = int( sys.argv[2] )
# Receive the ending index of the classes from the command line
feature_index_stop = int( sys.argv[3] )
# Receive the column number of the data frame that contains the first feature 
starting_column_for_features = int( sys.argv[4] )
# Receive the column number of the data frame that contains the classes from the
# command line
column_number_of_class = int( sys.argv[5] )

## Method accepts data frame containing the training data and trains model using
## Winnow-2 algorithm
def winnow2_train_model( current_df, start_of_attributes, theta, alpha, column_number_of_class ):

    # Determine the number of training set instances by calculating the number of rows
    number_of_instances = current_df.shape[0]
    
    # Calculate the number of columns in the training data set
    number_of_columns = current_df.shape[1]

    # Subtract Sample_ID column and Class column to arrive at the number of attributes
    number_of_attributes = number_of_columns - start_of_attributes
    
    # Create numpy array to store classification results
    training_results = np.zeros( shape=( number_of_instances, 5 ) )
    
    # Create vector filled with 1s to hold the calculated weights
    vector_of_weights = np.ones( number_of_attributes )

    for row_index in range( 0, number_of_instances ):
        weighted_sum = 0
        vector_product = ( current_df.iloc[row_index, start_of_attributes:number_of_columns].values * vector_of_weights )
        weighted_sum = vector_product.sum( )

        if weighted_sum > theta:
            prediction = 1
        else:
            prediction = 0
        
        # Store weighted sum
        training_results[row_index, 0] = weighted_sum
        # Store prediction
        training_results[row_index, 1] = prediction
    
        if (prediction == 1 and current_df.iloc[row_index, column_number_of_class] == 0):
            # Store result as demoted
            training_results[row_index, 3] = 1
        
            for index in range(0, len(vector_product)):
                if vector_product[index] != 0:
                    # Use a temporary variable to store the new weight for a particular index
                    temp = vector_of_weights[index] / alpha
                    # Update the vector of weights with the newly calculated weight for that index
                    vector_of_weights[index] = temp
                    
        elif (prediction == 0 and current_df.iloc[row_index, column_number_of_class] == 1):
            # Store result as promoted
            training_results[row_index, 2] = 1
            
            for index in range(0, len(vector_product)):
                if vector_product[index] != 0:
                    # Use a temporary variable to store the new weight for a particular index
                    temp = vector_of_weights[index] * alpha
                    # Update the vector of weights with the newly calculated weight for that index
                    vector_of_weights[index] = temp        
        else:
            # Store result as correct
            training_results[row_index, 4] = 1
    
    # return the vector containing the learned model and the training results            
    return(vector_of_weights, training_results)

## Method tests the learned model
def winnow2_test_model( this_df, vector_of_weights, start_of_attributes, theta, column_number_of_class ):
    
    # Determine the number of training set instances by calculating the number of rows
    number_of_instances = this_df.shape[0]
    #print("Number of instances: ", number_of_instances)
    
    # Calculate the number of columns in the training data set
    number_of_columns = this_df.shape[1]

    # Subtract Sample_ID column and Class column to arrive at the number of attributes
    number_of_attributes = number_of_columns - start_of_attributes
    
    # Create numpy array to store classification results
    testing_results = np.zeros( shape=( number_of_instances, 6 ) )
    
    # Initialize variables for counting        
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    # List holds the correct label of the false positive instances
    correct_class_of_false_positives = []
    # List holds the correct label of the false negative instances
    correct_class_of_false_negatives = []

    for row_index in range( 0, number_of_instances ):
        weighted_sum = 0
        vector_product = ( this_df.iloc[row_index, start_of_attributes:number_of_columns].values * vector_of_weights )
        weighted_sum = vector_product.sum( )

        if weighted_sum > theta:
            prediction = 1
        else:
            prediction = 0
            
        # Store weighted sum
        testing_results[row_index, 0] = weighted_sum
        # Store prediction
        testing_results[row_index, 1] = prediction
        
        if ( prediction == 1 and this_df.iloc[row_index, column_number_of_class] == 1 ):
            true_positive = true_positive + 1
            # Store result in numpy array
            testing_results[row_index, 2] = 1
            
        elif ( prediction == 0 and this_df.iloc[row_index, column_number_of_class] == 0 ):
            true_negative = true_negative + 1
            # Store result in numpy array
            testing_results[row_index, 3] = 1
        
        elif ( prediction == 1 and this_df.iloc[row_index, column_number_of_class] == 0 ):
            false_positive = false_positive + 1
            # Store result in numpy array
            testing_results[row_index, 4] = 1
            # Store the correct class
            correct_class_of_false_positives.append(this_df.iloc[row_index, (column_number_of_class + 1 )])
                    
        elif ( prediction == 0 and this_df.iloc[row_index, column_number_of_class] == 1 ):
            false_negative = false_negative + 1
            # Store result in numpy array
            testing_results[row_index, 5] = 1
            # Store the correct class
            correct_class_of_false_negatives.append(this_df.iloc[row_index, (column_number_of_class + 1)])
    
    return( testing_results, correct_class_of_false_positives, correct_class_of_false_negatives )
    


location_of_file = "/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_1/"

for index in range( feature_index_start, feature_index_stop ):
    # Read csv file containing the data set
    temp_df = pd.read_csv(location_of_file + "/Clean_Data_Files/" + input_file_name, header=[0], sep='\t') 
    
    # List to hold the unique names of the classes
    class_names = temp_df['Class'].unique().tolist()
    
    # For datasets with number of classes greater than 2
    temp_df["Multi_class"] = temp_df["Class"]
    multi_class_column = temp_df["Multi_class"]
    temp_df = temp_df.drop(["Multi_class"], axis = 1)    
    temp_df.insert(1, "Multi_class", multi_class_column)
    
    # Encode one class as '1' and the remaining as '0'
    temp_df["Class"] = [1 if x == class_names[index] else 0 for x in temp_df["Class"]]
    this_class_name = str(class_names[index])
    # Determine the number of instances in the full data set for each class 
    class_breakdown = temp_df.groupby(["Class"])["Class"].count().to_frame()
        
    # Create a data frame for the training data set
    df_training_data = temp_df.sample( frac = 0.67 )

    # Create a data frame for the testing data set
    df_testing_data = temp_df.drop( df_training_data.index )
    # Determine the number of instances in the test data set for each class 
    test_class_breakdown = df_testing_data.groupby(["Class"])["Class"].count().to_frame()
    
    # Set value of theta
    theta = 0.5
    # Set value of alpha
    alpha = 2

    # Train model
    winnow2_testing_vector, training_results = winnow2_train_model( df_training_data, starting_column_for_features, theta, alpha, column_number_of_class )
    # Convert model weights to a data frame
    weights_df = pd.DataFrame( winnow2_testing_vector )

	# Obtain the column names that correspond to the model weights
    attribute_names = list( df_training_data )
    attribute_names = attribute_names[ starting_column_for_features: ]    
    
    # Convert numpy array to data frame
    training_results_df = pd.DataFrame( training_results )
    training_results_df.columns = [ "Weighted_sum", "Prediction", "Promoted", "Demoted", "Correct" ]

    # Test model
    testing_results, correct_class_false_positives, correct_class_false_negatives = winnow2_test_model( df_testing_data, winnow2_testing_vector, starting_column_for_features, theta, column_number_of_class )
    # Convert numpy array that contains the test results to data frame
    testing_results_df = pd.DataFrame( testing_results )    
    testing_results_df.columns = [ "Weighted_sum", "Prediction", "True_positive", "True_negative", "False_positive", "False_negative" ]
    
    # Count the unique correct classes for the false positive predictions
    correct_false_positive_class_count = Counter(correct_class_false_positives)
    # Count the unique correct classes for the false negative predictions
    correct_false_negative_class_count = Counter(correct_class_false_negatives)
    
	# Prepare for output
	
	# Parse the input file name
    split_input_file_name = input_file_name.strip().split("_")
    output_file_name_list = []
    words_to_drop_from_name = ['clean']
    for split_index in range(0, len(split_input_file_name)):
        if words_to_drop_from_name[0] not in split_input_file_name[split_index]:
            output_file_name_list.append(split_input_file_name[split_index])
    output_file_name_final = '_'.join(output_file_name_list)
			
	# Write to an output file
    with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_1/Outputs/" + output_file_name_final + "_output_Winnow2_algorithmTEST_" + this_class_name, 'w') as outputFile:
        outputFile.write( "Data set: " + output_file_name_final + '\n\n')
        outputFile.write( "Algorithm: Winnow2" + '\n\n')
		
        outputFile.write( "Class Name for this model: " + this_class_name + ' (1)\n\n')
        # Print the learned model
        outputFile.write( "Learned model")
        
        for weights_index in range(0, len( winnow2_testing_vector )):
            outputFile.write( "\n" + attribute_names[weights_index] + ": " + str(winnow2_testing_vector[weights_index]) )
            
        # Print the summary statistics
        outputFile.write( "\n\nSummary Statistics")
        
        number_training_instances = df_training_data.shape[0]
        outputFile.write( "\nNumber of training instances: " + str(number_training_instances) ) 
                        
        number_testing_instances = df_testing_data.shape[0]
        outputFile.write( "\nNumber of testing instances: " + str(number_testing_instances) ) 
        outputFile.write( "\nNumber of test instances of class '0': " + str(test_class_breakdown.iloc[0,0]) )
        outputFile.write( "\nNumber of test instances of class '1': " + str(test_class_breakdown.iloc[1,0]) ) 
    
        true_positives = testing_results[:,2].sum()
        outputFile.write( "\nNumber of true positives: " + str(true_positives) ) 
        true_negatives = testing_results[:,3].sum()
        outputFile.write( "\nNumber of true negatives: " + str(true_negatives) ) 
        false_positives = testing_results[:,4].sum()
        outputFile.write( "\nNumber of false positives: " + str(false_positives) ) 
        false_negatives = testing_results[:,5].sum()
        outputFile.write( "\nNumber of false negatives: " + str(false_negatives) )
        outputFile.write( "\nCorrect false positive classes for confusion matrix: " + str(correct_false_positive_class_count) )
        outputFile.write( "\nCorrect false negative classes for confusion matrix: " + str(correct_false_negative_class_count) )
        
        accuracy = (true_positives + true_negatives) / number_testing_instances
        outputFile.write( "\nAccuracy: " + str(round(accuracy, 3)) ) 
        recall = true_positives / (true_positives + false_negatives)
        outputFile.write( "\nRecall: " + str(round(recall, 3)) ) 
        precision = true_positives / (true_positives + false_positives)
        outputFile.write( "\nPrecision: " + str(round(precision, 3)) ) 
        f_measure = (2*recall*precision) / (recall + precision)
        outputFile.write( "\nF measure: " + str(round(f_measure, 3)) ) 
		
		# Write results of the test set predictions to file
        testing_results_df.to_csv("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_1/Outputs/" + output_file_name_final + "_Test_set_classifications_Winnow2_algorithm_" + this_class_name , encoding='utf-8', index=False)








