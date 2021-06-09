##
 #  This program uses the logistic regression algorithm to train and test a machine 
 #  learning classifier. Input files are pre-processed data sets that must be formatted 
 #  such that the target variable for prediction is labelled "Class" and is located in 
 #  the last column of the data frame. If the data set has more than 2 classes, they
 #  must be one-hot encoded and located between the last feature column and the "Class"
 #  column. This program implements five-fold cross validation. The learned model
 #  and the summary statistics are written to a txt file.
 #  @author Michelle Amaral
 #  @version 1.0
##


import sys,re,os
# Load pandas
import pandas as pd
# Load numpy
import numpy as np
import random
import math

# Receive the name of input file from the command line
input_file_name = sys.argv[1]

# This method accesses a data frame and splits its row numbers into k lists of row
# numbers, each corresponding to k unique test folds for use with k fold cross-validation
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
        temp_data_frame = data_frame.loc[data_frame['Class'] == class_breakdown.index.values[class_index]]
        # Determine the actual row numbers of those instances
        row_numbers_for_class = list(temp_data_frame.index.values.astype(int))
        # Append these row numbers to list
        list_of_class_row_numbers.append(row_numbers_for_class)
        # Calculate the ratio class instances:number total instancess in big data set
        composition_ratio = len(row_numbers_for_class) / data_frame.shape[0]
        # Calculate the number of instances needed for each fold
        number_instances_of_this_class_needed = number_instances_per_fold * composition_ratio
        rounded_number_instances_of_this_class_needed = math.floor(number_instances_of_this_class_needed)
        number_instances_of_each_class.append( rounded_number_instances_of_this_class_needed)
    

    for k_index in range(0, number_of_folds):
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
            #print("length of row numbers of interest for ", class_index, ": ", len(row_numbers_of_interest))
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
        # Shuffle row numbers
        temp_row_numbers_for_this_fold = random.sample( temp_row_numbers_for_this_fold, len(temp_row_numbers_for_this_fold) )       
        # Append all row numbers to final list
        list_of_row_numbers_for_test_folds.append( temp_row_numbers_for_this_fold )
        
    return( list_of_row_numbers_for_test_folds )


# This method calculates cross entropy loss for a binary class classification problem
def cross_entropy_loss( predicted, truth ):
    if truth == 0:
        cost = -math.log2( 1 - predicted )
    else:
        cost = -math.log2( predicted )
    return( cost )
    

# This method calculates cross entropy loss for a multi class classification problem
def cross_entropy_loss_multi_class( predicted, truth ):
    running_total_cost = 0
    
    for array_index in range(0, len( predicted )):
        single_class_cost = truth[array_index] * math.log2(predicted[array_index])
        running_total_cost = running_total_cost + single_class_cost
    
    running_total_cost = -1 * running_total_cost
    
    return( running_total_cost )


# This method implements the sigmoid function 
def sigmoid_function( input_data ):
    value = 1 / (1 + np.exp( -(input_data) ))
    return( value )
    
    
# This method implements the logistic regression algorithm and uses gradient 
# descent to determine the coefficients for each attribute in a data frame
def binary_logistic_regression( df, learning_rate ):
    #df = df.drop(["Class"], axis = 1)    
    number_features = df.shape[1] - 1
    number_of_classes = 1
    
    class_df = df.iloc[:,number_features:df.shape[1]]
    df_only_features = df.drop(df.iloc[:,number_features:df.shape[1]], axis = 1)
    
    correct_classes_as_array = class_df.to_numpy()
    features_as_array = df_only_features.to_numpy()
    
    # Add a column of 1s for the intercept / bias
    intercept = np.ones((df.shape[0], 1))
    features_as_array = np.concatenate( (intercept, features_as_array), axis=1 )       
    
    # Prepare empty array to store the weight values for all classes            
    weights_of_all_classes = np.zeros((number_of_classes, number_features+ 1))
    for feature_index in range( 0, number_features ):
        for class_index in range( 0, number_of_classes ):
            weights_of_all_classes[class_index, feature_index] = np.random.uniform(-0.01, 0.01)     
    
    # Create empty list to hold cross-entropy loss for each epoch
    cost_for_all_epochs = []        
    for step_index in range(30):
        # Create empty matrix to store the sum of all the error      
        delta_weights_of_all_classes = np.zeros((number_of_classes, number_features+1))
        # Initialize variable to track cost
        cost_counter = 0
        for row_index in range( 0, df.shape[0] ):
            # Calculate output for this instance
            features_by_weights_product = ( features_as_array[row_index,] * weights_of_all_classes )
            # Sum the features_by_weights_product array by row to obtain output for each class
            output_for_each_class = features_by_weights_product.sum( axis=1 )
            
            # Predict class by running output through the sigmoid function
            y_hat = sigmoid_function( output_for_each_class )
            
            # Calculate error
            error = correct_classes_as_array[row_index] - y_hat
            # Calculate cost
            cost_counter = cost_counter + cross_entropy_loss( y_hat, correct_classes_as_array[row_index] )      
            error_by_feature_product_all_classes = []
            # Next, multiply the error by each feature (calculating gradient)
            for class_index in range( 0, number_of_classes ):
                resulting_product = error[class_index] * features_as_array[row_index,]
                error_by_feature_product_all_classes.append( resulting_product )

            delta_weights_this_instance = np.vstack( error_by_feature_product_all_classes )
            # Add to the running total
            delta_weights_of_all_classes = np.add( delta_weights_of_all_classes, delta_weights_this_instance )
        
        #average_delta_weights = delta_weights_of_all_classes / df.shape[0]
        delta_weights_learning_rate_product = delta_weights_of_all_classes * learning_rate
        # Update the weights   
        weights_of_all_classes = np.add( weights_of_all_classes, delta_weights_learning_rate_product )
        cost_for_all_epochs.append( cost_counter )
    return( weights_of_all_classes, cost_for_all_epochs )    


# This method accepts a test data frame and model coefficients and makes a class
# prediction for each test instance    
def binary_class_predict( df, weights_of_all_classes ):
    
    number_instances = df.shape[0]   
    number_features = df.shape[1] - 1
    
    # Generate numpy matrix for the class values
    class_df = df.iloc[:,number_features:df.shape[1]]
    correct_classes_as_array = class_df.to_numpy()
    
    # Generate numpy matrix for the feature values
    df_only_features = df.drop( df.iloc[:,number_features:df.shape[1]], axis = 1 )    
    features_as_array = df_only_features.to_numpy()
    
    # Add a column of 1s for the intercept / bias
    intercept = np.ones((df.shape[0], 1))
    features_as_array = np.concatenate( (intercept,features_as_array), axis=1 )
    
    # Create numpy array to store classification results
    testing_results = np.zeros( shape=( number_instances, 1 ) )
       
    correct_counter = 0
    for row_index in range( 0, number_instances ):
        the_features = features_as_array[row_index,]
        features_by_weights_product = the_features * weights_of_all_classes
        #print("\nthe_features: ", the_features)
        #print("weights: ", weights_of_all_classes)
        #print("product: ", features_by_weights_product)
        
        
        # Sum the features_by_weights_product array by row to obtain output for each class
        output_for_each_class = features_by_weights_product.sum( axis=1 ) 
        print("output_for_each_class: ", output_for_each_class)
        
        # Make prediction
        y_hat = np.round_( sigmoid_function(output_for_each_class) )
                
        # Store result in array
        testing_results[row_index, 0] = y_hat

        correct_class_this_instance = correct_classes_as_array[row_index]
        print("correct_class_this_instance: ", correct_class_this_instance)
        
        # Compare predicted class with correct class
        if y_hat == correct_class_this_instance:
            correct_counter = correct_counter + 1
    
    #all_predictions = np.vstack( predicted_results_all_instances )
    final_results = np.concatenate( (testing_results, correct_classes_as_array), axis=1 )
    print("final_results: ", final_results)
    print("correct_counter: ", correct_counter)
    print("number_instances: ", number_instances)
    # Calculate classification accuracy for this test data set   
    accuracy = correct_counter / number_instances
    print("accuracy: ", accuracy)
    # Calculate classification error for this test data set
    #classification_error = 1 - accuracy
    
    return( accuracy, final_results )  




# This method implements multi-class logistic regression with softmax           
def multi_class_logistic_regression( df, learning_rate, number_of_classes ):
    df = df.drop(["Class"], axis = 1)    
    number_features = df.shape[1] - number_of_classes
    
    class_df = df.iloc[:,number_features:df.shape[1]]
    df_only_features = df.drop(df.iloc[:,number_features:df.shape[1]], axis = 1)
    
    correct_classes_as_array = class_df.to_numpy()
    features_as_array = df_only_features.to_numpy()
    
    
    # Add a column of 1s for the intercept / bias
    intercept = np.ones((df.shape[0], 1))
    features_as_array = np.concatenate( (intercept,features_as_array), axis=1 )       
            
    weights_of_all_classes = np.zeros((number_of_classes, number_features+ 1))
    for feature_index in range( 0, number_features ):
        for class_index in range( 0, number_of_classes ):
            weights_of_all_classes[class_index, feature_index] = np.random.uniform(-0.01, 0.01)        
    cost_for_all_epochs_multi_class = []
    for step_index in range(25):
      
        delta_weights_of_all_classes = np.zeros((number_of_classes, number_features+1))
        cost_counter_multi_class = 0
        for row_index in range( 0, df.shape[0] ):
            features_by_weights_product = (features_as_array[row_index,] * weights_of_all_classes)
            # Sum the features_by_weights_product array by row to obtain output for each class
            output_for_each_class = features_by_weights_product.sum( axis=1 )
            # Raise output for each class to exponent 
            output_for_each_class_exponential = np.exp( output_for_each_class )
            
            # Sum the output over all classes for normalizing
            output_sum_of_all_classes = np.sum( output_for_each_class_exponential )
            # Determine the probability of each class
            probability_for_each_class = np.divide( output_for_each_class_exponential, output_sum_of_all_classes )
            
            # Calculate error
            error = correct_classes_as_array[row_index] - probability_for_each_class            
            cost_counter_multi_class = cost_counter_multi_class + cross_entropy_loss_multi_class( probability_for_each_class, correct_classes_as_array[row_index] )
            error_by_feature_product_all_classes = []
            for class_index in range( 0, number_of_classes ):
                resulting_product = error[class_index] * features_as_array[row_index,]
                error_by_feature_product_all_classes.append( resulting_product )

            delta_weights_this_instance = np.vstack( error_by_feature_product_all_classes )
            # Add to the running total
            delta_weights_of_all_classes = np.add( delta_weights_of_all_classes, delta_weights_this_instance )
        
        #average_delta_weights = delta_weights_of_all_classes / df.shape[0]
        delta_weights_learning_rate_product = delta_weights_of_all_classes * learning_rate
        # Update the weights   
        weights_of_all_classes = np.add( weights_of_all_classes, delta_weights_learning_rate_product )
        cost_for_all_epochs_multi_class.append( cost_counter_multi_class )
    return( weights_of_all_classes, cost_for_all_epochs_multi_class )    


def multi_class_predict( df, weights_of_all_classes, number_of_classes ):
    
    number_instances = df.shape[0]    
    df = df.drop( ["Class"], axis = 1 )    
    number_features = df.shape[1] - number_of_classes
    
    # Generate numpy matrix for the class values
    class_df = df.iloc[:,number_features:df.shape[1]]
    correct_classes_as_array = class_df.to_numpy()
    
    # Generate numpy matrix for the feature values
    df_only_features = df.drop( df.iloc[:,number_features:df.shape[1]], axis = 1 )    
    features_as_array = df_only_features.to_numpy()
    
    # Add a column of 1s for the intercept / bias
    intercept = np.ones((df.shape[0], 1))
    features_as_array = np.concatenate( (intercept,features_as_array), axis=1 )

    predicted_results_all_instances = []
       
    correct_counter = 0
    for row_index in range( 0, number_instances ):
        the_features = features_as_array[row_index,]
        features_by_weights_product = the_features * weights_of_all_classes
        print("\nthe_features: ", the_features)
        print("weights: ", weights_of_all_classes)
        print("product: ", features_by_weights_product)
        
        
        # Sum the features_by_weights_product array by row to obtain output for each class
        output_for_each_class = features_by_weights_product.sum( axis=1 ) 
        print("output_for_each_class: ", output_for_each_class)
            
        output_for_each_class_exponential = np.exp(output_for_each_class)
        print("expoen: ", output_for_each_class_exponential)

        # now sum the output over all classes
        output_sum_of_all_classes = np.sum( output_for_each_class_exponential )
        print("sum_of_all_classes: ", output_for_each_class_exponential)
            
        probability_for_each_class = np.divide( output_for_each_class_exponential, output_sum_of_all_classes )
        predicted_results_all_instances.append( probability_for_each_class )
        print("probability_for_each_class: ", probability_for_each_class)
        highest_probability_class_index = np.argmax(probability_for_each_class)
        print("highest_probability_class_index: ", highest_probability_class_index)
        
        print("correct_classes_as_array[row_index]: ", correct_classes_as_array[row_index])
        correct_class_this_instance = np.argmax(correct_classes_as_array[row_index])
        print("correct_class_this_instance: ", correct_class_this_instance)
        
        if highest_probability_class_index == correct_class_this_instance:
            correct_counter = correct_counter + 1
    
    all_predictions = np.vstack( predicted_results_all_instances )
    final_results = np.concatenate( (all_predictions,correct_classes_as_array), axis=1 )
    print("correct_counter: ", correct_counter)
    print("number_instances: ", number_instances)
    accuracy = correct_counter / number_instances
    print("accuracy: ", accuracy)
    #classification_error = 1 - accuracy

    
    return( accuracy, final_results )  



      

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


# Determine number of classes in the classification problem
# If there are more than 2 class, set up for multi-class classification
class_names = df['Class'].unique().tolist()
number_of_classes = len(class_names)


# Determine attribute names for the data set
column_names = df.columns.values.tolist()
number_of_columns_data_set = len(column_names)
number_of_columns_related_to_class = number_of_classes + 1
number_of_attributes = number_of_columns_data_set - number_of_columns_related_to_class
attribute_names = ["Intercept"]
for column_names_index in range(0, number_of_attributes):
    attribute_names.append( column_names[column_names_index] ) 

# Write results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_4/Outputs/" + output_file_name_final + "_output_Logistic_Regression", 'w') as outputFile:

    outputFile.write( "\nData set: " + output_file_name_final + "\n" )
    outputFile.write( "\nType of problem: Classification\n" )
    outputFile.write( "Algorithm: Logistic Regression" + '\n')
    #outputFile.write( "Class Name for this model: " + this_class_name + ' \n\n')
    
    print( "\nData set: ", output_file_name_final, "\n" )
    print( "\nType of problem: Classification\n" )
    
    outputFile.write( "\n***********************\n" )
    outputFile.write( "\n Logistic Regression \n" )
    outputFile.write( "\n***********************\n" )

        

    learning_rate = 0.01

    # Split data set into 5 folds 
    test_folds_row_numbers = split_for_cross_validation( df, 5 )

    # Create empty list to store accuracy measure for each of the 5 folds
    list_of_performances = []

    for index_lists in range(0, len(test_folds_row_numbers)):
        outputFile.write( "\n\nWhen fold " + str( index_lists ) + " is the test set:\n" )
        print("  When fold", index_lists, "is the test set:")
        # Obtain row numbers for test fold
        temp_df_row_list = test_folds_row_numbers[index_lists]
        # Obtain test data frame using row numbers for test fold
        temp_test_data_frame = df.iloc[temp_df_row_list,]
        temp_test_data_frame = temp_test_data_frame.reset_index( drop=True )
            
        # Obtain train data frame by dropping row numbers for test fold
        temp_train_data_frame = df.drop( df.index[temp_df_row_list] )
        # Shuffle the rows of the data frame
        temp_train_data_frame = temp_train_data_frame.iloc[np.random.permutation(len(temp_train_data_frame))]
        temp_train_data_frame = temp_train_data_frame.reset_index( drop=True )

        # This is a binary class classification problem
        if number_of_classes == 2:

            weights_list, cost_for_all_epochs = binary_logistic_regression( temp_train_data_frame, learning_rate )
            print("weights_list: ", weights_list)
            outputFile.write( "\n\tLearned model:")
            this_fold_performance, testing_results = binary_class_predict(temp_test_data_frame, weights_list)
            list_of_performances.append( this_fold_performance )
            print("list_of_performances: ", list_of_performances)
            for class_index in range( 0, 1 ):
            	for weights_index in range(0, number_of_attributes + 1):
                	outputFile.write( "\n\t\t" + attribute_names[weights_index] + ": " + str(weights_list[class_index][weights_index]) )
            
            outputFile.write( "\n\n\tClassification accuracy: " + str( this_fold_performance ) )           
            outputFile.write( "\n\tCost from all epochs: " + str( cost_for_all_epochs ) )
            if index_lists == 0:
            
                # Convert numpy array that contains the test results to data frame
                testing_results_df = pd.DataFrame( testing_results )
                testing_results_df.columns = [ "Predicted Class", "Correct Class" ]
                # Write results of the test set predictions to file
                testing_results_df.to_csv("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_4/Outputs/" + output_file_name_final + "_Test_set_classifications_for_fold_" + str(index_lists) + "_Logistic_Regression", encoding='utf-8', index=False)
        
        # This is a multi-class classification problem   
        else:
        	weights_list, cost_for_all_epochs_multi_class = multi_class_logistic_regression( temp_train_data_frame, learning_rate, number_of_classes )
        	print("weights_list: ", weights_list)
        	outputFile.write( "\n\tLearned model:\n")
        	this_fold_performance, all_predictions = multi_class_predict(temp_test_data_frame, weights_list, number_of_classes)
        	list_of_performances.append( this_fold_performance )
        	print("list_of_performances: ", list_of_performances)
        	for class_index in range( 0, number_of_classes ):
        		outputFile.write( "\n\t" + str(class_names[class_index]) )
        		for weights_index in range(0, number_of_attributes + 1 ):
        			outputFile.write( "\n\t\t" + attribute_names[weights_index] + ": " + str(weights_list[class_index][weights_index]) ) 
        	outputFile.write( "\n\n\tClassification accuracy: " + str( this_fold_performance ) )
        	outputFile.write( "\n\tCost from all epochs: " + str( cost_for_all_epochs_multi_class ) )
        	if index_lists == 0:
        	    testing_results_df = pd.DataFrame( all_predictions )
        	    column_names_list = []
        	    for name_index in range(0, len(class_names)):
        	        this_name = str(class_names[name_index]) + "_Predicted"
        	        column_names_list.append(this_name)
        	    for name_index in range(0, len(class_names)):
        	        this_name = str(class_names[name_index]) + "_Truth"
        	        column_names_list.append(this_name)
        	    
        	    #testing_results_df.columns = (class_names + class_names)
        	    testing_results_df.columns = column_names_list
        	    testing_results_df.to_csv("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_4/Outputs/" + output_file_name_final + "_Test_set_classifications_for_fold_" + str(index_lists) + "_Logistic_Regression", encoding='utf-8', index=False)
        	
    outputFile.write( "\n\n\nClassification accuracy across 5 folds: " + str(list_of_performances) + "\n" )
    average_classification_accuracy = np.mean( list_of_performances )
    outputFile.write( "\nAverage classification accuracy: " + str(round( average_classification_accuracy, 5 )) + "\n" )
    print("Average classification accuracy: ", average_classification_accuracy)
    std_classification_accuracy = np.std( list_of_performances )
    outputFile.write( "Standard deviation of classification accuracy: " + str(round(std_classification_accuracy, 5)) + "\n\n" )
    print("std_classification_accuracy: ", std_classification_accuracy)






