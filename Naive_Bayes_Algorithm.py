##
 #  This program uses the Naive Bayes algorithm to train and test a machine learning 
 #  classifier. The input is a pre-processed data set organized in a pandas dataframe. 
 #  The data set is randomly split into a training data set (2/3 of the number of   
 #  instances) and a test data set (1/3 of the number of instances). The summary
 #  statistics are written to a txt file. 
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

## Method accepts a data frame that has been split by class. For each feature, it 
## counts and stores the number of times a feature is '0' and the number of  
## times the feature is '1'. The values are stored in a dictionary object.
def create_lookup_dict( df_split_by_class, start_of_attributes ):
    
    number_class_instances = df_split_by_class.shape[0]
    number_of_columns = df_split_by_class.shape[1]
    
    lookup_dict = {}

    for column_index in range( start_of_attributes, number_of_columns ):    
        nesting_dictionary = {}
        # for a given feature, this is the number of times the value of that feature is 0 
        nesting_dictionary["0"] = (df_split_by_class.iloc[:,column_index].eq(0).astype(int).sum() / number_class_instances)
        # for a given feature, this is the number of times the value of that feature is 1
        nesting_dictionary["1"] = (df_split_by_class.iloc[:,column_index].eq(1).astype(int).sum() / number_class_instances)
    
        lookup_dict[column_index] = nesting_dictionary
    
    return(lookup_dict)

## Method tests a Naive Bayes classifier
def naive_bayes_model( test_df, class_0_dict, class_1_dict, prior_probability_class_0, prior_probability_class_1, start_of_attributes, column_number_of_class ):
    
    # Determine the number of training set instances by calculating the number of rows
    number_of_instances = test_df.shape[0]
    
    # Calculate the number of columns in the data set
    number_of_columns = test_df.shape[1]

    # Subtract Sample_ID column and Class column to arrive at the number of attributes
    number_of_attributes = number_of_columns - start_of_attributes
    
    # Create numpy array to store classification results
    testing_results = np.zeros( shape=( number_of_instances, 5 ) )
    
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
        # Initialize a variable to hold the product of the conditional probability
        # calculations with the prior probability of class '0'
        product_value_class_0 = 1 * prior_probability_class_0 
        # Initialize a variable to hold the product of the conditional probability
        # calculations with the prior probability of class '1'
        product_value_class_1 = 1 * prior_probability_class_1 
        # Store the correct category of an instance
        class_for_this_instance = test_df.iloc[row_index, column_number_of_class]
    
        for column_index in range( start_of_attributes, number_of_columns ):
        
            # probability calculation if class = 0
            if test_df.iloc[row_index, column_index] == 0:
                # calculation if the class is equal to 0
                product_value_class_0 = product_value_class_0 * class_0_dict[column_index]["0"]
            
            
            elif test_df.iloc[row_index, column_index] == 1:
                # calculation if the class is equal to 0
                product_value_class_0 = product_value_class_0 * class_0_dict[column_index]["1"]
            
        
            # probability calculation if class = 1
            if test_df.iloc[row_index, column_index] == 0:
                # calculation if the class is equal to 0
                product_value_class_1 = product_value_class_1 * class_1_dict[column_index]["0"]
            
            
            elif test_df.iloc[row_index, column_index] == 1:
                # calculation if the class is equal to 0
                product_value_class_1 = product_value_class_1 * class_1_dict[column_index]["1"]
        
    
        if product_value_class_0 > product_value_class_1:
            prediction = 0
        elif product_value_class_0 < product_value_class_1:
            prediction = 1
        else:
        	prediction = 0
            
        # Store prediction
        testing_results[row_index, 0] = prediction
        
        # Evaluate prediction vs correct label
        if ( prediction == 1 and class_for_this_instance == 1 ):
            true_positive = true_positive + 1
            # Store result in numpy array
            testing_results[row_index, 1] = 1
        
        elif ( prediction == 0 and class_for_this_instance == 0 ):
            true_negative = true_negative + 1
            # Store result in numpy array
            testing_results[row_index, 2] = 1
        
        elif ( prediction == 1 and class_for_this_instance == 0 ):
            false_positive = false_positive + 1
            # Store result in numpy array
            testing_results[row_index, 3] = 1
            correct_class_of_false_positives.append(test_df.iloc[row_index, (column_number_of_class + 1)])
                    
        elif ( prediction == 0 and class_for_this_instance == 1 ):
            false_negative = false_negative + 1
            # Store result in numpy array
            testing_results[row_index, 4] = 1
            correct_class_of_false_negatives.append(test_df.iloc[row_index, (column_number_of_class + 1)])
            
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
    df_training_data = temp_df.sample(frac = 0.67)
    
    # Create a data frame for the testing data set
    df_testing_data = temp_df.drop(df_training_data.index)
    # Determine the number of instances in the test data set for each class 
    test_class_breakdown = df_testing_data.groupby(["Class"])["Class"].count().to_frame()

    total_instances_training_df = df_training_data.shape[0]

    # Split the training data set into 2 data frames: one containing instances of class 0 and one containing instances of class 1
    df_class_0 = df_training_data.loc[df_training_data['Class'] == 0]
    df_class_1 = df_training_data.loc[df_training_data['Class'] == 1]
	
	# Calculate the prior probability of class 0
    prior_probability_class_0 = df_class_0.shape[0] / total_instances_training_df

    # Calculate the prior probability of class 1
    prior_probability_class_1 = df_class_1.shape[0] / total_instances_training_df

    # Create dictionary for training data set containing instances where class = 0
    class_0_dict = create_lookup_dict( df_class_0, starting_column_for_features )

    # Create dictionary for training data set containing instances where class = 1
    class_1_dict = create_lookup_dict( df_class_1, starting_column_for_features )
	
	# Train Naive Bayes classifier
    training_naive_bayes, correct_class_false_positives_train, correct_class_false_negatives_train = naive_bayes_model(df_training_data, class_0_dict, class_1_dict, prior_probability_class_0, prior_probability_class_1, starting_column_for_features, column_number_of_class)
    # Convert numpy array to data frame
    training_results_df = pd.DataFrame( training_naive_bayes )
    training_results_df.columns = [ "Prediction", "True_positive", "True_negative", "False_positive", "False_negative" ]    
    
    # Test the learned model
    testing_naive_bayes, correct_class_false_positives, correct_class_false_negatives = naive_bayes_model(df_testing_data, class_0_dict, class_1_dict, prior_probability_class_0, prior_probability_class_1, starting_column_for_features, column_number_of_class)
    # Convert numpy array to data frame
    testing_results_df = pd.DataFrame( testing_naive_bayes )    
    testing_results_df.columns = [ "Prediction", "True_positive", "True_negative", "False_positive", "False_negative" ]
    
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
			
	# Write to output file
    with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_1/Outputs/" + output_file_name_final + "_output_Naive_Bayes_algorithm_TEST" + this_class_name, 'w') as outputFile:
        outputFile.write( "Data set: " + output_file_name_final + '\n\n' )
        outputFile.write( "Algorithm: Naive Bayes\n" )		

        outputFile.write( "Class Name for this model: " + this_class_name + ' (1)\n')

        # Print the summary statistics
        outputFile.write( "\n\nSummary Statistics" )
                
        number_training_instances = df_training_data.shape[0]
        outputFile.write( "\n\nNumber of training instances: " + str(number_training_instances) ) 
        number_testing_instances = df_testing_data.shape[0]
        outputFile.write( "\nNumber of testing instances: " + str(number_testing_instances) ) 
        outputFile.write( "\nNumber of test instances of class '0': " + str( test_class_breakdown.iloc[0,0]) )
        outputFile.write( "\nNumber of test instances of class '1': " + str( test_class_breakdown.iloc[1,0]) ) 
    
        true_positives = testing_naive_bayes[:,1].sum()
        outputFile.write( "\nNumber of true positives: " + str(true_positives) ) 
        true_negatives = testing_naive_bayes[:,2].sum()
        outputFile.write( "\nNumber of true negatives: " + str(true_negatives) ) 
        false_positives = testing_naive_bayes[:,3].sum()
        outputFile.write( "\nNumber of false positives: " + str(false_positives) ) 
        false_negatives = testing_naive_bayes[:,4].sum()
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

    