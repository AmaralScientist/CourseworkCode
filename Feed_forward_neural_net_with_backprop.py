##
 #  This program trains a feed forward neural network using backpropagation for use 
 #  with problems of classification. The trained neural network is tested and the
 #  classification accuracy is returned. The number of hidden layers to include in 
 #  the network is specified on the command line. The number of nodes for the hidden
 #  layers is hard coded. Input files are pre-processed data sets that must be
 #  formatted such that the target variable for prediction is labelled "Class" and 
 #  is located in the last column of the data frame. If the data set has more than 2 
 #  classes, they must be one-hot encoded and located between the last feature column 
 #  and the "Class" column. This program implements five-fold cross validation.
 #  Summary statistics are written to a txt file.
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
# Receive the number of hidden layers from the command line
number_hidden_layers = sys.argv[2]


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


# This method calculates the weighted sum of the feature values for
# one instance of a data set. Returns the weighted sum.
def calculate_weighted_sum_of_features ( feature_vector, weights ):
    # Calculate the product of the feature vector and the weights
    product = ( feature_vector * weights )
    
    # Sum the product to obtain weighted sum
    weighted_sum = product.sum( )
    
    return( weighted_sum )
    

# This method implements the sigmoid function 
def sigmoid_function( input_data ):
    # Calculate output from sigmoid function
    value = 1 / (1 + np.exp( -(input_data) ))    
    return( value )

    
# This method calculates the derivative of the sigmoid function for use
# with the back propagation function
def derivative_of_sigmoid_function( output_from_output_layer ):
    results_list = []
    for output_class_index in range( 0, len(output_from_output_layer) ):
        result = output_from_output_layer[output_class_index] * ( 1 - output_from_output_layer[output_class_index] )
        results_list.append( result )

    return( results_list )


# This method initializes a set of weights for a neural network in a list-of-lists  
# structure, based upon the number of hidden layers requested
def initialize_neural_net( number_of_classes, number_features, number_hidden_layers ):
    
    network = []
    number_nodes_in_hidden_layer = 7
    
    # For networks with two hidden layers
    if number_hidden_layers == 2:
        number_of_layers_in_network = 3
        for number_of_layers_in_network_index in range( 0, number_of_layers_in_network ):
            #print("number_of_layers_in_network_index: ", number_of_layers_in_network_index)
            if number_of_layers_in_network_index == 0:
                weights_array_for_each_node_in_hidden_layer = []           
                # for the nodes in the first hidden layer
                for node_index in range( 0, number_nodes_in_hidden_layer ):
                    weights_array = np.zeros( number_features + 1 )
                    # initialize a weight for each of the features plus bias
                    for feature_index in range( 0, number_features + 1 ):
                        weights_array[feature_index] = np.random.uniform(-0.01, 0.01)
                    weights_array_for_each_node_in_hidden_layer.append( weights_array )
                network.append( weights_array_for_each_node_in_hidden_layer )
            
            if number_of_layers_in_network_index == 1:
                weights_array_for_each_node_in_hidden_layer = []           
                # second hidden layer
                for second_layer_node_index in range( 0, number_nodes_in_hidden_layer ):
                    #print("number_nodes_in_hidden_layer: ", number_nodes_in_hidden_layer)
                    #print("node_index: ", node_index)
                    weights_array = np.zeros( number_nodes_in_hidden_layer + 1 )
                    # inputs from first hidden layer
                    for hidden_node_index in range( 0, number_nodes_in_hidden_layer + 1 ):
                        #print("number_nodes_in_hidden_layer: ", number_nodes_in_hidden_layer)
                        #print("node_index: ", node_index)
                        weights_array[hidden_node_index] = np.random.uniform(-0.01, 0.01)
                    weights_array_for_each_node_in_hidden_layer.append( weights_array )
                network.append( weights_array_for_each_node_in_hidden_layer )
                
            elif number_of_layers_in_network_index == 2:
                weights_array_for_all_classes = []
                for class_index in range( 0, number_of_classes):
                    #print("number_of_classes: ", number_of_classes)
                    weights_array = np.zeros( number_nodes_in_hidden_layer + 1 )
                    for node_index in range( 0, number_nodes_in_hidden_layer + 1 ):
                        weights_array[node_index] = np.random.uniform(-0.01, 0.01)
                    weights_array_for_all_classes.append( weights_array )
                network.append( weights_array_for_all_classes )               
    
    # For networks with one hidden layer
    if number_hidden_layers == 1:
        # initialize weights for each of the hidden nodes
        for hidden_layer_index in range( 0, number_hidden_layers ):
            weights_array_for_each_node_in_hidden_layer = []           
            for node_index in range( 0, number_nodes_in_hidden_layer ):
                weights_array = np.zeros( number_features + 1 )
                for feature_index in range( 0, number_features + 1 ):
                    weights_array[feature_index] = np.random.uniform(-0.01, 0.01)
                weights_array_for_each_node_in_hidden_layer.append( weights_array )
            network.append( weights_array_for_each_node_in_hidden_layer )
    
        weights_array_for_all_classes = []
        for class_index in range( 0, number_of_classes):
            weights_array = np.zeros( number_nodes_in_hidden_layer + 1 )
            for node_index in range( 0, number_nodes_in_hidden_layer + 1 ):
                weights_array[node_index] = np.random.uniform(-0.01, 0.01)                
            weights_array_for_all_classes.append( weights_array )
        network.append( weights_array_for_all_classes )
    
    # For networks with no hidden layers       
    if number_hidden_layers == 0:
        weights_array_for_all_classes = []
        for class_index in range( 0, number_of_classes):
            weights_array = np.zeros( number_features + 1 )
            for number_features_index in range( 0, number_features + 1 ):
                weights_array[number_features_index] = np.random.uniform(-0.01, 0.01)                
            weights_array_for_all_classes.append( weights_array )
        network.append( weights_array_for_all_classes )


    return( network )

# This method calculates the weighted sum of a vector of inputs with the respective
# weights and passes that value through the sigmoid activation function.
def weighted_sums_logistic_activation( number_of_nodes, activation_outputs, feature_instance, network, network_index ):
      
    for node_index in range( 0, number_of_nodes ):
        # Calculate weighted sum of the features of this instance
        weighted_sum = calculate_weighted_sum_of_features( feature_instance, network[network_index][node_index] )
        # Obtain output by passing weighted sum through the sigmoid function
        output_from_activation_function = sigmoid_function( weighted_sum )
        activation_outputs[node_index] = output_from_activation_function
        
    return( activation_outputs )
            

# This method calculates the weighted sum of a vector of inputs with the respective
# weights and passes that value through the softmax function. The probability 
# of each class given a multi-class classification problem is returned.
def weighted_sums_softmax_activation( number_of_nodes, array_of_weighted_sums, feature_instance, network, network_index ):
    for node_index in range( 0, number_of_nodes ):
        # Calculate weighted sum of the features of this instance
        weighted_sum = calculate_weighted_sum_of_features( feature_instance, network[network_index][node_index] )
        array_of_weighted_sums[node_index] = weighted_sum

    # Raise output for each class to exponent 
    output_for_each_class_exponential = np.exp( array_of_weighted_sums )
    # Sum the output over all classes for normalizing
    output_sum_of_all_classes = np.sum( output_for_each_class_exponential )
    # Determine the probability of each class
    probability_for_each_class = np.divide( output_for_each_class_exponential, output_sum_of_all_classes )
    
    return( probability_for_each_class )
    
    

# This method propagates training instances through the network in the forward 
# direction   
def proposed_forward_propagation( feature_instance, network ):
    network_length = len( network )
    all_layers_outputs = []
    # Begin obtaining weighted sums and activation outputs starting with the 
    # first layer of the network
    for network_index in range( 0, len(network) ):
        number_output_nodes = len(network[len(network)-1])
        # This is either the first hidden layer or there are no hidden layers, 
        # so the input will be the feature values of a given instance. If this is the 
        # case and it is a multi-class classification problem do the following        
        if number_output_nodes > 1:
            if network_index == 0:
                number_of_nodes = len( network[network_index] )
                activation_outputs = np.zeros( number_of_nodes )
                weighted_sums = np.zeros( len(network[0]) )

                # If there are no hidden layers and this is multi-class
                if network_length == 1:
                    probability_for_each_class = weighted_sums_softmax_activation( number_of_nodes, weighted_sums, feature_instance, network, network_index )
                    output_from_output_layer = probability_for_each_class
                    all_layers_outputs.append( output_from_output_layer )
            
                # If there are one or two hidden layers and this is multi-class
                else:          
                    activation_outputs = weighted_sums_logistic_activation( number_of_nodes, activation_outputs, feature_instance, network, network_index )
                    # Add a term for the next layer's intercept
                    intercept = np.ones(1)
                    activation_outputs = np.concatenate( (intercept, activation_outputs), axis = 0 )
                    all_layers_outputs.append( activation_outputs )
                    next_inputs = activation_outputs
        
        
            # This is either the first hidden layer of a one-hidden layer network or
            # this is the first or second hidden layer of a two-hidden layer network
            # This is for problems that are multi-class classification
            if network_index != 0:
                # This is either the second hidden layer or the output layer, so the input
                # will be the output from a previous layer and not the feature values
                number_of_nodes = len( network[network_index] )
                activation_outputs = np.zeros( number_of_nodes )
                weighted_sums = np.zeros( number_output_nodes )
            
                # This is the output node, use softmax and don't add an intercept term
                if network_index == len(network) - 1:
                    probability_for_each_class = weighted_sums_softmax_activation( number_of_nodes, weighted_sums, next_inputs, network, network_index )                
                    output_from_output_layer = probability_for_each_class
                    all_layers_outputs.append( output_from_output_layer )
            
                # If there are two hidden layers and this is multi-class
                else:          
                    activation_outputs = weighted_sums_logistic_activation( number_of_nodes, activation_outputs, next_inputs, network, network_index )
                    # Add a term for the next layer's intercept
                    intercept = np.ones(1)
                    activation_outputs = np.concatenate( (intercept, activation_outputs), axis = 0 )
                    all_layers_outputs.append( activation_outputs )
                    next_inputs = activation_outputs
        
        
        
        # This is either the first hidden layer or there are no hidden layers, 
        # so the input will be the feature values of a given instance. If this is the
        # case and it is a binary class classification problem do the following:
        if number_output_nodes == 1:
            if network_index == 0:
                number_of_nodes = len( network[network_index] )
                activation_outputs = np.zeros( number_of_nodes )
            
                # If there are one or two hidden layers and this is binary class
                if network_length == 1:
                    # There are no hidden layers
                    activation_outputs = weighted_sums_logistic_activation( number_of_nodes, activation_outputs, feature_instance, network, network_index )
                    output_from_output_layer = activation_outputs
                    all_layers_outputs.append( output_from_output_layer )
                else:
                    #print("Here I am")
                    activation_outputs = weighted_sums_logistic_activation( number_of_nodes, activation_outputs, feature_instance, network, network_index )
                    # Add a term for the next layer's intercept
                    intercept = np.ones(1)
                    activation_outputs = np.concatenate( (intercept, activation_outputs), axis = 0 )
                    all_layers_outputs.append( activation_outputs )
                    next_inputs = activation_outputs
                
            # This is either the first hidden layer of a one-hidden layer network or
            # this is the first or second hidden layer of a two-hidden layer network
            # If this is a binary class classification problem do the following:
            elif network_index != 0:
                number_of_nodes = len( network[network_index] )
                activation_outputs = np.zeros( number_of_nodes )
            
                # If there are one or two hidden layers and this is binary class
                if network_index == len(network) - 1:
                    # This is the output layer
                    activation_outputs = weighted_sums_logistic_activation( number_of_nodes, activation_outputs, next_inputs, network, network_index )
                    output_from_output_layer = activation_outputs
                    all_layers_outputs.append( output_from_output_layer )
                else:
                    #print("network_index: ", network_index)
                    #print("next_inputs: ", next_inputs)
                    activation_outputs = weighted_sums_logistic_activation( number_of_nodes, activation_outputs, next_inputs, network, network_index )
                    # Add a term for the next layer's intercept
                    intercept = np.ones(1)
                    activation_outputs = np.concatenate( (intercept, activation_outputs), axis = 0 )
                    all_layers_outputs.append( activation_outputs )
                    next_inputs = activation_outputs                

    return( all_layers_outputs )   


# This method calculates the difference between the output of the output layer 
# and the target class value. Also calculates the cross entropy error for the 
# logistic function.
def binary_class_cross_entropy_error( outputs_from_all_layers, correct_classes_as_array ):
    number_of_layer_outputs = len(outputs_from_all_layers)
    output_from_output_layer = outputs_from_all_layers[(number_of_layer_outputs-1)]
    target_output_error = np.subtract( correct_classes_as_array, output_from_output_layer )
    
    if correct_classes_as_array == 0:
        cost = -math.log2( 1 - output_from_output_layer[0] )
    else:
        cost = -math.log2( output_from_output_layer[0] )
    
    return( target_output_error, cost )
    
    
# This method calculates the difference between the output of the output layer 
# and the target class value. Also calculates the cross entropy error of softmax
def multi_class_cross_entropy_error( outputs_from_all_layers, correct_classes_as_array ):
    number_of_layer_outputs = len(outputs_from_all_layers)
    output_from_output_layer = outputs_from_all_layers[(number_of_layer_outputs-1)]
    
    target_output_error_list = np.zeros( len( correct_classes_as_array ) )
    cross_entropy_error_list = np.zeros( len( correct_classes_as_array ) )
    
    for class_index in range( 0, len( correct_classes_as_array ) ):
        correct_classes = correct_classes_as_array[class_index]
        output = output_from_output_layer[class_index]
        
        target_output_error = np.subtract( correct_classes, output )
        target_output_error_list[class_index] = target_output_error
        cross_entropy_error = -( correct_classes ) * math.log2( output )
        cross_entropy_error_list[class_index] = cross_entropy_error
    total_cross_entropy = np.sum( cross_entropy_error_list )

    return( target_output_error_list, total_cross_entropy )
    
    

# This method implements the backpropagation algorithm
def backward_propagation( outputs_from_all_layers, error_of_output_layer, number_of_classes, network, feature_instance ):
    network_length = len( network )
    number_output_nodes = len(network[len(network)-1])
    deltas_from_all_layers = []
    
    for network_index in reversed( range( network_length ) ):  
        # This is the output layer
        if network_index == (network_length - 1):
            # This is a binary classification problem
            if number_output_nodes == 1:
                # Multiply the error obtained after subtracting the output from the target value by derivative of sigmoid function
                delta_value = error_of_output_layer * derivative_of_sigmoid_function( outputs_from_all_layers[network_index] )
                deltas_from_all_layers.append( delta_value )
            # This is a multi-class classification problem    
            else:
                delta_value = error_of_output_layer
                deltas_from_all_layers.append( delta_value )
                   
        else:
            # This network has one hidden layer and an output layer
            if (network_length == 2 or network_length == 3):
            
                error_for_this_layer = []
                # Calculate the error for each respective node
                for this_layer_node in range( 0, len(network[network_index]) ):           
                    error = 0
                    for this_node_next_layer_index in range( 0, len(network[network_index+1]) ):
                        error_for_this_node = np.sum(deltas_from_all_layers[0][this_node_next_layer_index] * network[network_index+1][this_node_next_layer_index])
                        error = error + error_for_this_node
                    error_for_this_layer.append( error )
                # Convert to numpy array for downstream calculations
                error_for_this_layer = np.array( error_for_this_layer )
            
                # prepare output, as the first value is for the bias            
                output_for_this_layer = outputs_from_all_layers[network_index]
                output_corresponding_to_each_node = output_for_this_layer[1:len(output_for_this_layer)]
                        
                delta_value = error_for_this_layer * derivative_of_sigmoid_function( output_corresponding_to_each_node )
                # Prepend these values so they are in order when weights are updated
                deltas_from_all_layers.insert( 0, delta_value )

    # Update the weights starting with the output layer
    for network_index in reversed( range( network_length ) ):
        # This is the first hidden layer, so use the feature values as the input
        if network_index == 0:
            for this_layer_node in range( 0, len(network[network_index]) ):
                for weight_index in range( 0, len(network[network_index][this_layer_node]) ):
                    # This is the bias weight
                    if weight_index == 0:
                        weight_update = deltas_from_all_layers[network_index][this_layer_node] * learning_rate
                        this_current_weight = network[network_index][this_layer_node][weight_index]
                        network[network_index][this_layer_node][weight_index] = this_current_weight + weight_update
                    else:
                        weight_update = deltas_from_all_layers[network_index][this_layer_node] * learning_rate * feature_instance[weight_index-1]
                        this_current_weight = network[network_index][this_layer_node][weight_index]
                        network[network_index][this_layer_node][weight_index] = this_current_weight + weight_update
        # This is either the second hidden layer or the output layer                
        else:
            for this_layer_node in range( 0, len(network[network_index]) ):
                for weight_index in range( 0, len(network[network_index][this_layer_node]) ):
                    # This is the bias weight
                    if weight_index == 0:
                        #weight_update = delta_value[this_layer_node]* learning_rate
                        weight_update = deltas_from_all_layers[network_index][this_layer_node] * learning_rate
                        this_current_weight = network[network_index][this_layer_node][weight_index]
                        network[network_index][this_layer_node][weight_index] = this_current_weight + weight_update
                    else:
                        weight_update = deltas_from_all_layers[network_index][this_layer_node] * learning_rate * outputs_from_all_layers[network_index-1][weight_index-1]
                        this_current_weight = network[network_index][this_layer_node][weight_index]
                        network[network_index][this_layer_node][weight_index] = this_current_weight + weight_update
                        

# This method initializes a new network then passes training instances, one-by-one,
# through the forward and back propagation methods
def train_neural_network( df, number_of_classes, number_hidden_layers ):
    if number_of_classes == 2:
        class_df = df["Class"]
        df = df.drop(["Class"], axis = 1)
        number_features = df.shape[1]
        df_only_features = df.drop(df.iloc[:,number_features:df.shape[1]], axis = 1)
        # Only one output unit is needed for 2-class classification problems
        number_of_classes = 1
       
    else:
        df = df.drop(["Class"], axis = 1)
        number_features = df.shape[1] - number_of_classes
        class_df = df.iloc[:,number_features:df.shape[1]]
        df_only_features = df.drop(df.iloc[:,number_features:df.shape[1]], axis = 1)
    
    correct_classes_as_array = class_df.to_numpy()
    features_as_array = df_only_features.to_numpy()
    
    # Add a column of 1s for the intercept / bias
    intercept = np.ones((df.shape[0], 1))
    features_as_array = np.concatenate( (intercept, features_as_array), axis=1 )
    
    # Initialize a network structure
    network = initialize_neural_net( number_of_classes, number_features, number_hidden_layers )
    
    # Loop over number of epochs
    number_of_epochs = 125
    cross_entropy_error_for_all_epochs = []
    for iteration_index in range( 0, number_of_epochs ):
        cross_entropy_error_per_epoch = []
        for instance_index in range( 0, df_only_features.shape[0] ):
            outputs_from_all_layers = proposed_forward_propagation( features_as_array[instance_index], network )
            if number_of_classes == 1:
                target_output_error, cross_entropy_error = binary_class_cross_entropy_error( outputs_from_all_layers, correct_classes_as_array[instance_index] )
            else:
                target_output_error, cross_entropy_error = multi_class_cross_entropy_error( outputs_from_all_layers, correct_classes_as_array[instance_index] )
            cross_entropy_error_per_epoch.append( cross_entropy_error )
            backward_propagation( outputs_from_all_layers, target_output_error, number_of_classes, network, features_as_array[instance_index] )    
        total_cross_entropy_error_per_epoch = np.sum( cross_entropy_error_per_epoch )
        cross_entropy_error_for_all_epochs.append( total_cross_entropy_error_per_epoch )
    return( network, cross_entropy_error_for_all_epochs )

    
# This method tests the trained network on a set of test data and returns the 
# accuracy and a numpy array containing the predicted and target values for each
# instance
def test_neural_network( df, number_of_classes, network ):
    # If this is a binary class classification problem do the following:
    if number_of_classes == 2:
        class_df = df["Class"]
        df = df.drop(["Class"], axis = 1)
        number_features = df.shape[1]
        df_only_features = df.drop(df.iloc[:,number_features:df.shape[1]], axis = 1)
        number_of_classes = 1
        
        # Create numpy array to store classification results
        number_instances = df.shape[0] 
        testing_results = np.zeros( shape=( number_instances, 2 ) )
       
    # If this is a multi-class classification problem do the following:
    else:
        df = df.drop(["Class"], axis = 1)
        number_features = df.shape[1] - number_of_classes
        class_df = df.iloc[:,number_features:df.shape[1]]
        df_only_features = df.drop(df.iloc[:,number_features:df.shape[1]], axis = 1)
        
        # Create an empty list to hold the predicted results for each instance
        predicted_results_all_instances = []
   
    correct_classes_as_array = class_df.to_numpy()
    features_as_array = df_only_features.to_numpy()
    
    # Add a column of 1s for the intercept / bias
    intercept = np.ones((df.shape[0], 1))
    features_as_array = np.concatenate( (intercept, features_as_array), axis = 1 )
        
    length_of_network = len( network )
    
    # Initialize a counter variable to track the number of correct predictions
    correct_counter = 0
    
    # If this is a binary class classification problem do the following:
    if number_of_classes == 1:
        # Create empty list to store accuracy measure for each of the 5 folds
        list_of_performances = []
        for instance_index in range( 0, df_only_features.shape[0] ):
            outputs_from_all_layers = proposed_forward_propagation( features_as_array[instance_index], network )
            output_of_output_layer = outputs_from_all_layers[length_of_network-1]
            
            if output_of_output_layer >= 0.5:
                output_prediction = 1
            else:
                output_prediction = 0
            
            # Store result in array
            testing_results[instance_index, 0] = output_prediction

            if output_prediction == correct_classes_as_array[instance_index]:
                correct_counter = correct_counter + 1
            
            # Store target value in array
            testing_results[instance_index, 1] = correct_classes_as_array[instance_index]

        accuracy = correct_counter / features_as_array.shape[0]
        return( accuracy, testing_results )
    
    # If this is a multi-class classification problem do the following:    
    else:    
        for instance_index in range( 0, df_only_features.shape[0] ):
            outputs_from_all_layers = proposed_forward_propagation( features_as_array[instance_index], network )
        
            output_of_output_layer = outputs_from_all_layers[length_of_network-1]
            output_for_each_class_exponential = np.exp(output_of_output_layer)

            # now sum the output over all classes
            output_sum_of_all_classes = np.sum( output_for_each_class_exponential )
            
            probability_for_each_class = np.divide( output_for_each_class_exponential, output_sum_of_all_classes )
            predicted_results_all_instances.append( probability_for_each_class )
            highest_probability_class_index = np.argmax(probability_for_each_class)
        
            index_of_correct_class = np.argmax(correct_classes_as_array[instance_index])
        
            if highest_probability_class_index == index_of_correct_class:
                correct_counter = correct_counter + 1
            
        all_predictions = np.vstack( predicted_results_all_instances )
        final_results = np.concatenate( (all_predictions,correct_classes_as_array), axis=1 )
        
        accuracy = correct_counter / features_as_array.shape[0]
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
number_of_classes = len( class_names )


# Determine attribute names for the data set
column_names = df.columns.values.tolist()
number_of_columns_data_set = len(column_names)
number_of_columns_related_to_class = number_of_classes + 1
number_of_attributes = number_of_columns_data_set - number_of_columns_related_to_class
attribute_names = ["Intercept"]
for column_names_index in range(0, number_of_attributes):
    attribute_names.append( column_names[column_names_index] ) 


# Write results to an output file
with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_5/Outputs/" + output_file_name_final + "_output_" + number_hidden_layers + "_hidden_layers", 'w') as outputFile:

    outputFile.write( "\nData set: " + output_file_name_final + "\n" )
    outputFile.write( "\nType of problem: Classification\n" )
    outputFile.write( "\nNumber of hidden layers: " + number_hidden_layers + "\n" )
      
    outputFile.write( "Activation function: Logistic" + '\n')
    #outputFile.write( "Class Name for this model: " + this_class_name + ' \n\n')
    
    print( "\nData set: ", output_file_name_final, "\n" )
    print( "\nType of problem: Classification\n" )
    
    learning_rate = 0.025

    # Set up cross-validation: split data set into 5 folds 
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
        
        trained_network, cross_entropy_error_for_all_epochs = train_neural_network( temp_train_data_frame, number_of_classes, int(number_hidden_layers) )
        
        # Next run test set through the network and evaluate accuracy
        accuracy, testing_results = test_neural_network( temp_test_data_frame, number_of_classes, trained_network )
        outputFile.write( "\n\n\tClassification accuracy: " + str( accuracy ) )
             
        list_of_performances.append( accuracy )
        
        # If this is a binary class classification problem do the following:
        if number_of_classes == 2:
            # If this is the first round of cross validation, print the predicted and target values
            if index_lists == 0:
            
                    # Convert numpy array that contains the test results to data frame
                    testing_results_df = pd.DataFrame( testing_results )
                    testing_results_df.columns = [ "Predicted Class", "Correct Class" ]
                    # Write results of the test set predictions to file
                    testing_results_df.to_csv("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_5/Outputs/" + output_file_name_final + "_Test_set_classifications_for_fold_" + str(index_lists) + "_" + number_hidden_layers + "_hidden_layers", encoding='utf-8', index=False)
        
        # If this is a multi-class classification problem do the following:
        else:
            # If this is the first round of cross validation, print the predicted and target values
            if index_lists == 0:
        	    testing_results_df = pd.DataFrame( testing_results )
        	    column_names_list = []
        	    for name_index in range(0, len(class_names)):
        	        this_name = str(class_names[name_index]) + "_Predicted"
        	        column_names_list.append(this_name)
        	    for name_index in range(0, len(class_names)):
        	        this_name = str(class_names[name_index]) + "_Truth"
        	        column_names_list.append(this_name)
        	    
        	    #testing_results_df.columns = (class_names + class_names)
        	    testing_results_df.columns = column_names_list
        	    testing_results_df.to_csv("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_5/Outputs/" + output_file_name_final + "_Test_set_classifications_for_fold_" + str(index_lists) + "_" + number_hidden_layers + "_hidden_layers", encoding='utf-8', index=False)


    outputFile.write( "\n\n\nClassification accuracy across 5 folds: " + str(list_of_performances) + "\n" )
    average_classification_accuracy = np.mean( list_of_performances )
    outputFile.write( "\nAverage classification accuracy: " + str(round( average_classification_accuracy, 5 )) + "\n" )
    print("Average classification accuracy: ", average_classification_accuracy)
    std_classification_accuracy = np.std( list_of_performances )
    outputFile.write( "Standard deviation of classification accuracy: " + str(round(std_classification_accuracy, 5)) + "\n\n" )
    print("std_classification_accuracy: ", std_classification_accuracy)
    outputFile.write( "\nCross entropy error: " + str(cross_entropy_error_for_all_epochs) + "\n\n" )
    

        

