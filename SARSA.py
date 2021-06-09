##
 #  This program implements the SARSA algorithm for the racetrack problem where an  
 #  agent (a race car) is trained and tested on various racetracks via reinforcement 
 #  learning. Input files are racetracks represented in ASCII with the first line 
 #  containing the size of the track and the remainder containing one character at
 #  each position as follows: 'S' is the starting line, 'F' is the finish line, '.' is
 #  open racetrack, and '#' is a wall. A user can choose between two crash scenarios:
 #  (1) race car crashes into wall and is placed at the nearest open spot on the
 #  racetrack with velocity = (0, 0) and (2) race car crashes into wall and is 
 #  returned to the starting line with velocity = (0, 0). A user can input the 
 #  desired number of training episodes at the command line. One hundred time trials
 #  are performed following training. The number of moves taken to cross the finish
 #  line are tracked and averaged over all of the time trials.
 #
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
import copy


# Receive the name of input file from the command line
input_file_name = sys.argv[1]
# User selects '1' for the soft crash condition and '2' for bad crash
selected_crash_option = int(sys.argv[2])
# User inputs the number of training episodes
number_training_episodes = sys.argv[3]


# This method loads the input file, which is one of the race tracks. It returns
# a list of lists containing the parsed track and the dimensions of the track.
def process_input_file( input_file_name ):
    y_axis_track = []
    with open(input_file_name, 'r') as full_track:
        for line in full_track:
            if line[:1].isdigit():
                dimensions = line
                y_dimension = dimensions.split(',')[0]
                x_dimension = dimensions.split(',')[1]
            else:
                x_axis_track = list( line )
                x_axis_elements = []
                for element in x_axis_track:
                    if element != '\n':
                        x_axis_elements.append( element )
                y_axis_track.append( x_axis_elements )
    return( y_axis_track, x_dimension, y_dimension )


# This method returns a list of coordinates corresponding to the location of the 
# track's starting line and finish line
def get_start_and_finish_line_coordinates( track ):
    finish_line_coordinates = []
    starting_line_coordinates = []
    # Iterate over each row of the track
    for row in range( 0, len(track) ):
        # Iterate over each element contained in the row
        for element in range( 0, len(track[row]) ):
            # Capital F is the finish line
            if track[row][element] == "F":
                finish_coordinate = ( element, row )
                finish_line_coordinates.append( finish_coordinate )
            if track[row][element] == "S":
                start_coordinate = ( element, row )
                starting_line_coordinates.append( start_coordinate )
    return( starting_line_coordinates, finish_line_coordinates )
    

# This method generates a dictionary for use in training that is indexed by
# position : velocity : action. All q values are initialized to random values 
# from a normal distribution with mean = 0 and stdev = 1. Q values for the
# finish line coordinates are initialized to 0.
def create_training_dictionary( x_dimension, y_dimension, finish_line_coordinates ):
    # Generate all possible positions of racetrack
    possible_positions = generate_possible_positions( x_dimension, y_dimension )

    # Generate possible actions that can be taken by the agent
    possible_acceleration_values = [-1, 0, 1]
    actions = generate_combinations( possible_acceleration_values )

    # Generate possible velocities that environment can take on
    possible_velocities = []
    for value_index in range(-5, 6):
        possible_velocities.append( value_index )
    velocities = generate_combinations( possible_velocities )
    
    # Create training dictionary and initialize q values
    position_dictionary_list = []
    
    for positions_index in range( 0, len(possible_positions) ):
        velocity_dictionary_list = []
        for velocities_index in range( 0, len(velocities) ):        
            q_values = []
            for i in range( 0, len(actions) ):
                q_value = np.random.normal(0, 1, 1)[0]
                q_values.append( q_value )
            actions_dictionary = dict( zip(actions, q_values) )
            velocity_dictionary_list.append( actions_dictionary )
        velocity_dictionary = dict( zip(velocities, velocity_dictionary_list) )
        position_dictionary_list.append( velocity_dictionary )
    training_dictionary = dict( zip(possible_positions, position_dictionary_list) )

    # Initialize the positions corresponding to the finish line to zero
    for items in range( 0, len(finish_line_coordinates) ):
        if finish_line_coordinates[items] in training_dictionary:
            for velocities in training_dictionary[ finish_line_coordinates[items] ]:
                for actions in training_dictionary[ finish_line_coordinates[items] ][velocities]:
                    training_dictionary[ finish_line_coordinates[items] ][velocities][actions] = 0
    
    return( training_dictionary )


# This method generates all 2-integer combinations of a list of values
# Returns a list of tuples in the form, for example, (acceleration in x, acceleration in y) 
def generate_combinations( values_to_combine ):
    list_of_combinations = []

    for i in range( 0, len(values_to_combine) ):
        for k in range( 0, len(values_to_combine) ):
            list_of_combinations.append( (values_to_combine[i], values_to_combine[k]) )
            
    return( list_of_combinations )

    
# Given the dimensions of an environment (race track), this method generates
# all possible coordinates for indexing. Returns list of tuples.
def generate_possible_positions( x_dimension, y_dimension ):    
    possible_x_values = []
    for x_value_index in range( 0, int(x_dimension) ):
        possible_x_values.append( x_value_index )
        
    possible_y_values = []
    for y_value_index in range( 0, int(y_dimension) ):
        possible_y_values.append( y_value_index )
    
    list_of_positions = []
    for i in range( 0, len(possible_x_values) ):
        for k in range( 0, len(possible_y_values) ):
            list_of_positions.append( (possible_x_values[i], possible_y_values[k]) )
            
    return( list_of_positions )


# This method resets the q values for the finish line coordinates to zero
def zero_finish_line_coordinates( training_dictionary, finish_line_coordinates ):
    # Initialize finish line coordinate Q values to zero
    for items in range( 0, len(finish_line_coordinates) ):
        if finish_line_coordinates[items] in training_dictionary:
            for velocities in training_dictionary[ finish_line_coordinates[items] ]:
                for actions in training_dictionary[ finish_line_coordinates[items] ][velocities]:
                    training_dictionary[ finish_line_coordinates[items] ][velocities][actions] = 0
    
    return( training_dictionary )


# This method randomly chooses a starting position and velocity during training
def randomly_choose_state( x_dimension, y_dimension, parsed_track ):

    # Randomly choose an x-coordinate position
    x_random = ( random.randint(x_dimension[0], x_dimension[1]) )
    
    # Randomly choose a y-coordinate position
    y_random = ( random.randint(y_dimension[0], y_dimension[1]) )
    
    if parsed_track[y_random][x_random] == "#":
        print("CUIDADO! CUIDADO! CUIDADO!")
    
    random_position = ( x_random, y_random )
    
    # Randomly choose an x-velocity
    x_velocity_random = ( random.randint( -5, 5 ) )
    
    # Randomly choose a y-velocity
    y_velocity_random = ( random.randint( -5, 5 ) )
    
    random_velocity = ( x_velocity_random, y_velocity_random )    
    return( random_position, random_velocity )
    
        

# This method calculates the velocity component of s'. Accounts for an
# acceleration failure occurring 20% of the time  
def calculate_state_s_prime( current_velocity, acceleration, current_position ):
    #print("s prime calculation, current_position: ", current_position)
    #print("acceleration: ", acceleration)
    # First calculate updated velocity
    acceleration_fail = random.random()
    #print("acceleration_fail: ", acceleration_fail)
    # If the random float is greater than 0.8, acceleration has failed
    
    if acceleration_fail > 0.8:
        acceleration = (0,0)

    minimum_velocity = -5
    maximum_velocity = 5
    
    x_velocity = current_velocity[0]
    y_velocity = current_velocity[1]
    
    x_acceleration = acceleration[0]
    y_acceleration = acceleration[1]
    
    updated_x_velocity = x_velocity + x_acceleration
    # Check whether updated velocity is within limits
    if updated_x_velocity > maximum_velocity:
        updated_x_velocity = x_velocity
    if updated_x_velocity < minimum_velocity:
        updated_x_velocity = x_velocity
            
    updated_y_velocity = y_velocity + y_acceleration
    if updated_y_velocity > maximum_velocity:
        updated_y_velocity = y_velocity
    if updated_y_velocity < minimum_velocity:
        updated_y_velocity = y_velocity
        
    updated_velocity = ( updated_x_velocity, updated_y_velocity )    
        
    # Now calculate updated position
    x_position = current_position[0]
    y_position = current_position[1]
    
    x_velocity = updated_velocity[0]
    y_velocity = updated_velocity[1]
    
    updated_x_position = x_position + x_velocity
    updated_y_position = y_position + y_velocity
    
    updated_position = ( updated_x_position, updated_y_position )
    #print("s prime calculation, updated_position: ", updated_position)
    
    return( updated_velocity, updated_position )


# When a new position on the track has been determined, this method determines
# the symbol associated with that location on the race track map 
def check_racetrack_location( this_position, parsed_track ):
    #print("checking the racetrack location at: ", this_position)
    x_coordinate = this_position[0]
    y_coordinate = this_position[1]
    location_on_track = parsed_track[y_coordinate][x_coordinate]
    #print("this is the location: ", location_on_track)
    return( location_on_track )

    
# This method pulls information pertinent to the L race track including coordinates
# corresponding to chunks of the track coordinates so training can proceed using
# backups. Also returns coordinates of the wall behind the finish line.    
def get_L_track_specifications( ):
    # For doing backups, this is for the starting positions
    backup_coordinate_list = []

    x_chunk_of_track = (32, 35)
    y_chunk_of_track = (2, 3)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (32, 35)
    y_chunk_of_track = (4, 5)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )

    x_chunk_of_track = (32, 35)
    y_chunk_of_track = (6, 9)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )

    even_x_coordinates_for_selection = []
    for even_x_chunk_index in range( 32, 0, -2 ):
        even_x_coordinates_for_selection.append( even_x_chunk_index )

    odd_x_coordinates_for_selection = []
    for odd_x_chunk_index in range( 31, 0, -2 ):
        odd_x_coordinates_for_selection.append( odd_x_chunk_index )

    list_of_combinations = []
    for combo_index in range( 0, len(even_x_coordinates_for_selection)):
        combination = (odd_x_coordinates_for_selection[combo_index], even_x_coordinates_for_selection[combo_index])
        list_of_combinations.append(combination)
    
    y_chunk_of_track = (6, 9)

    for x_y_combo_index in range( 0, len(list_of_combinations) ):
        x_y_combination = [list_of_combinations[x_y_combo_index], y_chunk_of_track]
        backup_coordinate_list.append( x_y_combination )
    
    x_chunk_of_track = (1, 2)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )

    wall_behind_finish_coordinates = [(32, 0), (33, 0), (34, 0), (35, 0)]      
    return( backup_coordinate_list, wall_behind_finish_coordinates )


# This method pulls information pertinent to the O race track including coordinates
# corresponding to chunks of the track coordinates so training can proceed using
# backups. Also returns coordinates of the wall behind the finish line.       
def get_O_track_specifications( ):
    # For doing backups, this is for the starting positions
    backup_coordinate_list = [ ]
    
    x_chunk_of_track = (1, 4)

    even_y_coordinates_for_selection = []
    for even_y_chunk_index in range( 12, 20, 2 ):
        even_y_coordinates_for_selection.append( even_y_chunk_index )

    odd_y_coordinates_for_selection = []
    for odd_y_chunk_index in range( 13, 21, 2 ):
        odd_y_coordinates_for_selection.append( odd_y_chunk_index )

    list_of_combinations = []
    for combo_index in range( 0, len(even_y_coordinates_for_selection) ):
        combination = (even_y_coordinates_for_selection[combo_index], odd_y_coordinates_for_selection[combo_index])
        list_of_combinations.append(combination)

    for x_y_combo_index in range( 0, len(list_of_combinations) ):
        x_y_combination = [x_chunk_of_track, list_of_combinations[x_y_combo_index]]
        backup_coordinate_list.append( x_y_combination )
    
    x_chunk_of_track = (2, 5)
    y_chunk_of_track = (20, 20)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )

    x_chunk_of_track = (3, 6)
    y_chunk_of_track = (21, 21)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )

    x_chunk_of_track = (3, 6)
    y_chunk_of_track = (22, 22)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )

    x_chunk_of_track = (4, 6)
    y_chunk_of_track = (23, 23)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )

    y_chunk_of_track = (22, 23)

    even_x_coordinates_for_selection = []
    for even_x_chunk_index in range( 8, 22, 2 ):
        even_x_coordinates_for_selection.append( even_x_chunk_index )

    odd_x_coordinates_for_selection = []
    for odd_x_chunk_index in range( 7, 23, 2 ):
        odd_x_coordinates_for_selection.append( odd_x_chunk_index )

    list_of_combinations = []
    for combo_index in range( 0, len(even_x_coordinates_for_selection) ):
        combination = (odd_x_coordinates_for_selection[combo_index], even_x_coordinates_for_selection[combo_index])
        list_of_combinations.append(combination)

    for x_y_combo_index in range( 0, len(list_of_combinations) ):
        x_y_combination = [list_of_combinations[x_y_combo_index], y_chunk_of_track]
        backup_coordinate_list.append( x_y_combination )

    x_chunk_of_track = (18, 21)
    y_chunk_of_track = (21, 21)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )

    x_chunk_of_track = (19, 22)
    y_chunk_of_track = (20, 20)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )


    x_chunk_of_track = (20, 23)

    even_y_coordinates_for_selection = []
    for even_y_chunk_index in range( 18, 4, -2 ):
        even_y_coordinates_for_selection.append( even_y_chunk_index )

    odd_y_coordinates_for_selection = []
    for odd_y_chunk_index in range( 19, 3, -2 ):
        odd_y_coordinates_for_selection.append( odd_y_chunk_index )

    list_of_combinations = []
    for combo_index in range( 0, len(even_y_coordinates_for_selection) ):
        combination = (even_y_coordinates_for_selection[combo_index], odd_y_coordinates_for_selection[combo_index])
        list_of_combinations.append(combination)

    for x_y_combo_index in range( 0, len(list_of_combinations) ):
        x_y_combination = [x_chunk_of_track, list_of_combinations[x_y_combo_index]]
        backup_coordinate_list.append( x_y_combination )
    
    x_chunk_of_track = (18, 22)
    y_chunk_of_track = (3, 4)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )


    y_chunk_of_track = (1, 2)

    even_x_coordinates_for_selection = []
    for even_x_chunk_index in range( 20, 2, -2 ):
        even_x_coordinates_for_selection.append( even_x_chunk_index )

    odd_x_coordinates_for_selection = []
    for odd_x_chunk_index in range( 21, 3, -2 ):
        odd_x_coordinates_for_selection.append( odd_x_chunk_index )

    list_of_combinations = []
    for combo_index in range( 0, len(even_x_coordinates_for_selection) ):
        combination = (even_x_coordinates_for_selection[combo_index], odd_x_coordinates_for_selection[combo_index])
        list_of_combinations.append(combination)

    for x_y_combo_index in range( 0, len(list_of_combinations) ):
        x_y_combination = [list_of_combinations[x_y_combo_index], y_chunk_of_track]
        backup_coordinate_list.append( x_y_combination )
    
    x_chunk_of_track = (18, 22)
    y_chunk_of_track = (2, 6)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )

    x_chunk_of_track = (1, 4)

    even_y_coordinates_for_selection = []
    for even_y_chunk_index in range( 6, 12, 2 ):
        even_y_coordinates_for_selection.append( even_y_chunk_index )

    odd_y_coordinates_for_selection = []
    for odd_y_chunk_index in range( 5, 11, 2 ):
        odd_y_coordinates_for_selection.append( odd_y_chunk_index )

    list_of_combinations = []
    for combo_index in range( 0, len(even_y_coordinates_for_selection) ):
        combination = (odd_y_coordinates_for_selection[combo_index], even_y_coordinates_for_selection[combo_index])
        list_of_combinations.append(combination)

    for x_y_combo_index in range( 0, len(list_of_combinations) ):
        x_y_combination = [x_chunk_of_track, list_of_combinations[x_y_combo_index]]
        backup_coordinate_list.append( x_y_combination )

    wall_behind_finish_coordinates = [(1, 11), (2, 11), (3, 11), (4, 11)]      
    return( backup_coordinate_list, wall_behind_finish_coordinates )
    

# This method pulls information pertinent to the R race track including coordinates
# corresponding to chunks of the track coordinates so training can proceed using
# backups. Also returns coordinates of the wall behind the finish line.    
def get_R_track_specifications( ):
    # For doing backups, this is for the starting positions
    backup_coordinate_list = []

    x_chunk_of_track = (24, 28)
    y_chunk_of_track = (24, 25)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (23, 27)
    y_chunk_of_track = (22, 23)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (23, 27)
    y_chunk_of_track = (20, 21)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (23, 27)
    y_chunk_of_track = (17, 19)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (21, 27)
    y_chunk_of_track = (16, 16)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (19, 23)
    y_chunk_of_track = (15, 15)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (16, 20)
    y_chunk_of_track = (14, 14)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (14, 18)
    y_chunk_of_track = (13, 13)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (13, 17)
    y_chunk_of_track = (12, 12)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (12, 16)
    y_chunk_of_track = (11, 11)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (10, 14)
    y_chunk_of_track = (10, 10)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (12, 16)
    y_chunk_of_track = (9, 9)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (14, 18)
    y_chunk_of_track = (8, 8)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (15, 19)
    y_chunk_of_track = (7, 7)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (17, 21)
    y_chunk_of_track = (6, 6)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (19, 23)
    y_chunk_of_track = (5, 5)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (18, 24)
    y_chunk_of_track = (4, 4)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (13, 26)
    y_chunk_of_track = (3, 3)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (13, 25)
    y_chunk_of_track = (2, 2)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (15, 21)
    y_chunk_of_track = (1, 1)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (9, 15)
    y_chunk_of_track = (1, 1)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (5, 12)
    y_chunk_of_track = (2, 2)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (3, 12)
    y_chunk_of_track = (3, 3)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 7)
    y_chunk_of_track = (4, 4)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (5, 5)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (6, 6)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (7, 7)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (8, 8)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (9, 9)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (10, 10)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (11, 11)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (1, 5)
    y_chunk_of_track = (12, 12)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (1, 5)
    y_chunk_of_track = (13, 13)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (1, 5)
    y_chunk_of_track = (14, 14)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (1, 5)
    y_chunk_of_track = (15, 15)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (16, 16)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (17, 17)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (18, 18)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (19, 19)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (2, 6)
    y_chunk_of_track = (20, 20)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (1, 5)
    y_chunk_of_track = (21, 21)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (1, 5)
    y_chunk_of_track = (22, 22)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (1, 5)
    y_chunk_of_track = (23, 23)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (1, 5)
    y_chunk_of_track = (24, 24)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
    
    x_chunk_of_track = (1, 5)
    y_chunk_of_track = (25, 25)
    this_chunk = [x_chunk_of_track, y_chunk_of_track]
    backup_coordinate_list.append( this_chunk )
        
    wall_behind_finish_coordinates = [(24, 27), (25, 27), (26, 27), (27, 27), (28, 27)]      
    return( backup_coordinate_list, wall_behind_finish_coordinates )
    

# This method handles the cases where the race car has hit a wall and crashed. There
# are two options: (1) race car crashes into wall and is placed at the nearest open spot 
# on the racetrack with velocity = (0, 0) and (2) race car crashes into wall and is
# returned to the starting line with velocity = (0, 0).
def handle_crash( updated_position, parsed_track, crash_option, starting_line_coordinates ):
    #print("there was a crash at this updated_position: ", updated_position)
    
    # Handle the milder version of a crash (car placed at nearest open spot on track)
    if crash_option == 1:
        x_updated = updated_position[0]
        y_updated = updated_position[1]
    
        open_cell_list = []
        for row_index in range( 0, len(parsed_track) ):
            for element_index in range( 0, len(parsed_track[row_index]) ):
                if parsed_track[row_index][element_index] == ".":
                    open_coordinate = (element_index, row_index)
                    open_cell_list.append( open_coordinate )
    
        distances = []
        for open_index in range( 0, len(open_cell_list) ):
            x_open = open_cell_list[open_index][0]
            y_open = open_cell_list[open_index][1]
        
            distance = math.sqrt( (( x_open - x_updated ) ** 2) + (( y_open - y_updated ) ** 2) )
            distances.append( distance )
        #print("Distances: ", distances)
        nearest_open_cell_index = distances.index( min(distances) )
        #print("nearest_open_cell_index: ", nearest_open_cell_index)
        new_position = open_cell_list[nearest_open_cell_index]
        #print("new_position: ", new_position)
    
    # Handle the crash condition in which the car is placed at the starting line
    else:
        starting_index = random.randint( 0, (len(starting_line_coordinates)-1) )
        new_position = starting_line_coordinates[starting_index]    
    return( new_position )
    
# This method implements Bresenham's line generation algorithm. It is used to ensure
# that the race car stays on the track as it travels and, if it crashes, will return
# the coordinate at which the car intersects with the wall.     
def bresenham_algorithm( current_coordinates, new_coordinates, parsed_track ):  
    
    x_1 = current_coordinates[0]
    y_1 = current_coordinates[1]

    x_2 = new_coordinates[0]
    y_2 = new_coordinates[1]
        
    # if x2, y2 are within the bounds of the track
    if (x_2 >= 0) and (x_2 < int(x_dimension)) and (y_2 >= 0) and (y_2 < int(y_dimension)):
        #print("I am in bounds")
        if (y_2 < y_1) and (x_2 < x_1):
            #print("y1 is greater than y2 and x1 is greater than x2")
            if ((x_2 - x_1) == 0):
                slope = 1
            else:
                slope = abs((y_2 - y_1) / (x_2 - x_1))
            m_new = 2*(y_1 - y_2)
            slope_error_new = m_new - (x_2 - x_1)

            y = y_1
            intervening_coordinates = []
            
            for x_index in range( x_1, (x_2-1), -1 ):
                these_coordinates = ( x_index, y )
                if these_coordinates != current_coordinates:
                    intervening_coordinates.append( these_coordinates )
                if (slope_error_new >= 0):  
                    y = y - 1
        
                    if y < y_2:
                        break
                    slope_error_new = slope_error_new - 2*slope
            if new_coordinates != these_coordinates:
                intervening_coordinates.append( (x_2, y_2) )
                    
        if (y_2 >= y_1) and (x_2 < x_1):
            #print("y2 is greater than or equal to y1 and x1 is greater than x2")
            if ((x_2 - x_1) == 0):
                slope = 1
            else:
                slope = abs((y_2 - y_1) / (x_2 - x_1))
            m_new = 2*(y_2 - y_1)
            slope_error_new = m_new - (x_2 - x_1)

            y = y_1
            intervening_coordinates = []
            
            for x_index in range( x_1, (x_2-1), -1 ):
                these_coordinates = ( x_index, y )
                if these_coordinates != current_coordinates:
                    intervening_coordinates.append( these_coordinates )
                if (slope_error_new >= 0):  
                    y = y + 1
        
                    if y > y_2:
                        break
                    slope_error_new =slope_error_new - 2*slope

            if new_coordinates != these_coordinates:
                intervening_coordinates.append( (x_2, y_2) )
                    
                    
        if (y_2 >= y_1) and (x_2 >= x_1):
            if ((x_2 - x_1) == 0):
                slope = 1
            else:
                slope = abs((y_2 - y_1) / (x_2 - x_1))
            m_new = 2*(y_2 - y_1)
            slope_error_new = m_new - (x_2 - x_1)

            y = y_1
            intervening_coordinates = []
            
            for x_index in range( x_1, (x_2+1) ):
                these_coordinates = ( x_index, y )
                if these_coordinates != current_coordinates:
                    intervening_coordinates.append( these_coordinates )
                if (slope_error_new >= 0):  
                    y = y + 1
        
                    if y > y_2:
                        break
                    slope_error_new =slope_error_new - 2*slope

            if new_coordinates != these_coordinates:
                intervening_coordinates.append( (x_2, y_2) )
                
        
        if (y_2 < y_1) and (x_2 >= x_1):
            if ((x_2 - x_1) == 0):
                slope = 1
            else:
                slope = abs((y_2 - y_1) / (x_2 - x_1))
            m_new = 2*(y_1 - y_2)
            slope_error_new = m_new - (x_2 - x_1)

            y = y_1
            intervening_coordinates = []
            
            for x_index in range( x_1, (x_2+1) ):
                these_coordinates = ( x_index, y )
                if these_coordinates != current_coordinates:
                    intervening_coordinates.append( these_coordinates )
                if (slope_error_new >= 0):  
                    y = y - 1
                    if y < y_2:
                        break
                    slope_error_new =slope_error_new - 2*slope

            if new_coordinates != these_coordinates:
                intervening_coordinates.append( (x_2, y_2) )
    
    # new x, new y, or both are outside the racetrack boundary
    else:
        #print("one or more of me are out of bounds")
        if (y_2 < y_1) and (x_2 < x_1):
            if ((x_2 - x_1) == 0):
                slope = 1
            else:
                slope = abs((y_2 - y_1) / (x_2 - x_1))
            m_new = 2*(y_1 - y_2)
            slope_error_new = m_new - (x_2 - x_1)

            y = y_1
            intervening_coordinates = []
            
            for x_index in range( x_1, (x_2-1), -1 ):
                these_coordinates = ( x_index, y )
                if (x_index >= 0) and (x_index < int(x_dimension)) and (y >= 0) and (y < int(y_dimension)) and (these_coordinates != current_coordinates):
                    intervening_coordinates.append( these_coordinates )
                if (slope_error_new >= 0):  
                    y = y - 1
        
                    if y < y_2:
                        break
                    slope_error_new = slope_error_new - 2*slope
                    
        if (y_2 >= y_1) and (x_2 < x_1):
            if ((x_2 - x_1) == 0):
                slope = 1
            else:
                slope = abs((y_2 - y_1) / (x_2 - x_1))
            m_new = 2*(y_2 - y_1)
            slope_error_new = m_new + (x_2 - x_1)

            y = y_1
            intervening_coordinates = []
            
            for x_index in range( x_1, (x_2-1), -1 ):
                these_coordinates = ( x_index, y )
                if (x_index >= 0) and (x_index < int(x_dimension)) and (y >= 0) and (y < int(y_dimension)) and (these_coordinates != current_coordinates):
                    intervening_coordinates.append( these_coordinates )
                if (slope_error_new >= 0):  
                    y = y + 1
        
                    if y > y_2:
                        break
                    slope_error_new =slope_error_new - 2*slope
                    
                    
        if (y_2 >= y_1) and (x_2 >= x_1):
            if ((x_2 - x_1) == 0):
                slope = 1
            else:
                slope = abs((y_2 - y_1) / (x_2 - x_1))
            m_new = 2*(y_2 - y_1)
            slope_error_new = m_new - (x_2 - x_1)

            y = y_1
            intervening_coordinates = []
            
            if ((x_2 - x_1) == 0):
                for x_index in range( x_1, (x_2+2) ):
                    these_coordinates = ( x_index, y )
                    #print("these_coordinates: ")
                    if (x_index >= 0) and (x_index < int(x_dimension)) and (y >= 0) and (y < int(y_dimension)) and (these_coordinates != current_coordinates):
                        intervening_coordinates.append( these_coordinates )
                    #print("slope_error_new: ", slope_error_new)
                    if (slope_error_new >= 0):  
                        y = y + 1
                        if y > y_2:
                            break
                        slope_error_new =slope_error_new - 2*slope
            
            else:
                for x_index in range( x_1, (x_2+1) ):
                    these_coordinates = ( x_index, y )
                    #print("these_coordinates: ")
                    if (x_index >= 0) and (x_index < int(x_dimension)) and (y >= 0) and (y < int(y_dimension)) and (these_coordinates != current_coordinates):
                        intervening_coordinates.append( these_coordinates )
                    #print("slope_error_new: ", slope_error_new)
                    if (slope_error_new >= 0):  
                        y = y + 1
        
                        if y > y_2:
                            break
                        slope_error_new =slope_error_new - 2*slope
                
        
        if (y_2 < y_1) and (x_2 >= x_1):
            if ((x_2 - x_1) == 0):
                slope = 1
            else:
                slope = abs((y_2 - y_1) / (x_2 - x_1))
            m_new = 2*(y_1 - y_2)
            slope_error_new = m_new - (x_2 - x_1)

            y = y_1
            intervening_coordinates = []
            
            if ((x_2 - x_1) == 0):
                for x_index in range( x_1, (x_2+2) ):
                    these_coordinates = ( x_index, y )
                    if (x_index >= 0) and (x_index < int(x_dimension)) and (y >= 0) and (y < int(y_dimension)) and (these_coordinates != current_coordinates):
                        intervening_coordinates.append( these_coordinates )
                    if (slope_error_new >= 0):  
                        y = y - 1
                        if y < y_2:
                            break
                        slope_error_new =slope_error_new - 2*slope
            
            else:
                for x_index in range( x_1, (x_2+1) ):
                    these_coordinates = ( x_index, y )
                    if (x_index >= 0) and (x_index < int(x_dimension)) and (y >= 0) and (y < int(y_dimension)) and (these_coordinates != current_coordinates):
                        intervening_coordinates.append( these_coordinates )
                    if (slope_error_new >= 0):  
                        y = y - 1
                        if y < y_2:
                            break
                        slope_error_new =slope_error_new - 2*slope

    #print("intervening coordinates: ", intervening_coordinates)
        
    return( intervening_coordinates )

# This method determines the next action for the race car to take using an epsilon 
# greedy approach.     
def determine_action( this_position, this_velocity, epsilon ):
    # Locate starting state and q values in the training dictionary
    for positions in training_dictionary:
        if positions == this_position:
            for velocities in training_dictionary[positions]:
                if velocities == this_velocity:
                    list_of_actions = []
                    list_of_q_values = []
                    for actions in training_dictionary[positions][velocities]:
                        list_of_actions.append( actions )
                        list_of_q_values.append( training_dictionary[positions][velocities][actions] )
    #print("\n\nlist_of_q_values: ", list_of_q_values)
            
    # Select action using epsilon greedy strategy
    choose_max_q_or_random = random.random()           
    # If the random float is less than 1-epsilon, choose the action corresponding
    # to the maximum q value
    if choose_max_q_or_random <= ( 1 - epsilon ):
        this_q_value = max( list_of_q_values )
        largest_q_index = list_of_q_values.index( max(list_of_q_values) )
        action_to_take = list_of_actions[ largest_q_index ]
    # If the random float is greater than 0.7, choose the action corresponding
    # to a random q value, 30% of the time choose random q value
    else:
        random_index = random.randint( 0, len(list_of_q_values)-1 )
        this_q_value = list_of_q_values[ random_index ]
        action_to_take = list_of_actions[ random_index ]
    #print("This is the action to take: ", action_to_take)
    
    return( action_to_take, this_q_value )


# This method implements the SARSA algorithm for training the agent.        
def train_agent( training_dictionary, wall_behind_finish_coordinates, x_chunk_of_track, y_chunk_of_track, starting_line_coordinates, finish_line_coordinates, parsed_track, crash_option, number_training_episodes, final_training ):
    learning_rate = 0.3
    discount = 0.9
    reward = -1
    
    x_chunk_indices = []
    for x_chunk_index in range( x_chunk_of_track[0], (x_chunk_of_track[1] + 1) ):
        x_chunk_indices.append( x_chunk_index )
    y_chunk_indices = []
    for y_chunk_index in range( y_chunk_of_track[0], (y_chunk_of_track[1] + 1) ):
        y_chunk_indices.append( y_chunk_index )
    
    attempts_for_finish = []
    for episode_index in range( 0, int(number_training_episodes) ):
        print("number_training_episodes: ", number_training_episodes)
        # Re-zero the finish line q values
        training_dictionary = zero_finish_line_coordinates( training_dictionary, finish_line_coordinates )
        # Pick random position and random velocity
        this_position, this_velocity = randomly_choose_state( x_chunk_of_track, y_chunk_of_track, parsed_track )
        print("\n\n\n\n\n\n\nStarting position of the car for this episode: ", this_position)
        print("This is episode: ", episode_index)
        
        
        #####Pick the first action using epsilon greedy
        if (this_position[0] in x_chunk_indices) and (this_position[1] in y_chunk_indices):
            action_to_take, this_q_value = determine_action( this_position, this_velocity, 0.5 )
        else:
            action_to_take, this_q_value = determine_action( this_position, this_velocity, 0.1 )
        
        number_iterations = 0
        for iteration_index in range( 0, 10000 ):
             
            # Now take the action by calculating new velocity and new position
            updated_velocity, updated_position = calculate_state_s_prime( this_velocity, action_to_take, this_position )
            # Obtain the path that the race car took
            if this_position != updated_position:
                intervening_coordinates = bresenham_algorithm( this_position, updated_position, parsed_track )
                # Evaluate that path    
                track_positions = []        
                for intervening_index in range( 0, len(intervening_coordinates) ):
                    track_location = check_racetrack_location( intervening_coordinates[intervening_index], parsed_track )
                    track_positions.append( track_location )
        
                # Evaluate the positions based on the race track symbols
                track_index = 0
                while track_index < len(track_positions):
                    if track_positions[track_index] == ".":
                        updated_position = intervening_coordinates[track_index]
                        track_index = track_index + 1
                    elif track_positions[track_index] == "F":
                        print("Crossed the finish!\n\n\n\n\n\n")
                        updated_position = (-1, -1)
                        break
                    elif track_positions[track_index] == "S":
                        if (track_index != 0) and (track_positions[track_index-1] == "."):
                            updated_position = intervening_coordinates[track_index-1]
                            updated_velocity = (0,0)
                            break
                        elif track_index == 0:
                            updated_position = intervening_coordinates[track_index]
                            updated_velocity = (0,0)
                            break
                    elif track_positions[track_index] == "#":
                        #print("A crash occurred")
                        updated_position = handle_crash( intervening_coordinates[track_index], parsed_track, crash_option, starting_line_coordinates )
                        updated_velocity = (0,0)
                        break
            
            if updated_position != (-1, -1):
                # select a prime using epsilon greedy strategy
                if (updated_position[0] in x_chunk_indices) and (updated_position[1] in y_chunk_indices):
                    a_prime_action_to_take, this_q_value_for_a_prime = determine_action( updated_position, updated_velocity, 0.5 )
                else:
                    a_prime_action_to_take, this_q_value_for_a_prime = determine_action( updated_position, updated_velocity, 0.1 )
            
                # Now update the q values
                for positions in training_dictionary:
                    if positions == updated_position:
                        for velocities in training_dictionary[positions]:
                            if velocities == updated_velocity:
                                list_of_actions = []
                                list_of_q_values = []
                                for actions in training_dictionary[positions][velocities]:
                                    if actions == a_prime_action_to_take:
                                        on_policy_q_value = training_dictionary[positions][velocities][actions]
                                    
                                
                training_dictionary[positions][velocities][actions] = (( 1 - learning_rate ) * this_q_value) + ( learning_rate * (reward + (discount * on_policy_q_value)) )
                
                # Updated state becomes the current state for the next iteration
                this_position, this_velocity = updated_position, updated_velocity
                action_to_take = a_prime_action_to_take
                number_iterations = number_iterations + 1
                
            else:
                break
        print("number_iterations: ", number_iterations)
        attempts_for_finish.append( number_iterations )
    
    # Re-zero the finish line q values before returning
    training_dictionary = zero_finish_line_coordinates( training_dictionary, finish_line_coordinates )
    
    return( training_dictionary )


# This method runs a time trial for the trained agent. 
def time_trial( training_dictionary, parsed_track, crash_option, starting_line_coordinates ):
    
    this_velocity = (0,0)
    starting_index = random.randint( 0, len(starting_line_coordinates)-1 )
    this_position = starting_line_coordinates[starting_index]
    
    print("\n\n\n\nHere is the starting position: ", this_position)
    
    number_of_moves = 0
    
    for iteration_index in range(0, 20000):
        
        action_to_take, this_q_value = determine_action( this_position, this_velocity, 0.1 )
                
        # Now take the action by calculating new velocity and new position
        updated_velocity, updated_position = calculate_state_s_prime( this_velocity, action_to_take, this_position )
        number_of_moves = number_of_moves + 1
        # Obtain the path that the race car took
        if this_position != updated_position:
            intervening_coordinates = bresenham_algorithm( this_position, updated_position, parsed_track )
            # Evaluate that path    
            track_positions = []        
            for intervening_index in range( 0, len(intervening_coordinates) ):
                track_location = check_racetrack_location( intervening_coordinates[intervening_index], parsed_track )
                track_positions.append( track_location )
        
            #print("track_positions: ", track_positions)
                
            # Evaluate the positions based on the race track symbols
            track_index = 0
            while track_index < len(track_positions):
                if track_positions[track_index] == ".":
                    updated_position = intervening_coordinates[track_index]
                    track_index = track_index + 1
                elif track_positions[track_index] == "F":
                    print("Crossed the finish!")
                    updated_position = (-1, -1)
                    return( number_of_moves )
                    break
                elif track_positions[track_index] == "S":
                    if (track_index != 0) and (track_positions[track_index-1] == "."):
                        updated_position = intervening_coordinates[track_index-1]
                        break
                    elif track_index == 0:
                        updated_position = intervening_coordinates[track_index]
                        break
                elif track_positions[track_index] == "#":
                    #print("A crash occurred")
                    updated_position = handle_crash( intervening_coordinates[track_index], parsed_track, crash_option, starting_line_coordinates )
                    break
        
        if updated_position == (-1, -1):
            break
        else:
            # Updated state becomes the current state for the next iteration
            this_position, this_velocity = updated_position, updated_velocity
    
    return( number_of_moves )
            

################################################
############## Main Driver #####################
################################################   

# Parse input file name for output file
split_input_path = input_file_name.strip().split("/")
split_input_file_name = split_input_path[7].split("_")
split_track_name = split_input_file_name[0].split("-")
output_file_name_final = split_track_name[0]

# Load input file and determine dimensions of the track
parsed_track, x_dimension, y_dimension = process_input_file( input_file_name )
# Identify the coordinates of the start and finish line
starting_line_coordinates, finish_line_coordinates = get_start_and_finish_line_coordinates( parsed_track )
# Initialize a dictionary to store q values during training
training_dictionary = create_training_dictionary( x_dimension, y_dimension, finish_line_coordinates )
# Initialize a dictionary to store q values during training
#policy_dictionary = create_policy_dictionary( x_dimension, y_dimension, finish_line_coordinates )

if split_track_name[0] == "L":
    backup_coordinate_list, wall_behind_finish_coordinates = get_L_track_specifications( )
if split_track_name[0] == "O":
    backup_coordinate_list, wall_behind_finish_coordinates = get_O_track_specifications( )
if split_track_name[0] == "R":
    backup_coordinate_list, wall_behind_finish_coordinates = get_R_track_specifications( )
    

for starting_index in range( 0, len(backup_coordinate_list) ):
    x_chunk_of_track = backup_coordinate_list[starting_index][0]
    y_chunk_of_track = backup_coordinate_list[starting_index][1]
    crash_option = 1
    final_training = 0
    if starting_index == len(backup_coordinate_list) - 1:
        crash_option = selected_crash_option
        final_training = 1
    
    training_dictionary = train_agent( training_dictionary, wall_behind_finish_coordinates, x_chunk_of_track, y_chunk_of_track, starting_line_coordinates, finish_line_coordinates, parsed_track, crash_option, number_training_episodes, final_training )


with open ("/Users/michelleamaral/Desktop/Intro_to_Machine_Learning/Programming_Project_6/Outputs/" + output_file_name_final + "_track_output_SARSA_" + number_training_episodes + "_training_episodes", 'w') as outputFile:

    outputFile.write( "\nTrack: " + output_file_name_final + "\n" )
    outputFile.write( "Algorithm: SARSA" + '\n')
    outputFile.write( "Number of training episodes: " + str(number_training_episodes) + "\n" )
    outputFile.write( "Crash scenario: " + str(selected_crash_option) + "\n" )
    outputFile.write( "\n*************************\n" )
    outputFile.write( " Time trial performances" )
    outputFile.write( "\n*************************\n" )
    
    performance_list = []
    for races_index in range(0, 100):
        number_of_moves = time_trial( training_dictionary, parsed_track, selected_crash_option, starting_line_coordinates )
        outputFile.write( "Time trial " + str(races_index) + ":  " + str(number_of_moves) + "\n" )
        performance_list.append( number_of_moves )
        print("number_of_moves: ", number_of_moves)
    
    average_performance = np.mean( performance_list )
    outputFile.write( "\n\nAverage number of moves to complete time trial: " + str(round( average_performance, 3 )) + "\n" )
    std_performances = np.std( performance_list )
    outputFile.write( "\nStandard deviation: " + str(round( std_performances, 3 )) + "\n\n\n" )
    
    outputFile.write( "Race cars were provided with a maximum of 20,000 moves to complete time trial. " + "\n" )
    
    outputFile.write( "\n\nAll outputs, raw: " + str(performance_list) + "\n" )
    
    print("All performances: ", performance_list )
    print("average_performance: ", average_performance)
    print("std_performances: ", std_performances)




