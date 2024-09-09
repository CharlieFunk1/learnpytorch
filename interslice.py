#Input and output:
#interslice(<text.file>) Takes a data or txt file as input 
#X_train, y_train, X_test, y_test, output_key= interslice(<text.file>)
#Outputs to 5 arrays^

import numpy as np
import random


#MODE TODO can either be random or linear.  Linear splits 20% of the data off the end for testing.  Random randomly selects 20% of the data for testing. This will eventually be an argument.
#MODE = "random"
#input_file = "iris.data"

def interslice(input_file):
    #Set up internal variables and arrays
    data_file = np.loadtxt(input_file, dtype=str, delimiter=",")
    number_of_rows = int(data_file.shape[0])
    number_of_columns = int(data_file.shape[1])
    rand_rows = []
    rows_left = list(range(0,number_of_rows))
    #Generate random list of rows = 20% of total rows
    i = 0
    while i < int(0.20 * number_of_rows):
        rand_number = random.randint(0,number_of_rows - 1)
        nope = 0
        for j in rand_rows:
            if rand_number == j:
                nope = 1
        if nope == 0:
            rand_rows.append(rand_number)
            i += 1

    #Delete randomly selected rows from training list (rows_left)
    for rand in rand_rows:
        rows_left.remove(rand)

    #Setup output arrays and lists    
    X_train = np.zeros(shape = (int(number_of_rows - (number_of_rows * 0.2)), (number_of_columns - 1)), dtype=float)
    X_test = np.zeros(shape = (int(number_of_rows - (number_of_rows * 0.8)), (number_of_columns - 1)), dtype=float)
    y_train = np.loadtxt(input_file, dtype=str, delimiter=",")
    y_test = np.loadtxt(input_file, dtype=str, delimiter=",")
    y_train_list = []
    y_test_list = []
    
    #Distributes data to X and Y train and test arrays
    i = 0
    for r in rand_rows:
        X_test[i] = data_file[[r],0:(number_of_columns - 1)]
        y_test_list.append(y_test[r][number_of_columns - 1])
        i += 1
        
    j = 0    
    for L in rows_left:
        X_train[j] = data_file[[L],0:(number_of_columns - 1)]
        y_train_list.append(y_train[L][number_of_columns - 1])
        j += 1
        
    #Converts training Y array from strings to ints and creates an output key.
    j = 0
    number_of_entries = 0
    list_of_outputs = []
    y_train_list_num = np.zeros(shape = (int(number_of_rows - (number_of_rows * 0.2)), 1), dtype=int)
    list_of_outputs.append([y_train_list[0], number_of_entries])
    while j < (len(y_train_list) - 1):
        #Converts train list to numbers
        y_train_list_num[j] = number_of_entries
        #Creates output key
        if list_of_outputs[number_of_entries][0] != y_train_list[(j + 1)]:
            list_of_outputs.append([y_train_list[j + 1], number_of_entries + 1])
            number_of_entries += 1
        j += 1
    #Fills in last entry because I cant be bothered
    y_train_list_num[j] = number_of_entries

    #Converts testing Y array from strings to ints.
    k = 0
    L = 0
    y_test_list_num = np.zeros(shape = (int(number_of_rows - (number_of_rows * 0.8)), 1), dtype=int)
    while k <= (len(y_test_list) - 1):
        L = 0 
        while L < len(list_of_outputs):
            if y_test_list[k] == list_of_outputs[L][0]:
                y_test_list_num[k] = list_of_outputs[L][1]
            L += 1
        k += 1
    #print(rand_rows) 
    #returns all stuff
    return X_train, y_train_list_num, X_test, y_test_list_num, list_of_outputs#, y_test_list


    #print(y_test_list_num)
    #print(list_of_outputs)
    
    #print(data_file[[0], :])
    #print(rand_rows)
    #print("****************************TRAIN************************")    
    #print(X_train)
    #print(y_train_list)
    #print(len(y_train_list))
    #print(y_train)
    #print("****************************TEST************************")    
    #print(X_test)
    #print(y_test)
    #print(y_test_list)
    #print(y_train_list)
#X_train_final, y_train_final, X_test_final, y_test_final, output_key, y_test_list_inter = interslice("iris.data")

#q = 0
#while q < 120:
#    print (X_train_final[q], y_train_final[q], q)
#    q += 1
#
#w = 0
#while w < 30:
#    print(X_test_final[w], y_test_final[w], w, y_test_list_inter[w])
#    w += 1

