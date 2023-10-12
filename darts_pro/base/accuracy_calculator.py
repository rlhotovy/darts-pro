import math

active_calibration_session = True
throw_num = 1
# store throw results for eventual plotting - calculations for std. dev. could be rewritten to iterate through the x/y_coord key values here
throw_results = []
# store x coordinate results from throws - will eventually want to change this to the distance between the thrown dart and intended target
x_results = []
# store x coordinate results from throws - will eventually want to change this to the distance between the thrown dart and intended target
y_results = []

# start the calibration session
# i attempted to have the "no" block of this loop set active_calibration_session to false and exit the session, but using break works for now
while active_calibration_session == True:
    throw_dart = input("Calibration session active. Throw dart? Y | N: ")
    if throw_dart.lower() == "y":
        print("Throw #{}".format(throw_num))

        # gather coordinate data from thrown dart (expected units from -1 to 1 as % of board radius)
        x_coord = input("Enter dart x coordinate: ")
        y_coord = input("Enter dart y coordinate: ")

        # construct dictionary containing throw data 
        throw_result = {"throw_num": throw_num, "x_coord": x_coord, "y_coord": y_coord}

        # add dictionary with throw data to throw_results list
        throw_results.append(throw_result)

        # add x and y coordinate data to lists to be used for calculating standard deviation
        # there's probably a way I can iterate through the coordinate keys in the throw_results dictionary, but this method was quicker to implement
        x_results.append((float(x_coord)))
        y_results.append((float(y_coord)))

        # Need more than one sample to calculate standard deviation
        if throw_num > 1:
            # Calculate std. dev. of x from center of target 0 instead of population mean
            x_n = len(x_results)
            sum_x = sum(x_results)
            stdev_x = math.sqrt((sum_x**2)/x_n)

            # Calculate std. dev. of y from center of target 0 instead of population mean
            y_n = len(x_results)
            sum_y = sum(y_results)
            stdev_y = math.sqrt((sum_y**2)/y_n)

            print("Current accuracy rating: {},{}".format(stdev_x, stdev_y))
        else:
            print("Not enough darts thrown to determine accuracy rating, please throw another dart")

        # Increment throw number    
        throw_num += 1

    # Allow user to exit calibration session    
    elif throw_dart.lower() == "n":
        print("Ending calibration session.")
        if throw_num > 2:
            # print("Final player calibration rating: {},{}".format(xcord_stdev, ycord_stdev))
            print("Final Updated accuracy rating: {},{}".format(stdev_x, stdev_y))
        else:
            print("Not enough darts thrown to determine accuracy rating")
        break

    # Continue the while loop if unexpected input is used
    else:
        print("Invalid input. Please use Y or N")
        continue
