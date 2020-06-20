import csv
import math
import os
# Bee Counting
# Author: Flurin Schwerzmann


'''
Parameters:
    csv_file_name: input file with all unfiltered events
    min_nest_time: time a bee has to stay inside a hole to qualify for residency in milliseconds
    min_fly_time: time a bee has to be in flight to qualify for residency in milliseconds
'''

def extract_agroscope_metrics(csv_file_name, output_folder, min_nest_time=40000, min_fly_time=40000):

    # VARIABLES
    
    data = []  # TIME, BEE, ACTION, HOLE
    multi_nests = []  # nests used by more than one bee
    multi_residents = []  # bees with more than one nest
    errors = []  # list of erroneous movements
    residential_movements = []  # list of movements qualifying for residency
    address_book = []  # list of nests used by no more than one bee
    boozer_book = []  # list with the number of holes a bee went into before finding it's nest
    flight_book = []  # list with duration of collecting flights
        
    debug_error_correction = False
    debug_find_residents = False
    debug_multi_residents = False
    debug_show_error_list = False
    
    
    # FUNCTIONS
    
    # correction principle: insert a fitting movement
    # used for hole-ordered search as well as bee-ordered search
    def correct_missing_movement(index, search_type):
        if debug_error_correction:
            print("--- Error Correction: " + search_type + " ---")
            print("Old:")
            print(data[index - 1])
            print(data[index])
            print(data[index + 1])
            print(data[index + 2])
    
        # enter: insert leave
        if data[index][2] == 'Enter':
            # the timestamp of the inserted leave is exactly 1 millisecond after
            hms = data[index + 1][0].split(':')
            new_seconds = float(hms[2]) + 0.001
            new_movement_timestamp = hms[0] + ':' + hms[1] + ':' + str(new_seconds)
            data[index + 1: index + 1] = [[new_movement_timestamp, data[index][1], 'Missing Leave', data[index][3]]]
    
        # leave: insert enter
        elif data[index][2] == 'Leave':
            # the timestamp of the inserted enter is exactly 1 millisecond before
            hms = data[index + 1][0].split(':')
            new_seconds = float(hms[2]) - 0.001
            new_movement_timestamp = hms[0] + ':' + hms[1] + ':' + str(new_seconds)
            data[index + 1: index + 1] = [
                [new_movement_timestamp, data[index + 1][1], 'Missing Enter', data[index + 1][3]]]
    
        if debug_error_correction:
            print("New:")
            print(data[index - 1])
            print(data[index])
            print(data[index + 1])
            print(data[index + 2])
            print(data[index + 3])
    
    
    # convert a timestamp to milliseconds
    def timestamp_to_millis(timestamp):
        hms = timestamp.split(':')
        return float(hms[0]) * 3.6E6 + float(hms[1]) * 60E3 + float(hms[2]) * 1000
    
    
    # convert milliseconds to minutes and seconds
    def millis_to_min_sec(millis):
        minutes = math.floor(millis / 60000)
        if minutes < 10:
            minutes = "0" + str(minutes)
    
        seconds = round((millis % 60000) / 1000)
        if seconds < 10:
            seconds = "0" + str(seconds)
    
        return [minutes, seconds]
    
    
    # take measurements of a valid movement qualified as residence
    # drunkenness: number of holes a bee flies to before finding it's nest
    # collection time: time between leaving home for a collection flight and finding home again afterwards
    def take_measurements(index, home, bee):
        time_leaving_home = data[index + 1][0]
        drunkenness = 0
        correction = 0
        # make sure to keep track on the same bee
        while data[index + (2 * drunkenness) + 2][1] == bee:
            # check if bee has found it's home (with valid enter movement)
            if data[index + (2 * drunkenness) + 2][3] == home and data[index + (2 * drunkenness) + 2][2] == 'Enter':
                # print(str(bee) + " flew to " + str(drunkenness) + " holes before finding it's nest " + str(home))
                boozer_book.append([bee, drunkenness - correction])
    
                time_back_home = data[index + (2 * drunkenness) + 2][0]
                collection_time = timestamp_to_millis(time_back_home) - timestamp_to_millis(time_leaving_home)
                flight_book.append([bee, millis_to_min_sec(collection_time)])
    
                return
    
            # make sure to correct for inserted missing movements
            if "Missing" in data[index + (2 * drunkenness) + 2][2] or "Missing" in data[index + (2 * drunkenness) + 3][2]:
                correction += 1
    
            drunkenness += 1
    
    
    # check if movement qualifies for confirmation of residence
    # if residence is detected, measurements are taken and the movement is registered as residential movement
    def check_resident(index):
        # residency check always starts with enter
        if data[index][2] == 'Enter':
            nest_time = timestamp_to_millis(data[index + 1][0]) - timestamp_to_millis(data[index][0])
            fly_time = timestamp_to_millis(data[index + 2][0]) - timestamp_to_millis(data[index + 1][0])
            # times must be great enough to qualify as a resident
            if nest_time > min_nest_time and fly_time > min_fly_time:
                # avoid corrected movements
                if data[index + 1][2] == 'Leave' and data[index + 2][2] == 'Enter':
                    if debug_find_residents:
                        print("Found resident: " + str(data[index]))
                    residential_movements.append([data[index][3], data[index][1]])
                    take_measurements(index, data[index][3], data[index][1])
                else:
                    if debug_find_residents:
                        print("Ignored corrected movement: " + str(data[index]))
        return
    
    
    # STATISTICS
    
    # read CSV
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            # print(f"At {row[0]}, bee {row[1]} did {row[2]} from/to nest {row[3]}.")
            if (row[1] == '?'): continue
            data.append([row[0], row[1], row[2], row[3]])
            # print(data[line_count])
            line_count += 1
        # print(f'Processed {line_count} lines.')
    
    # sort by holes
    data.sort(key=lambda x: x[3])
    
    # hole-based error check: check for double leave or enter of same bee on each hole
    nest_index = 0
    for line in data:
        next_line = data[nest_index + 1]
        # check if next line is still same nest
        if line[3] == next_line[3]:
            # check if leave after leave or enter after enter
            if line[2] == next_line[2]:
                # check if same bee (found an error!)
                if line[1] == next_line[1]:
                    errors.append([line, "unpaired " + str(line[2]) + " on this hole"])
                    correct_missing_movement(nest_index, "hole-based")
                # more than one bee in same nest (found a possible multi-nest!)
                # else:
                #     multi_nests.append(line)
        nest_index += 1
        if nest_index == len(data) - 1:
            break
    
    # sort again by time
    data.sort(key=lambda x: x[0])
    
    # sort by bee
    data.sort(key=lambda x: x[1])
    
    # bee-based error check: check if leave for every enter of each bee's movements exists
    bee_index = 0
    for line in data:
        next_line = data[bee_index + 1]
        # check if next line is still about the same bee
        if line[1] == next_line[1]:
            # check for enter without leave (and vice versa)
            if line[2] == next_line[2]:
                errors.append([line, "missing enter or leave of bee " + str(line[1])])
                correct_missing_movement(bee_index, "bee-based")
            # movement seems valid, check if resident
            # else:
            #     check_resident(bee_index)
        bee_index += 1
        if bee_index == len(data) - 2:
            break
    
    # find residences (a separate for loop is used for clarity)
    bee_index = 0
    for line in data:
        # check if next two lines are still about the same bee
        if line[1] == data[bee_index + 1][1] == data[bee_index + 2][1]:
            check_resident(bee_index)
        bee_index += 1
        if bee_index == len(data) - 2:
            break
    
    # sort  residential movements by nests to prepare for creation of address book
    residential_movements.sort(key=lambda x: x[0])
    
    # create address book from residential movements
    movement_index = 0
    for movement in residential_movements:
        # check for duplicate movements
        if residential_movements[movement_index + 1] != residential_movements[movement_index]:
            address_book.append(movement)
        movement_index += 1
        if movement_index == len(residential_movements) - 1:
            break
    
    # check if a nest is used by more than one bee
    address_index = 0
    for outer in address_book:
        address_count = 0
        # count how often address appears in address book
        for inner in address_book:
            if outer[0] == inner[0]:
                address_count += 1
        if address_count > 1:
            multi_nests.append(outer[0])
            multi_residents.append(outer[1])
        address_index += 1
        if address_index == len(residential_movements) - 1:
            break
    
    # nests used by more than one bee may not be counted for the address book
    for multi_nest in multi_nests:
        for address in address_book:
            if address[0] == multi_nest:
                if debug_multi_residents: print("removed an address due to multi resident: " + str(address))
                address_book.remove(address)
    
    # bees using a nest together with another bee may not be counted for flight book
    for multi_resident in multi_residents:
        for flight in flight_book:
            if flight[0] == multi_resident:
                if debug_multi_residents: print("removed a flight due to multi resident: " + str(flight))
                flight_book.remove(flight)
    
    # DATA OUTPUT
    
    if debug_show_error_list:
        print("\nErrors:")
        print(*errors, sep="\n")
    
    if debug_multi_residents:
        print("\nMulti nests:")
        print(*multi_nests, sep="\n")
    
    if debug_multi_residents:
        print("\nMulti residents:")
        print(*multi_residents, sep="\n")
    
    
    address_book_csv = os.path.join(output_folder,"address_book.csv")
    if os.path.exists(address_book_csv):
        os.remove(address_book_csv)
    with open(address_book_csv, 'a') as f:
        f.write("BEE,NEST\n")
        for address in address_book:
            f.write(str(address[0]) + ", " + address[1])
    
    boozer_book_csv = os.path.join(output_folder,"boozer_book.csv")
    if os.path.exists(boozer_book_csv):
        os.remove(boozer_book_csv)
    with open(boozer_book_csv, 'a') as f:
        f.write("BEE,HOLES\n")
        for shamble in boozer_book:
            f.write(str(shamble[0]) + ", " + str(shamble[1]))
    
    flight_list_csv = os.path.join(output_folder,"flight_list.csv")
    if os.path.exists(flight_list_csv):
        os.remove(flight_list_csv)
    with open(flight_list_csv, 'a') as f:
        f.write("TIME,BEE,DURATION\n")
        for flight in flight_book:
            f.write(str(flight[0]) + ", " + str(flight[1][0]) + ":" + str(flight[1][1]))
    