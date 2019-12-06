#!/bin/bash

# Read the textfile which holds the arguments and then parse those arguments 
# into the run file for an algorithm
file="testbed.txt"

# it reads the file per line, IFS leaves in the whitespacing
# [[ -n "$Line" ]] -> this prevents the last line from being ignored"
# I set the IFS to tab spaces that splits based on tabs per line
while IFS=$'\t' read -r line  ;
	do 
		# get input arguments
		words=($line)
		if [ "${words[0]}" == "#" ]
		then
			echo 'Line Commented'
		else
			# print out the environment
			echo " "
			echo 'Inputting argument parameters'
			echo "Now Running ${words[0]} environment"
			
			# Running the python script
			python3 run_main.py -file ${words[0]}

			# close all programs to avoid memory leak
			kill -9 $(jobs -p)
		fi
	
# checking if the entire file has been read
done < "$file"

#Run the bash file ./NAME_OF_BASH_FILE
#if you cant run this just go chmod +x the_file_name