# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 21:08:47 2022

@author: USER
"""

def punctuation_counter(data):
    count = 0;  
      
    for i in range (0, len (data)):   
        #Checks whether given character is a punctuation mark  
        if str[i] in ('!', "," ,"\'" ,";" ,"\"", ".", "-" ,"?"):  
            count = count + 1;  
              
    print ("Total number of punctuation characters exists in string: ");  
    print (count);
    


  