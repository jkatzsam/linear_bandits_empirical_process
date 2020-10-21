To run the experiments, follow these steps:

    1. Make sure there is a folder named "results" in your directory (the data from the experiment will be saved here).
    
    2. Go to parallelize.py and after the line "if __name__=='__main__':" uncomment the relevant function to run the 
    corresponding experiment. You may specify the number of cores and the number of trials. Then, at the command line
    type "parallelize.py".
    
    3. The plot.ipynb contains code to plot the results. Go to the relevant cell and load in the data at the line that says
    "data = pickle.load(open('./results/NAME_OF_FILE_TO_PLOT','rb'))" (see the function for the experiment to find the name
    of the generated data file.)
