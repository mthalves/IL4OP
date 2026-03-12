import inspect
import os
import datetime
import warnings

######
# EXCEPTION METHODS
######
# returns the current line number in our program.
def lineno():
    return inspect.currentframe().f_back.f_lineno

######
# WRITE METHODS
######
class LogFile:

    def __init__(self,problem_name,scenario,method,exp_num,*args):
        # creating the path
        if(not os.path.isdir("./logs")):
            os.mkdir("./logs")

        self.exp_name = problem_name if scenario is None or scenario == '' else problem_name+'_'+scenario
        self.header = args

        main_problem = problem_name.split('.')[0]
        self.path = "./logs/"+main_problem+'/'
        if(not os.path.isdir(self.path)):
            os.mkdir(self.path)

        # defining filename
        self.start_time = datetime.datetime.now()
        self.filename = str(method)+'_'+str(self.exp_name)+'_'+str(exp_num)+'.csv'

        # creating the result file
        self.write_header()

    def write_header(self):
        with open(self.path+self.filename, 'w') as logfile:
            for header in self.header[0]:
                logfile.write(str(header)+";")
            logfile.write('\n')

    def write(self,*args):
        with open(self.path+self.filename, 'a') as logfile:
            if(not len(args) ==len(self.header)):
                warnings.warn("Initialisation and writing have different sizes .")

            for key in args[0]:
                logfile.write(str(args[0][key])+";")

            logfile.write('\n')