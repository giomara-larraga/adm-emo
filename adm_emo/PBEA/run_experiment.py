import os
path = "./irace-experiments-desdeo/PBEA/CSI/"
command = "irace --scenario " + path + "scenario.txt --parameter-file "+path+"parameters.txt"
os.system(command)