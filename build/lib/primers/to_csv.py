import glob
import os
import yaml

files = glob.glob("*.yml")
#print(primers)
with open("primers.csv","w") as out:
    out.write("OligoName,Sequence\n")

    for f in files:
        primer = yaml.load(open(f))
        prefix = os.path.splitext(f)[0]
        fw_name = "%s_fw"%prefix
        rw_name = "%s_rw"%prefix
        fw = primer["Forward"]["primer"]["5'"] 
        rw = primer["Reverse"]["primer"]["5'"]
        out.write("%s,%s\n"%(fw_name,fw))
        out.write("%s,%s\n"%(rw_name,rw))
