import os

def modify_name_lat(filename,save_file):
    with open(filename) as f:
       lines = f.readlines()
    
    for i in range(len(lines)-1):
         if i==0:
            lines[i]=lines[i].split(" ")[0]+"_new \n"
         elif lines[i]=='\n':
            lines[i+1]=lines[i+1].split(" ")[0]+"_new \n"
    #this is writing the new lat file 
    with open(save_file, 'w') as file:
         file.writelines(lines)

def read2txt(filename):
  os.system("gunzip -c lat.1.gz |lattice-copy --write-compact=true ark:- ark,t:- |int2sym.pl -f 3 data/lang/words.txt > "+filename+" ")

def txt2write(filename):
  os.system("sym2int.pl -f 3  data/lang/words.txt "+filename+" |lattice-copy --write-compact=true ark,t:- ark:- |gzip - > lat.1.gz")


def main():
  filename='lat.txt' 
  save_file='new_lat2.txt'
  read2txt(filename)
  modify_name_lat(filename,save_file) 
  txt2write(save_file) 

  
if __name__== "__main__":
  main()
