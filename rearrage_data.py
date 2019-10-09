import random



#for each set rearrage the files
def rearrage_files(filepath_spks,filepath_wavs,sentpers,list_2spks_file,mix_2_spk_file):
    #read speakers
    #filepath_spks='Out_file.txt'
    spks=[]
    with open(filepath_spks) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            print("Line {}: {}".format(cnt, line.strip()))
            spks.append(line[:-1])
            line = fp.readline()
            cnt += 1
        
    print("The number of speakers is:{}".format(cnt))
    #list all spks 
    all_wav_path={}
    for i in spks:
        all_wav_path[i]=[]


    #read wav files
    #filepath_wavs='tt.flist'
   
    with open(filepath_wavs) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            print("Line {}: {}".format(cnt, line.strip()))
            spk=line.split('/')[9]
            all_wav_path[spk].append(line[:-1])
            line = fp.readline()
            cnt += 1
    print("The number of wav files is:{}".format(cnt))

    #make mixture pairs, test
    spk1=[]
    spk2=[]
    for i in range(len(spks)/2):
        for j in range(len(spks)-len(spks)/2):
            list_spk1=random.sample(all_wav_path[spks[i]], sentpers)
            list_spk2=random.sample(all_wav_path[spks[j+len(spks)/2]], sentpers)
            temp=[]
            spk1.extend(list_spk1)
            spk2.extend(list_spk2)
    #print(spk1)
    #print(len(spk1))


    #read SNRs values from WSJ0-2mix
    #read wav files
    filepath_ref=mix_2_spk_file
    snrs1=[]
    snrs2=[]
    with open(filepath_ref) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            print("Line {}: {}".format(cnt, line.strip()))
            snrs1.append(line.split()[1])
            snrs2.append(line.split()[3])
            line = fp.readline()
            cnt += 1
    #print(len(snrs1))
    #merge two speakers in one text file,
    L=len(snrs1)/4
    spk1_0=random.sample(spk1, L)
    spk2_0=random.sample(spk2, L)
    list_2spks = [spk1_0,snrs1,spk2_0,snrs2] 
    with open(list_2spks_file, "w") as file:
        for x in zip(*list_2spks):
            file.write("{0}\t{1}\t{2}\t{3}\n".format(*x))

    #check the list if necessary
    #fp=open('list_2spks_tt.txt')
    #line=fp.readline()
    #print(line.split('\t')[1])

if __name__ == "__main__":
    #Let's list all files we needed
    filepath_spks_test='tt_spks.txt'
    filepath_spks_dev='dev_spks.txt'
    filepath_spks_train='tr_spks.txt'

    filepath_wavs_test='tt.flist'
    filepath_wavs_dev='dev.flist'
    filepath_wavs_train='tr.flist'

    #save the path with two speakers and SNRs
    mix_2_spk_test='mix_2_spk_tt.txt'
    mix_2_spk_dev='mix_2_spk_cv.txt'
    mix_2_spk_train='mix_2_spk_tr.txt'
    rearrage_files(filepath_spks_test,filepath_wavs_test,8,'sm_list_mix_2spks_tt.txt',mix_2_spk_test)
    rearrage_files(filepath_spks_dev,filepath_wavs_dev,13,'sm_list_mix_2spks_dev.txt',mix_2_spk_dev)
    rearrage_files(filepath_spks_train,filepath_wavs_train,2,'sm_list_mix_2spks_tr.txt',mix_2_spk_train)
