# new
#2018/09/18
#this network is based on tensorflow, readme for more details


#install P862

download C code:
http://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en
unzip P862_annex_A_2005_CD*.zip

cd Software/P862_annex_A_2005_CD/source
gcc -o PESQ -lm *.c

./PESQ

 cd Software/P862_annex_A_2005_CD/conform
 ../source/PESQ +8000 or105.wav dg105.wav
