start ""/b "GDF2_VS.exe" -p hz_sigma.data -g grammar.txt -n 10000 -e 5 -b 2 -r 20220505 -f norm_sigma_N10000.dat
start ""/b "GDF2_VS.exe" -p hz_sigma.data -g grammar_notrig.txt -n 10000 -e 5 -b 2 -r 20220505 -f notrig_sigma_N10000.dat
start ""/b "GDF2_VS.exe" -p hz_sigma.data -g grammar_nolog.txt -n 10000 -e 5 -b 2 -r 20220505 -f nolog_sigma_N10000.dat
start ""/b "GDF2_VS.exe" -p hz_sigma.data -g grammar_power.txt -n 10000 -e 5 -b 2 -r 20220505 -f power_sigma_N10000.dat
