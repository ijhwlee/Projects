start ""/b "GDF2_VS.exe" -p hz_sigma.data -g grammar.txt -n 10000 -e 5 -b1 -f norm_range_N10000.dat
start ""/b "GDF2_VS.exe" -p hz_sigma.data -g grammar_notrig.txt -n 10000 -e 5 -b 1 -f notrig_range_N10000.dat
start ""/b "GDF2_VS.exe" -p hz_sigma.data -g grammar_nolog.txt -n 10000 -e 5 -b 1 -f nolog_range_N10000.dat
start ""/b "GDF2_VS.exe" -p hz_sigma.data -g grammar_power.txt -n 10000 -e 5 -b 1 -f power_range_N10000.dat
