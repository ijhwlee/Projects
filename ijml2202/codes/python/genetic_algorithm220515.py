import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lcdm_series220515 import *
"""
Obtained analytic functions by genetic algorithm
genetic algorithm obtained functions with excluding divergences bwteen z = 0.9 ~ 1.5
"""
#user defined power functions
def udf1(x):
    return x
def udf2(x):
    return x*x
def udf3(x):
    return x*x*x
def udf4(x):
    return np.power(x,4)
def udf5(x):
    return np.power(x, 5)
def udf6(x):
    return np.power(x, 6)

# function dictionary
ga_functions = {}

# use grammar.txt and N = 1000, -1.0630457246003029e+01
# no restriction for H(z) values
# 99.19+x1/(5.69/log(x1+(x1))/3.2)*62.9+01.99+x1-7.7/sqrt(abs((x1)))
def ga_func_norm1(x1):
    v = 99.19+x1/(5.69/np.log(x1+(x1))/3.2)*62.9+1.99+x1-7.7/np.sqrt(np.abs((x1)))
    return v
# use grammar.txt and N = 1000, -1.2509993431760298e+01
# use global range [30, 600]
# 89.8+(x1*(exp((sqrt(10.66)))))*x1+log((x1*x1))/exp(x1)*sqrt(24.99)-x1^log(exp(x1))
def ga_func_norm_range1(x1):
    v = 89.8+(x1*(np.exp((np.sqrt(10.66)))))*x1+np.log((x1*x1))/np.exp(x1)*np.sqrt(24.99)-x1**np.log(np.exp(x1))
    return v
# use grammar.txt and N = 1000, -8.5851724180589795e+00
# use sigma range [-10sigma, 10sigma]
# 71.1*(sqrt(x1)^abs(2.58)+sqrt((((cos(sqrt(x1)))))))-sqrt(x1)-cos(log(x1)*160.8)*(((((cos(((x1)))))))+exp(0.7)*abs(sin(x1))+x1*sin(x1))
def ga_func_norm_sigma1(x1):
    v = 71.1*(np.sqrt(x1)**np.abs(2.58)+np.sqrt((((np.cos(np.sqrt(x1)))))))-np.sqrt(x1)-np.cos(np.log(x1)*160.8)*(((((np.cos(((x1)))))))+np.exp(0.7)*np.abs(np.sin(x1))+x1*np.sin(x1))
    return v

# use grammar.txt and N = 2000, -1.0629064570521166e+01
# no restriction for H(z) values
# 99.19+x1/(5.68/log(x1+(x1))/3.2)*62.9+01.99+x1-7.7/sqrt(abs((x1)))
def ga_func_norm2(x1):
    v = 99.19+x1/(5.69/np.log(x1+(x1))/3.2)*62.9+1.99+x1-7.7/np.sqrt(np.abs((x1)))
    return v
# use grammar.txt and N = 2000, -1.2489536997078396e+01
# use global range [30, 600]
# 89.9+(x1*(exp((sqrt(10.67)))))*x1+log((x1*x1))/exp(x1)*sqrt(26.56)-x1^x1
def ga_func_norm_range2(x1):
    v = 89.9+(x1*(np.exp((np.sqrt(10.67)))))*x1+np.log((x1*x1))/np.exp(x1)*np.sqrt(26.56)-x1**x1
    return v
# use grammar.txt and N = 2000, -8.5366999071665735e+00
# use sigma range [-10sigma, 10sigma]
# 71.0*(sqrt(x1)^abs(2.58)+sqrt((((cos(sqrt(x1)))))))-sqrt(x1)-cos(log(x1)*160.7)*(((((cos(((x1)))))))+exp(0.9)*abs(sin(x1))+x1*sin(x1))
def ga_func_norm_sigma2(x1):
    v = 71.0*(np.sqrt(x1)**np.abs(2.58)+np.sqrt((((np.cos(np.sqrt(x1)))))))-np.sqrt(x1)-np.cos(np.log(x1)*160.7)*(((((np.cos(((x1)))))))+np.exp(0.9)*np.abs(np.sin(x1))+x1*np.sin(x1))
    return v

# use grammar.txt and N = 10000, -8.7598459749145494e+00
# no restriction for H(z) values
# 90.45+x1/(8.96/log(x1+x1)/(428.16))*0.73^log((x1-sin(44.99/(591.0*log(x1)))))
def ga_func_norm10(x1):
    v = 90.45+x1/(8.96/np.log(x1+x1)/(428.16))*0.73**np.log((x1-np.sin(44.99/(591.0*np.log(x1)))))
    return v
# use grammar.txt and N = 10000, -9.5237519295163384e+00
# use global range [30, 600]
# (89.13+(x1*exp(sqrt(10.2))*x1+4.0+log((x1))*7.62/x1^x1+sin(2141.9*(x1))*exp(sqrt(x1))-cos((log((exp(cos(86.64/(x1)/(cos(396.12))))^29.1))))))
def ga_func_norm_range10(x1):
    v = (89.13+(x1*np.exp(np.sqrt(10.2))*x1+4.0+np.log((x1))*7.62/x1**x1+np.sin(2141.9*(x1))*np.exp(np.sqrt(x1))-np.cos((np.log((np.exp(np.cos(86.64/(x1)/(np.cos(396.12))))**29.1))))))
    return v
# use grammar.txt and N = 10000, -8.1687981821569178e+00
# use sigma range [-10sigma, 10sigma]
# 71.0*(sqrt(x1)^abs(2.57)+sqrt((((cos(sqrt(x1)))))))-sqrt(x1)-cos(log(x1)*160.7)*(sin(((cos(((x1))))))+exp(0.7)*abs(sin(x1))+x1+sin(abs(x1)-sqrt(exp(x1))*((329.9))))
def ga_func_norm_sigma10(x1):
    v = 71.0*(np.sqrt(x1)**np.abs(2.57)+np.sqrt((((np.cos(np.sqrt(x1)))))))-np.sqrt(x1)-np.cos(np.log(x1)*160.7)*(np.sin(((np.cos(((x1))))))+np.exp(0.7)*np.abs(np.sin(x1))+x1+np.sin(np.abs(x1)-np.sqrt(np.exp(x1))*((329.9))))
    return v

# functions with grammar.txt 
func_norm_list = [ga_func_norm1, ga_func_norm2, ga_func_norm10]
func_norm_range_list = [ga_func_norm_range1, ga_func_norm_range2, ga_func_norm_range10]
func_norm_sigma_list = [ga_func_norm_sigma1, ga_func_norm_sigma2, ga_func_norm_sigma10]
N_norm_list = [1000, 2000, 10000]

ga_func_nobound = {}
ga_func_nobound["functions"] = func_norm_list
ga_func_nobound["bound"] = "No bound"
ga_func_global = {}
ga_func_global["functions"] = func_norm_range_list
ga_func_global["bound"] = "Global range"
ga_func_local = {}
ga_func_local["functions"] = func_norm_sigma_list
ga_func_local["bound"] = "Local range"

ga_functions_norm = {}
ga_functions_norm["nobound"] = ga_func_nobound
ga_functions_norm["global"] = ga_func_global
ga_functions_norm["local"] = ga_func_local
ga_functions_norm["N"]=N_norm_list
ga_functions_norm["grammar"] = "Normal"

ga_functions["norm"] = ga_functions_norm


# use grammar_notrig.txt and N = 1000, -1.2067646492065625e+01
# no restriction for H(z) values
# 9.65*exp(x1)+(70.5+((log(((sqrt(78.0)^x1))*x1+x1)))*8.8)-tanh((tanh(0.06))-(sqrt(7.77))*((x1))/(log(x1)/6.8*abs((85.63))/9.0/abs(x1)/(7.48)/x1))
def ga_func_notrig1(x1):
    v = 9.65*np.exp(x1)+(70.5+((np.log(((np.sqrt(78.0)**x1))*x1+x1)))*8.8)-np.tanh((np.tanh(0.06))-(np.sqrt(7.77))*((x1))/(np.log(x1)/6.8*np.abs((85.63))/9.0/np.abs(x1)/(7.48)/x1))
    return v
# use grammar_notrig.txt and N = 1000, -1.0285630520091525e+01
# use global range [30, 600]
# sqrt(4692.7)*(sqrt((exp(x1)*abs(tanh(x1))+sqrt(tanh(sqrt(log(sqrt(867.0)))/72611.9^abs(x1^2.7)/271.3*abs((40.66))^09.6+(x1)/((sqrt((65.19)))))))))
def ga_func_notrig_range1(x1):
    v = np.sqrt(4692.7)*(np.sqrt((np.exp(x1)*np.abs(np.tanh(x1))+np.sqrt(np.tanh(np.sqrt(np.log(np.sqrt(867.0)))/72611.9**np.abs(x1**2.7)/271.3*np.abs((40.66))**9.6+(x1)/((np.sqrt((65.19)))))))))
    return v
# use grammar_notrig.txt and N = 1000, -1.0301255711028125e+01
# use sigma range [-10sigma, 10sigma]
# sqrt((abs(exp(x1)))*5011.1-((51.8/x1))-exp(x1^x1)*(abs(x1))+abs((((tanh(tanh(19.06)))/x1)+((7521.4+((((98.71))))-x1)^tanh((x1)))/x1)))
def ga_func_notrig_sigma1(x1):
    v = np.sqrt((np.abs(np.exp(x1)))*5011.1-((51.8/x1))-np.exp(x1**x1)*(np.abs(x1))+np.abs((((np.tanh(np.tanh(19.06)))/x1)+((7521.4+((((98.71))))-x1)**np.tanh((x1)))/x1)))
    return v

# use grammar_notrig.txt and N = 2000, -1.1803291098126802e+01
# no restriction for H(z) values
# 9.65*exp(x1)+(70.4+((log(((sqrt(78.3)^x1))*x1+x1)))*8.9)-tanh(log((0.09))-(sqrt(7.77))*((x1))/(log(x1)/6.9*abs((exp(2.15)))/9.0/abs(x1)/(7.78)/x1))
def ga_func_notrig2(x1):
    v = 9.65*np.exp(x1)+(70.4+((np.log(((np.sqrt(78.3)**x1))*x1+x1)))*8.9)-np.tanh(np.log((0.09))-(np.sqrt(7.77))*((x1))/(np.log(x1)/6.9*np.abs((np.exp(2.15)))/9.0/np.abs(x1)/(7.78)/x1))
    return v
# use grammar_notrig.txt and N = 2000, -9.0620747515841220e+00
# use global range [30, 600]
# sqrt(267.4)*(exp(((sqrt(x1)*abs(tanh(x1))+abs(tanh((log(6010.15))-x1^5.3)))))+sqrt(3.09))
def ga_func_notrig_range2(x1):
    v = np.sqrt(267.4)*(np.exp(((np.sqrt(x1)*np.abs(np.tanh(x1))+np.abs(np.tanh((np.log(6010.15))-x1**5.3)))))+np.sqrt(3.09))
    return v
# use grammar_notrig.txt and N = 2000, -1.0243748586173707e+01
# use sigma range [-10sigma, 10sigma]
# (sqrt(((exp((x1)))*5010.0-((51.8/x1))-exp(x1^x1)*abs(x1)+abs(((((exp(exp((tanh(513.05)))))-(7848.41-(((98.74)))*x1)^tanh((x1)))/x1))))))
def ga_func_notrig_sigma2(x1):
    v = (np.sqrt(((np.exp((x1)))*5010.0-((51.8/x1))-np.exp(x1**x1)*np.abs(x1)+np.abs(((((np.exp(np.exp((np.tanh(513.05)))))-(7848.41-(((98.74)))*x1)**np.tanh((x1)))/x1))))))
    return v

# use grammar_notrig.txt and N = 10000, -8.2318225750199421e+00
# no restriction for H(z) values
# (9.9*exp(x1)+((68.99+log(((sqrt(93.1^x1)))*x1+x1)*8.0)))/tanh(exp(x1)/x1^100.04+x1*tanh((0.19)+x1))
def ga_func_notrig10(x1):
    v = (9.9*np.exp(x1)+((68.99+np.log(((np.sqrt(93.1**x1)))*x1+x1)*8.0)))/np.tanh(np.exp(x1)/x1**100.04+x1*np.tanh((0.19)+x1))
    return v
# use grammar_notrig.txt and N = 10000, -8.3577212189265939e+00
# use global range [30, 600]
# sqrt(261.1)*(exp(((sqrt(x1)*abs(tanh(x1))+abs(tanh((sqrt(0121.64))-x1^5.4)))))+sqrt(3.33))
def ga_func_notrig_range10(x1):
    v = np.sqrt(261.1)*(np.exp(((np.sqrt(x1)*np.abs(np.tanh(x1))+np.abs(np.tanh((np.sqrt(121.64))-x1**5.4)))))+np.sqrt(3.33))
    return v
# use grammar_notrig.txt and N = 10000, -7.7194247444324979e+00
# use sigma range [-10sigma, 10sigma]
# (sqrt(abs(exp((x1))*5111.85-(80.0)/x1-exp(x1^x1)*abs(x1)+abs(((exp(abs((x1)^x1+(x1)-sqrt(56.11)))*tanh(x1^(20.30))*51.26-7.65))))))
def ga_func_notrig_sigma10(x1):
    v = (np.sqrt(np.abs(np.exp((x1))*5111.85-(80.0)/x1-np.exp(x1**x1)*np.abs(x1)+np.abs(((np.exp(np.abs((x1)**x1+(x1)-np.sqrt(56.11)))*np.tanh(x1**(20.30))*51.26-7.65))))))
    return v

# functions with grammar_notrig.txt 
func_notrig_list = [ga_func_notrig1, ga_func_notrig2, ga_func_notrig10]
func_notrig_range_list = [ga_func_notrig_range1, ga_func_notrig_range2, ga_func_notrig_range10]
func_notrig_sigma_list = [ga_func_notrig_sigma1, ga_func_notrig_sigma2, ga_func_notrig_sigma10]
N_notrig_list = [1000, 2000, 10000]

ga_func_nobound = {}
ga_func_nobound["functions"] = func_notrig_list
ga_func_nobound["bound"] = "No bound"
ga_func_global = {}
ga_func_global["functions"] = func_notrig_range_list
ga_func_global["bound"] = "Global range"
ga_func_local = {}
ga_func_local["functions"] = func_notrig_sigma_list
ga_func_local["bound"] = "Local range"

ga_functions_notrig = {}
ga_functions_notrig["nobound"] = ga_func_nobound
ga_functions_notrig["global"] = ga_func_global
ga_functions_notrig["local"] = ga_func_local
ga_functions_notrig["N"]=N_notrig_list
ga_functions_notrig["grammar"] = "Excl. Trig."

ga_functions["notrig"] = ga_functions_notrig

# use grammar_nolog.txt and N = 1000, -7.6868841571755571e+00
# no restriction for H(z) values
# sqrt(exp(x1))*(70.1+exp(x1*4.45*x1*x1-x1*(exp(abs(x1)))^(x1))*x1)
def ga_func_nolog1(x1):
    v = np.sqrt(np.exp(x1))*(70.1+np.exp(x1*4.45*x1*x1-x1*(np.exp(np.abs(x1)))**(x1))*x1)
    return v
# use grammar_nolog.txt and N = 1000, -9.5719151661179307e+00
# use global range [30, 600]
# sqrt(9993.45*sqrt(sqrt(x1)^7.38+x1/tanh(sqrt(x1)+tanh((x1))/tanh(abs(((7747.5))))-(((tanh(x1^061.02)))))))
def ga_func_nolog_range1(x1):
    v = np.sqrt(9993.45*np.sqrt(np.sqrt(x1)**7.38+x1/np.tanh(np.sqrt(x1)+np.tanh((x1))/np.tanh(np.abs(((7747.5))))-(((np.tanh(x1**61.02)))))))
    return v
# use grammar_nolog.txt and N = 1000, -9.8489640635986806e+00
# use sigma range [-10sigma, 10sigma]
# 77.00*sqrt(x1/(tanh(tanh(x1)))^tanh((abs(sqrt(6.5))))^sqrt((2.5)/x1)+(x1)*x1)
def ga_func_nolog_sigma1(x1):
    v = 77.00*np.sqrt(x1/(np.tanh(np.tanh(x1)))**np.tanh((np.abs(np.sqrt(6.5))))**np.sqrt((2.5)/x1)+(x1)*x1)
    return v

# use grammar_nolog.txt and N = 2000, -7.5090931582769160e+00
# no restriction for H(z) values
# sqrt(exp(x1))*(70.0+exp(x1-x1^x1+(tanh((x1/(tanh((1.52^(x1)^x1))^((((x1)*tanh(x1))))+exp(tanh((x1))^x1^sqrt(((983135.42)))))))))^5.99)
def ga_func_nolog2(x1):
    v = np.sqrt(np.exp(x1))*(70.0+np.exp(x1-x1**x1+(np.tanh((x1/(np.tanh((1.52**(x1)**x1))**((((x1)*np.tanh(x1))))+np.exp(np.tanh((x1))**x1**np.sqrt(((983135.42)))))))))**5.99)
    return v
# use grammar_nolog.txt and N = 2000, -8.1579246207075116e+00
# use global range [30, 600]
# sqrt(9960.03*sqrt(sqrt(x1)^7.30+x1/tanh(sqrt(x1)+tanh(x1*tanh(abs(exp(37384.91))))-tanh(x1^94342.1)*sqrt(sqrt(sqrt((4.75)))+tanh((85.8/(exp(1.6))+(x1-exp(0.5))))))))
def ga_func_nolog_range2(x1):
    v = np.sqrt(9960.03*np.sqrt(np.sqrt(x1)**7.30+x1/np.tanh(np.sqrt(x1)+np.tanh(x1*np.tanh(np.abs(np.exp(37384.91))))-np.tanh(x1**94342.1)*np.sqrt(np.sqrt(np.sqrt((4.75)))+np.tanh((85.8/(np.exp(1.6))+(x1-np.exp(0.5))))))))
    return v
# use grammar_nolog.txt and N = 2000, -9.8462744805717790e+00
# use sigma range [-10sigma, 10sigma]
# 76.93*sqrt(x1/(tanh(tanh(x1)))^tanh((abs(sqrt(6.4))))^sqrt((2.1)/x1)+(x1)*x1)
def ga_func_nolog_sigma2(x1):
    v = 76.93*np.sqrt(x1/(np.tanh(np.tanh(x1)))**np.tanh((np.abs(np.sqrt(6.4))))**np.sqrt((2.1)/x1)+(x1)*x1)
    return v

# use grammar_nolog.txt and N = 10000, -6.5832280031985189e+00
# no restriction for H(z) values
# sqrt(exp(x1))*70.14+exp(x1-x1*x1+3.64*sqrt(tanh(x1^(2400.0/(abs((((abs(5.79))))))-sqrt((4289.8^x1))-9.9))))*((((x1))))
def ga_func_nolog10(x1):
    v = np.sqrt(np.exp(x1))*70.14+np.exp(x1-x1*x1+3.64*np.sqrt(np.tanh(x1**(2400.0/(np.abs((((np.abs(5.79))))))-np.sqrt((4289.8**x1))-9.9))))*((((x1))))
    return v
# use grammar_nolog.txt and N = 10000, -7.4784947088811942e+00
# use global range [30, 600]
# sqrt(9968.52*sqrt(sqrt(x1)^7.40+x1/tanh(sqrt(x1)+tanh(x1*tanh(((sqrt(abs(0062.43))))))-tanh(x1^52612.1)*sqrt(sqrt(sqrt((4.98)))+tanh(((sqrt((59.5/15.94+(tanh(x1))*036.5))))-((x1))^exp(x1))))))
def ga_func_nolog_range10(x1):
    v = np.sqrt(9968.52*np.sqrt(np.sqrt(x1)**7.40+x1/np.tanh(np.sqrt(x1)+np.tanh(x1*np.tanh(((np.sqrt(np.abs(62.43))))))-np.tanh(x1**52612.1)*np.sqrt(np.sqrt(np.sqrt((4.98)))+np.tanh(((np.sqrt((59.5/15.94+(np.tanh(x1))*36.5))))-((x1))**np.exp(x1))))))
    return v
# use grammar_nolog.txt and N = 10000, -7.1196472035161529e+00
# use sigma range [-10sigma, 10sigma]
# 70.00*sqrt(exp(x1))+exp(((x1))-(x1)^abs(exp(x1))+x1*x1*3.14-x1-tanh((tanh(tanh(9.79)/(tanh(20.00))))-exp((sqrt(x1)))*abs(x1)^abs(7.44)))
def ga_func_nolog_sigma10(x1):
    v = 70.00*np.sqrt(np.exp(x1))+np.exp(((x1))-(x1)**np.abs(np.exp(x1))+x1*x1*3.14-x1-np.tanh((np.tanh(np.tanh(9.79)/(np.tanh(20.00))))-np.exp((np.sqrt(x1)))*np.abs(x1)**np.abs(7.44)))
    return v

# functions with grammar_nolog.txt 
func_nolog_list = [ga_func_nolog1, ga_func_nolog2, ga_func_nolog10]
func_nolog_range_list = [ga_func_nolog_range1, ga_func_nolog_range2, ga_func_nolog_range10]
func_nolog_sigma_list = [ga_func_nolog_sigma1, ga_func_nolog_sigma2, ga_func_nolog_sigma10]
N_nolog_list = [1000, 2000, 10000]

ga_func_nobound = {}
ga_func_nobound["functions"] = func_nolog_list
ga_func_nobound["bound"] = "No bound"
ga_func_global = {}
ga_func_global["functions"] = func_nolog_range_list
ga_func_global["bound"] = "Global range"
ga_func_local = {}
ga_func_local["functions"] = func_nolog_sigma_list
ga_func_local["bound"] = "Local range"

ga_functions_nolog = {}
ga_functions_nolog["nobound"] = ga_func_nobound
ga_functions_nolog["global"] = ga_func_global
ga_functions_nolog["local"] = ga_func_local
ga_functions_nolog["N"]=N_nolog_list
ga_functions_nolog["grammar"] = "Excl. Log."

ga_functions["nolog"] = ga_functions_nolog

# use grammar_power.txt and N = 1000, -1.1757119314558777e+01
# no restriction for H(z) values
# udf2(9.0)+(x1*10.47+(udf3(x1+udf1(x1)+x1)/x1-((x1^udf1((x1^1.89))))/(x1)))
def ga_func_power1(x1):
    v = udf2(9.0)+(x1*10.47+(udf3(x1+udf1(x1)+x1)/x1-((x1**udf1((x1**1.89))))/(x1)))
    return v
# use grammar_power.txt and N = 1000, -1.3585675560632749e+01
# use global range [30, 600]
# udf2(9.0)+(x1)*29.0+udf3(((x1))+(x1))-udf3(x1)-1.69/x1-((x1))^udf1(udf1(udf3((udf1((x1))))/4.3))
def ga_func_power_range1(x1):
    v = udf2(9.0)+(x1)*29.0+udf3(((x1))+(x1))-udf3(x1)-1.69/x1-((x1))**udf1(udf1(udf3((udf1((x1))))/4.3))
    return v
# use grammar_power.txt and N = 1000, -1.0531631439948802e+01
# use sigma range [-10sigma, 10sigma]
# udf3(x1)*29.9/x1+udf2((8.61)+x1)-udf2(x1)^2.4
def ga_func_power_sigma1(x1):
    v = udf3(x1)*29.9/x1+udf2((8.61)+x1)-udf2(x1)**2.4
    return v

# use grammar_power.txt and N = 2000, -1.0640885124284996e+01
# no restriction for H(z) values
# udf2(8.0)+x1*53.77+(udf2(x1-(udf1(0.99))+(x1)))*(2.6)
def ga_func_power2(x1):
    v = udf2(8.0)+x1*53.77+(udf2(x1-(udf1(0.99))+(x1)))*(2.6)
    return v
# use grammar_power.txt and N = 2000, -1.0371019828950601e+01
# use global range [30, 600]
# udf2(9.0)+(x1)*92.3/udf1(x1+udf3(1.0))*x1-1.8/(udf1(udf1(0.3)))
def ga_func_power_range2(x1):
    v = udf2(9.0)+(x1)*92.3/udf1(x1+udf3(1.0))*x1-1.8/(udf1(udf1(0.3)))
    return v
# use grammar_power.txt and N = 2000, -1.0180809542870932e+01
# use sigma range [-10sigma, 10sigma]
# udf2(x1)*29.99+udf1(udf1(udf2(8.55+x1)))-udf2(x1)^2.44+x1+x1
def ga_func_power_sigma2(x1):
    v = udf2(x1)*29.99+udf1(udf1(udf2(8.55+x1)))-udf2(x1)**2.44+x1+x1
    return v

# use grammar_power.txt and N = 10000, -7.6952311366883084e+00
# no restriction for H(z) values
# 64.3+x1*(52.41)+udf2((((x1)))-0.2^x1)*(6.89+9.99/udf3((udf3(x1/(udf2(udf3(x1)))))-x1+udf3(x1))*x1)
def ga_func_power10(x1):
    v = 64.3+x1*(52.41)+udf2((((x1)))-0.2**x1)*(6.89+9.99/udf3((udf3(x1/(udf2(udf3(x1)))))-x1+udf3(x1))*x1)
    return v
# use grammar_power.txt and N = 10000, -8.9223572803737845e+00
# use global range [30, 600]
# 9.8+x1*92.01/(((x1+udf2(1.0))))*x1-9.21+74.28+udf3((x1*udf3(((udf2(udf3(x1))/12.1)))^udf1(udf1(udf1(x1))/(988.99)+((udf3((udf2((x1))-(udf3((x1))))))))))
def ga_func_power_range10(x1):
    v = 9.8+x1*92.01/(((x1+udf2(1.0))))*x1-9.21+74.28+udf3((x1*udf3(((udf2(udf3(x1))/12.1)))**udf1(udf1(udf1(x1))/(988.99)+((udf3((udf2((x1))-(udf3((x1))))))))))
    return v
# use grammar_power.txt and N = 10000, -7.9879412023203358e+00
# use sigma range [-10sigma, 10sigma]
# udf2(x1)*(26.94)+udf2(((8.42+x1)))-udf2(x1)^2.4+x1*8.20+45.69/(udf1(udf1(x1)^(x1-(udf2((4.7)))))+x1^udf2(x1)^udf3(x1))
def ga_func_power_sigma10(x1):
    v = udf2(x1)*(26.94)+udf2(((8.42+x1)))-udf2(x1)**2.4+x1*8.20+45.69/(udf1(udf1(x1)**(x1-(udf2((4.7)))))+x1**udf2(x1)**udf3(x1))
    return v

# functions with grammar_power.txt 
func_power_list = [ga_func_power1, ga_func_power2, ga_func_power10]
func_power_range_list = [ga_func_power_range1, ga_func_power_range2, ga_func_power_range10]
func_power_sigma_list = [ga_func_power_sigma1, ga_func_power_sigma2, ga_func_power_sigma10]
N_power_list = [1000, 2000, 10000]

ga_func_nobound = {}
ga_func_nobound["functions"] = func_power_list
ga_func_nobound["bound"] = "No bound"
ga_func_global = {}
ga_func_global["functions"] = func_power_range_list
ga_func_global["bound"] = "Global range"
ga_func_local = {}
ga_func_local["functions"] = func_power_sigma_list
ga_func_local["bound"] = "Local range"

ga_functions_power = {}
ga_functions_power["nobound"] = ga_func_nobound
ga_functions_power["global"] = ga_func_global
ga_functions_power["local"] = ga_func_local
ga_functions_power["N"]=N_power_list
ga_functions_power["grammar"] = "Power"

ga_functions["power"] = ga_functions_power

bound_names = ["No bound", "Global range", "Local range"]
grammar_names = ["Normal", "Excl. Trig.", "Excl. Log.", "Power"]

# standard LCDM function
# params[0] : Ok0, params[1] : OL0, H[2]:H0
def H_LCDM(params, train_x):
    k0 = params[0]
    L = params[1]
    H0 = params[2]
    m = 1 + k0 - L
    train_x = np.array(train_x)
    v = H0*np.sqrt(m*np.power(1+train_x, 3) - k0*np.power(1+train_x, 2) + L)
    return v

# standard flat LCDM function
def H_flat_LCDM(train_x):
    params = [0.0, 0.749, 71.716]
    return H_LCDM(params, train_x)

#define H_GA_power function, return H_GA_power(z_i) values
# params : parameters for power function
# H_GA(z) = a + b*z + c*z^2 - z**d
def H_GA_power(params, train_x):
    if np.isscalar(train_x): # single point evaluation
        z = train_x
        v = params[0] + params[1]*z + params[2]*z**2 - z**params[3]
        return v

    y_data = []
    for idx in range(len(train_x)):
        z = train_x[idx]
        v = params[0] + params[1]*z + params[2]*z**2 - z**params[3]
        y_data.append(v)
    return y_data

#define H_GA_swampland function, return H_GA_swampland(z_i) values
# params : parameters for power function
# H_GA(z) = a + b*z^2/(1+z)
def H_GA_swampland(params, train_x):
    if np.isscalar(train_x): # single point evaluation
        z = train_x
        v = params[0] + params[1]*z**2/(1+z)
        return v

    y_data = []
    for idx in range(len(train_x)):
        z = train_x[idx]
        v = params[0] + params[1]*z**2/(1+z)
        y_data.append(v)
    return y_data

regularization = 0.001
def set_regularization(beta):
    global regularization
    regularization = beta
    
#define H_ML_none function, return H_ML_none(z_i) values
# params : parameters for power function
# H_ML(z) = sum_{i=0}^{6} w_i z^i
def H_ML_none(params, train_x):
    return H_ML_orderN(params, train_x, len(params))

#define H_ML_ridge function, return H_ML_ridge(z_i) values
# params : parameters for power function
# H_ML(z) = sum_{i=0}^{6} w_i z^i + alpha*|w|^2
def H_ML_ridge(params, train_x):
    global regularization
    logL = H_ML_orderN(params, train_x, len(params))
    penalty = 0
    for idx in range(len(params)):
        penalty += params[idx]*params[idx]
    penalty = regularization * penalty
    return logL - penalty/2

#define H_ML_lasso function, return H_ML_lasso(z_i) values
# params : parameters for power function
# H_ML(z) = sum_{i=0}^{6} w_i z^i + alpha*|w|
def H_ML_lasso(params, train_x):
    global regularization
    logL = H_ML_orderN(params, train_x, len(params))
    penalty = 0
    for idx in range(len(params)):
        penalty += np.abs(params[idx])
    penalty = regularization * penalty
    return logL - penalty/2

#define H_ML_orderN function, return H_ML_orderN(z_i) values
# params : parameters for power function
# H_ML(z) = sum_{i=0}^{N} w_i z^i
def H_ML_orderN(params, train_x, N):
    if np.isscalar(train_x): # single point evauation
        z = train_x
        v = 0
        for idx in range(N):
            v = v + params[idx]*z**idx
        return v

    y_data = []
    for idx in range(len(train_x)):
        z = train_x[idx]
        v = 0
        for idx in range(N):
            v = v + params[idx]*z**idx
        y_data.append(v)
    return y_data

#define H_ML_order6 function, return H_ML_order6(z_i) values
# params : parameters for power function
# H_ML(z) = sum_{i=0}^{6} w_i z^i
def H_ML_order6(params, train_x):
    return H_ML_orderN(params, train_x, 7)

def ML_function_lcdm(train_x):
    # the following value may change and defined in Bayesian_Dynesty_Restricted_LCDM_ga_20220515.py
    #params = [71.715,   27.0007, 21.9178, 0.748175, -3.63099,  1.13841,   0.677204]
    params = [63.46318396, 41.18223296225411, 27.820341815835107, -4.325635302717404, -3.290827462428077, 4.0316961728520555, -1.3210472799573765]
    return H_ML_orderN(params, train_x, len(params))

def ML_function_lsf(train_x):
    # the following value may change and defined in Bayesian_Dynesty_Restricted_LCDM_ga_20220515.py
    params = [52.0140703, 228.90797962, -720.58440476, 1231.36686791, -963.14303437, 348.5058823, -47.31896143]
    return H_ML_orderN(params, train_x, len(params))

def ML_function_nopenalty(train_x):
    # the following value may change and defined in Bayesian_Dynesty_Restricted_LCDM_ga_20220515.py
    #params = [67.13235836,   81.78085976, -241.57614411,  528.20611295, -461.12867443,  178.80176075,  -25.57120566]
    params = [70.84767,   26.299116, 19.417992,  14.409067, -0.05665167,  -9.064274,  2.673782]
    return H_ML_orderN(params, train_x, len(params))

def ML_function_ridge(train_x):
    # the following value may change and defined in Bayesian_Dynesty_Restricted_LCDM_ga_20220515.py
    #params = [70.19502303, 29.21557494, 17.8032764 , 10.55336865,  1.02931078, -6.94154205,  1.88447721]
    params = [73.676994, 22.457187, 17.413399 , 9.35244,  1.5714686, -3.7502832,  0.6009149]
    return H_ML_orderN(params, train_x, len(params))

def ML_function_lasso(train_x):
    # the following value may change and defined in Bayesian_Dynesty_Restricted_LCDM_ga_20220515.py
    #params = [70.88043162, 26.34870065, 19.28389071, 13.82066085,  1.04845811, -9.66146965,  2.77664855]
    params = [70.73978, 25.773214, 26.183893, 3.3298175,  -6.5561234e-6, -3.5875168,  0.96844947]
    return H_ML_orderN(params, train_x, len(params))

# ML function dictionary
ML_functions = {}
ML_functions["lcdm"] = ML_function_lcdm
ML_functions["lsf"] = ML_function_lsf
ML_functions["nopenalty"] = ML_function_nopenalty
ML_functions["ridge"] = ML_function_ridge
ML_functions["lasso"] = ML_function_lasso

ML_names = [r"${\rm \Lambda CDM}$ Power","Least Square", "No Reg.", "Ridge", "Lasso"]
ML_types = ['lcdm', 'lsf', 'nopenalty', 'ridge', 'lasso']


#define drawing functions
def draw_data(train_x, train_y, train_sigma):
    fig, ax = plt.subplots(figsize=(10,8))
    #sns.lineplot(x=train_x, y=train_y, alpha=0.8, ax=ax)
    sns.scatterplot(x=train_x, y=train_y, alpha=0.8, ax=ax, marker="o", color='b')
    ax.errorbar(train_x, train_y, train_sigma, color='tab:blue', ecolor='tab:blue')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$x$', fontsize=20)

    #plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

list_markers = ["o", "s", "v", "^", "<", ">", "X"]
markers_size = 7

def draw_comparison_bound(grammar, N_idx, train_x, train_y, train_sigma):
    global list_markers, markers_size
    # draw comparison between bound types
    bound_types = ['nobound', 'global', 'local']
    lower = train_x[0]
    upper = train_x[len(train_x)-1]
    data_list = []
    for idx in range(len(bound_types)):
        y_data = []
        bound_type = bound_types[idx]
        for idx1 in range(len(train_x)):
            y_data.append(ga_functions[grammar][bound_type]['functions'][N_idx](train_x[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        bound = bound_types[idx]
        sns.lineplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound]['bound']))
        #sns.scatterplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound][bound_type]))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)
    
    plt.legend(loc='best', fontsize=20)
    plt.title(r"$N={0}$".format(ga_functions[grammar]['N'][N_idx]))
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

def draw_comparison_grammar(bound_type, N_idx, train_x, train_y, train_sigma, show_title = False):
    global list_markers, markers_size
    # draw comparison between grammar types
    grammar_types = ['norm', 'notrig', 'nolog', 'power']
    lower = train_x[0]
    upper = train_x[len(train_x)-1]
    markers = []
    data_list = []
    for idx in range(len(grammar_types)):
        y_data = []
        grammar = grammar_types[idx]
        for idx1 in range(len(train_x)):
            y_data.append(ga_functions[grammar][bound_type]['functions'][N_idx](train_x[idx1]))
        data_list.append(y_data)
        markers.append(list_markers[idx])

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        grammar = grammar_types[idx]
        sns.lineplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker=markers[idx%markers_size], label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
        #sns.scatterplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)
    
    plt.legend(loc='best', fontsize=20)
    plt.title(r"$N={0}$".format(ga_functions[grammar]['N'][N_idx]))
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    fig.savefig("compare_grammar1"+bound_type+".eps", format='eps')
    plt.show()

    # draw comparison between grammar types
    grammar_types = ['norm', 'notrig', 'nolog', 'power']
    lower = 0.05
    upper = 2.5
    delta = 0.02
    x_data = np.arange(lower, upper, delta)
    data_list = []
    for idx in range(len(grammar_types)):
        y_data = []
        grammar = grammar_types[idx]
        for idx1 in range(len(x_data)):
            y_data.append(ga_functions[grammar][bound_type]['functions'][N_idx](x_data[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        grammar = grammar_types[idx]
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker=markers[idx%markers_size], label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
        #sns.scatterplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)
    
    plt.legend(loc='best', fontsize=20)
    if show_title :
        plt.title(r"$N={0}$".format(ga_functions[grammar]['N'][N_idx]))
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    fig.savefig("compare_grammar2"+bound_type+".eps", format='eps')
    plt.show()

def draw_functions(grammar, bound_type, train_x, train_y, train_sigma):
    global list_markers, markers_size
    # draw only points
    lower = train_x[0]
    upper = train_x[len(train_x)-1]
    data_list = []
    for idx in range(len(ga_functions[grammar][bound_type]['functions'])):
        y_data = []
        for idx1 in range(len(train_x)):
            y_data.append(ga_functions[grammar][bound_type]['functions'][idx](train_x[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.scatterplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"$N = {0}$ {1} {2}".format(ga_functions[grammar]['N'][idx], ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    # draw points and lines
    lower = train_x[0]
    upper = train_x[len(train_x)-1]
    data_list = []
    for idx in range(len(ga_functions[grammar][bound_type]['functions'])):
        y_data = []
        for idx1 in range(len(train_x)):
            y_data.append(ga_functions[grammar][bound_type]['functions'][idx](train_x[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"$N = {0}$ {1} {2}".format(ga_functions[grammar]['N'][idx], ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    #draw values between observed points
    lower = 0.05
    upper = 2.5
    delta = 0.02
    x_data = np.arange(lower, upper, delta)
    data_list = []
    for idx in range(len(ga_functions[grammar][bound_type]['functions'])):
        y_data = []
        for idx1 in range(len(x_data)):
            y_data.append(ga_functions[grammar][bound_type]['functions'][idx](x_data[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"$N = {0}$ {1} {2}".format(ga_functions[grammar]['N'][idx], ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    #draw values between observed points indetail
    lower = 0.05
    upper = 2.5
    delta = 0.005
    x_data = np.arange(lower, upper, delta)
    data_list = []
    for idx in range(len(ga_functions[grammar][bound_type]['functions'])):
        y_data = []
        for idx1 in range(len(x_data)):
            y_data.append(ga_functions[grammar][bound_type]['functions'][idx](x_data[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"$N = {0}$ {1} {2}".format(ga_functions[grammar]['N'][idx], ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    #draw values near zero
    lower = 0.05
    upper = 0.2
    delta = 0.02
    x_data = np.arange(lower, upper, delta)
    data_list = []
    for idx in range(len(ga_functions[grammar][bound_type]['functions'])):
        y_data = []
        for idx1 in range(len(x_data)):
            y_data.append(ga_functions[grammar][bound_type]['functions'][idx](x_data[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"$N = {0}$ {1} {2}".format(ga_functions[grammar]['N'][idx], ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='upper left', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    #draw values large z
    lower = 1.95
    upper = 2.5
    delta = 0.02
    x_data = np.arange(lower, upper, delta)
    data_list = []
    for idx in range(len(ga_functions[grammar][bound_type]['functions'])):
        y_data = []
        for idx1 in range(len(x_data)):
            y_data.append(ga_functions[grammar][bound_type]['functions'][idx](x_data[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"$N = {0}$ {1} {2}".format(ga_functions[grammar]['N'][idx], ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    draw_comparison_bound(grammar, 0, train_x, train_y, train_sigma)
    draw_comparison_bound(grammar, 1, train_x, train_y, train_sigma)
    draw_comparison_bound(grammar, 2, train_x, train_y, train_sigma)

def draw_comparison_ML(train_x, train_y, train_sigma):
    global list_markers, markers_size, ML_types
    # draw comparison between ML types
    lower = train_x[0]
    upper = train_x[len(train_x)-1]
    markers = []
    data_list = []
    for idx in range(len(ML_types)):
        y_data = []
        ML = ML_types[idx]
        for idx1 in range(len(train_x)):
            y_data.append(ML_functions[ML](train_x[idx1]))
        data_list.append(y_data)
        markers.append(list_markers[idx])
    y_data_lcdm = H_flat_LCDM(train_x)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    sns.lineplot(x=train_x, y=y_data_lcdm, alpha=0.8, ax=ax, marker=list_markers[0], label=r"${\rm \Lambda CDM}$")
    for idx in range(len(data_list)):
        ML = ML_types[idx]
        sns.lineplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker=list_markers[(idx+1)%markers_size], label=r"{0}".format(ML_names[idx]))
        #sns.scatterplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)
    
    plt.legend(loc='best', fontsize=20)
    #plt.title(r"$N={0}$".format(ga_functions[grammar]['N'][N_idx]))
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

    # draw comparison between ML types
    lower = 0.05
    upper = 2.5
    delta = 0.02
    x_data = np.arange(lower, upper, delta)
    data_list = []
    for idx in range(len(ML_types)):
        y_data = []
        ML = ML_types[idx]
        for idx1 in range(len(x_data)):
            y_data.append(ML_functions[ML](x_data[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])
    y_data_lcdm_detail = H_flat_LCDM(x_data)

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    sns.lineplot(x=x_data, y=y_data_lcdm_detail, alpha=0.8, ax=ax, marker=list_markers[0], label=r"${\rm \Lambda CDM}$")
    for idx in range(len(data_list)):
        ML = ML_types[idx]
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker=list_markers[(idx+1)%markers_size], label=r"{0}".format(ML_names[idx]))
        #sns.scatterplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)
    
    plt.legend(loc='best', fontsize=20)
    #if show_title :
    #    plt.title(r"$N={0}$".format(ga_functions[grammar]['N'][N_idx]))
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    fig.savefig("hz_ML_comparison_LCDM.eps", format='eps')
    plt.show()

    # draw comparison between ML types
    lower = train_x[0]
    upper = train_x[len(train_x)-1]
    markers = []
    data_list = []
    for idx in range(1, len(ML_types)):
        y_data = []
        ML = ML_types[idx]
        for idx1 in range(len(train_x)):
            y_data.append(ML_functions[ML](train_x[idx1]))
        data_list.append(y_data)
        markers.append(list_markers[idx])

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(1, len(data_list)):
        ML = ML_types[idx]
        sns.lineplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker=markers[idx%markers_size], label=r"{0}".format(ML_names[idx]))
        #sns.scatterplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)
    
    plt.legend(loc='best', fontsize=20)
    #plt.title(r"$N={0}$".format(ga_functions[grammar]['N'][N_idx]))
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

    # draw comparison between ML types
    lower = 0.05
    upper = 2.5
    delta = 0.02
    x_data = np.arange(lower, upper, delta)
    data_list = []
    for idx in range(1, len(ML_types)):
        y_data = []
        ML = ML_types[idx]
        for idx1 in range(len(x_data)):
            y_data.append(ML_functions[ML](x_data[idx1]))
        data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        ML = ML_types[idx]
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker=list_markers[(idx+1)%markers_size], label=r"{0}".format(ML_names[idx+1]))
        #sns.scatterplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0} {1}".format(ga_functions[grammar]['grammar'], ga_functions[grammar][bound_type]['bound']))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)
    
    plt.legend(loc='best', fontsize=20)
    #if show_title :
    #    plt.title(r"$N={0}$".format(ga_functions[grammar]['N'][N_idx]))
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

def draw_ML_functions(h_ML, params, train_x, train_y, train_sigma, label):
    global list_markers, markers_size
    # draw only points
    lower = train_x[0]
    upper = train_x[len(train_x)-1]
    data_list = []
    y_data = []
    for idx1 in range(len(train_x)):
        y_data.append(h_ML(params, train_x[idx1]))
    data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.scatterplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0}".format(label))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    # draw points and lines
    lower = train_x[0]
    upper = train_x[len(train_x)-1]
    data_list = []
    y_data = []
    for idx1 in range(len(train_x)):
        y_data.append(h_ML(params, train_x[idx1]))
    data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=train_x, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0}".format(label))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    #draw values between observed points
    lower = 0.05
    upper = 2.5
    delta = 0.02
    x_data = np.arange(lower, upper, delta)
    data_list = []
    y_data = []
    for idx1 in range(len(x_data)):
        y_data.append(h_ML(params, x_data[idx1]))
    data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0}".format(label))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='upper left', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    #draw values between observed points indetail
    lower = 0.05
    upper = 2.5
    delta = 0.005
    x_data = np.arange(lower, upper, delta)
    data_list = []
    y_data = []
    for idx1 in range(len(x_data)):
        y_data.append(h_ML(params, x_data[idx1]))
    data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0}".format(label))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    #draw values near zero
    lower = 0.05
    upper = 0.2
    delta = 0.02
    x_data = np.arange(lower, upper, delta)
    data_list = []
    y_data = []
    for idx1 in range(len(x_data)):
        y_data.append(h_ML(params, x_data[idx1]))
    data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0}".format(label))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='upper left', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    #draw values large z
    lower = 1.95
    upper = 2.5
    delta = 0.02
    x_data = np.arange(lower, upper, delta)
    data_list = []
    y_data = []
    for idx1 in range(len(x_data)):
        y_data.append(h_ML(params, x_data[idx1]))
    data_list.append(y_data)

    train_x1 = []
    train_y1 = []
    train_sigma1 = []
    for idx in range(len(train_x)):
        if train_x[idx] >= lower and train_x[idx] <= upper:
            train_x1.append(train_x[idx])
            train_y1.append(train_y[idx])
            train_sigma1.append(train_sigma[idx])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=train_x1, y=train_y1, alpha=0.8, ax=ax, marker="o", color='gray',label="Observed Data")
    ax.errorbar(train_x1, train_y1, train_sigma1, color='tab:gray', ecolor='tab:gray')

    for idx in range(len(data_list)):
        sns.lineplot(x=x_data, y=data_list[idx], alpha=0.8, ax=ax, marker="o", label=r"{0}".format(label))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'$H(z)$', fontsize=20)
    ax.set_xlabel(r'$z$', fontsize=20)

    plt.legend(loc='best', fontsize=20)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

    draw_comparison_ML(train_x, train_y, train_sigma)


"""
  best fitted H(z) using dynesty
"""
# best fitting function for LCDM with varying Omega_k, Omega_L, H0
def hz_dynesty(x, params):
    k0 = params[0]
    L = params[1]
    H0 = params[2]
    m = 1 + k0 - L
    v = H0*np.sqrt(m*np.power(1+x, 3) - k0*np.power(1+x, 2) + L)
    return v

#define optimise functions
def fun(x, t, y, s):
    v = 0
    for idx in range(len(x)):
        v = v + x[idx]*np.power(t, idx)
    r = (v - y)/s
    return r
def fun_none(x, t, y, s):
    v = 0
    for idx in range(len(x)):
        v = v + x[idx]*np.power(t, idx)
    r = np.sum(((v - y)/s)**2)/2
    return r
def fun_ridge(x, t, y, s):
    regularization = 0.001
    v = 0
    for idx in range(len(x)):
        v = v + x[idx]*np.power(t, idx)
    penalty = 0
    for idx in range(len(x)):
        penalty = penalty + x[idx]*x[idx]
    r = (np.sum(((v - y)/s)**2)+regularization*penalty)/2
    return r
def fun_lasso(x, t, y, s):
    regularization = 0.001
    v = 0
    for idx in range(len(x)):
        v = v + x[idx]*np.power(t, idx)
    penalty = 0
    for idx in range(len(x)):
        penalty = penalty + np.abs(x[idx])
    r = (np.sum(((v - y)/s)**2)+regularization*penalty)/2
    return r

def fun_fit(t, w0,w1,w2,w3,w4,w5,w6):
    v = 0
    params = [w0,w1,w2,w3,w4,w5,w6]
    for idx in range(len(params)):
        v = v + params[idx]*np.power(t, idx)
    return v

def fun_fit_LCDM_series(t, H0,OL0):
    v = 0
    for idx in range(7):
        v = v + LCDM_coefficients[idx](H0, OL0)*np.power(t, idx)
    return v

def fun_fit_LCDM(t, H0,OL0):
    v = 0
    v = H0*np.sqrt((1-OL0)*(1+t)**3 + OL0)
    return v

# primitive function names
primitive_funtions = ["Constant", "x", "x2", "Sine", "Cosine", "Exp", "Log"]
primitive_operators = ["+", "-", "*", "/", "Power" ]
# binary tree structure for individuals
