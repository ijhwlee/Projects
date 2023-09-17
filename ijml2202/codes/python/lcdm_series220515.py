import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
LCDM function serie sexpansion related function
"""

def LCDM_series_0(H0, OL):
    return H0

def LCDM_series_1(H0, OL):
    v = -(3/2)* H0* (-1+OL)
    return v

def LCDM_series_2(H0, OL):
    v = -(3/8)* H0 *(-1+OL)* (1+3* OL)
    return v

def LCDM_series_3(H0, OL):
    v = -(1/16) *H0* (-1+OL)* (-1-18* OL+27* OL**2)
    return v

def LCDM_series_4(H0, OL):
    v = -(3/128) *H0* (-1+OL)**2* (-1-54* OL+135* OL**2)
    return v

def LCDM_series_5(H0, OL):
    v = -(3/256)* H0* (-1+OL)**2* (1+117* OL-621* OL**2+567* OL**3)
    return v

def LCDM_series_6(H0, OL):
    v = -((H0* (-1+OL)**2 *(-7-1512 *OL+13554* OL**2-27216* OL**3+15309* OL**4))/1024)
    return v

def LCDM_series_7(H0, OL):
    v = -((9* H0* (-1+OL)**3 *(-1+9 *OL)* (1+369* OL-1197* OL**2+891* OL**3))/2048)
    return v

def LCDM_series_8(H0, OL):
    v = -((9* H0 *(-1+OL)**3 *(11+6105 *OL-111474* OL**2+460242* OL**3-665577* OL**4+312741* OL**5))/32768)
    return v

def LCDM_series_9(H0, OL):
    v = -((H0* (-1+OL)**3 *(-143-115830* OL+2915055* OL**2-17277300* OL**3+39814335* OL**4-39405366* OL**5+14073345* OL**6))/65536)
    return v

def LCDM_series_10(H0, OL):
    v = -((33* H0* (-1+OL)**4* (-13-14742* OL+477009* OL**2-3455460* OL**3+9410661* OL**4-10746918* OL**5+4349943* OL**6))/262144)
    return v

def LCDM_series_11(H0, OL):
    v = -(1/524288)*3* H0* (-1+OL)**4 *(221+338793 *OL-14231295* OL**2+136682397* OL**3-520600041* OL**4+928584891* OL**5-778639797* OL**6+247946751* OL**7)
    return v

def LCDM_series_12(H0, OL):
    v = -(1/4194304)*H0* (-1+OL)**4* (-4199-8465184* OL+450317556* OL**2-5526881424* OL**3+27716415246* OL**4-68927661504* OL**5+90183018276 *OL**6-59507220240* OL**7+15620645313 *OL**8)
    return v

def LCDM_series_13(H0, OL):
    v = -(1/8388608)*3 *H0 *(-1+OL)**5 *(-2261-5860512* OL+381037932* OL**2-5526881424* OL**3+31980479130* OL**4-90136172736 *OL**5+131805949788* OL**6-96127048080 *OL**7+27636526323 *OL**8)
    return v

def LCDM_series_14(H0, OL):
    v = -(1/33554432)*3* H0* (-1+OL)**5* (7429+24270543* OL-1932817788* OL**2+34494659676* OL**3-250256026170* OL**4+915460227234* OL**5-1844901840492* OL**6+2078861851692 *OL**7-1227851383779 *OL**8+296105639175 *OL**9)
    return v

def LCDM_series_15(H0, OL):
    v = -(1/67108864)*H0* (-1+OL)**5* (-37145-150437250 *OL+14463037215* OL**2-311730051960* OL**3+2762960993550 *OL**4-12622673966508* OL**5+32944049044470* OL**6-51102512002872 *OL**7+46614923756523* OL**8-23096239855650 *OL**9+4796911354635 *OL**10)
    return v

def LCDM_series_16(H0, OL):
    v = -(1/2147483648)*459* H0* (-1+OL)**6 *(-2185-10815750* OL+1228885515* OL**2-30561769800* OL**3+306995665950* OL**4-1567521603684* OL**5+4521732221790* OL**6-7682076967752 *OL**7+7616817607275 *OL**8-4075807033350 *OL**9+909218492055 *OL**10)
    return v

def LCDM_series_17(H0, OL):
    v = -(1/4294967296)*27* H0* (-1+OL)**6* (63365+378479145* OL-50806120365* OL**2+1491703028775* OL**3-17829420555150* OL**4+109982527576506 *OL**5-392666455448442* OL**6+857426988013902* OL**7-1162650728341575* OL**8+955254209957325 *OL**9-435515657694345* OL**10+84557319761115 *OL**11)
    return v

def LCDM_series_18(H0, OL):
    v = -(1/17179869184)*3* H0* (-1+OL)**6* (-1964315-14001637320* OL+2199423862350* OL**2-75379723826400 *OL**3+1056694985000775* OL**4-7725211153810416* OL**5+33237551752365492 *OL**6-89686155550957344 *OL**7+156156431640027075* OL**8-175371316842040200 *OL**9+122739041116472670 *OL**10-48705016182402240 *OL**11+8371174656350385 *OL**12)
    return v

def LCDM_series_19(H0, OL):
    v = -(1/34359738368)*9* H0* (-1+OL)**7* (-1137235-9580067640 *OL+1736387259750 *OL**2-67445016055200 *OL**3+1056694985000775* OL**4-8538391275264144 *OL**5+40234931068652964 *OL**6-118008099409154400 *OL**7+221906508120038475* OL**8-267672009916798200 *OL**9+200258435505823830 *OL**10-84592922843119680 *OL**11+15420584893277025 *OL**12)
    return v

def LCDM_series_20(H0, OL):
    v = -(1/274877906944)*9* H0* (-1+OL)**7* (7960645+78547684215* OL-16408260850410* OL**2+732628320234570* OL**3-13237874006038425* OL**4+124313443389443637 *OL**5-689165820773517276 *OL**6+2420848696930772220 *OL**7-5595379245220939965 *OL**8+8625735017807677425 *OL**9-8783609659063099290 *OL**10+5673894064446329370 *OL**11-2106451896421641615 *OL**12+342336984630749955 *OL**13)
    return v

# array for LCDM series expansion coefficients
LCDM_coefficients = [LCDM_series_0,LCDM_series_1,LCDM_series_2, LCDM_series_3, LCDM_series_4, LCDM_series_5,
                     LCDM_series_6,LCDM_series_7,LCDM_series_8, LCDM_series_9, LCDM_series_10, LCDM_series_11,
                     LCDM_series_12,LCDM_series_13,LCDM_series_14, LCDM_series_15, LCDM_series_16, LCDM_series_17,
                     LCDM_series_18,LCDM_series_19,LCDM_series_20]

list_markers = ["o", "s", "v", "^", "<", ">", "X"]
markers_size = 7

def draw_LCDM_coeffs(idxs, OLs, zs, showTitle = False):
    global list_markers, markers_size
    # draw LCDM coeffcients
    markers = []
    data_list = []
    for idx in range(len(OLs)):
        y_data = []
        for idx1 in range(len(idxs)):
            y_data.append(LCDM_coefficients[idx1](1.0, OLs[idx]))
        data_list.append(y_data)
        markers.append(list_markers[idx])

    fig, ax = plt.subplots(figsize=(10,8))

    for idx in range(len(data_list)):
        sns.lineplot(x=idxs, y=data_list[idx], alpha=0.8, ax=ax, marker=markers[idx%markers_size], label=r"$\Omega_\Lambda^0 = {0:.2f}$".format(OLs[idx]))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(r'Coeff.', fontsize=20)
    ax.set_xlabel(r'$n$', fontsize=20)
    
    plt.legend(loc='best', fontsize=20)
    fig.tight_layout()
    plt.show()

    # draw LCDM series term contibution
    #zs = np.arange(0, 3.0, 0.5)
    for idx0 in range(len(zs)):
        z = zs[idx0]
        markers = []
        data_list = []
        for idx in range(len(OLs)):
            y_data = []
            for idx1 in range(len(idxs)):
                if z > 0 :
                    y_data.append(np.abs(LCDM_coefficients[idx1](1.0, OLs[idx])*np.power(z, idx1)))
                else:
                    y_data.append(LCDM_coefficients[idx1](1.0, OLs[idx])*np.power(z, idx1))
            data_list.append(y_data)
            markers.append(list_markers[idx])

        fig, ax = plt.subplots(figsize=(10,8))

        for idx in range(len(data_list)):
            sns.lineplot(x=idxs, y=data_list[idx], alpha=0.8, ax=ax, marker=markers[idx%markers_size], label=r"$\Omega_\Lambda^0 = {0:.2f}$".format(OLs[idx]))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylabel(r'Term Value', fontsize=20)
        ax.set_xlabel(r'$n$', fontsize=20)
        if z > 0 :
            ax.set_yscale('log')
        if showTitle:
            ax.set_title(r'$z = {0:.2f}$'.format(z))
        
        plt.legend(loc='best', fontsize=20)
        fig.tight_layout()
        plt.show()
