import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    gravity_compareson()
    #forces_compareson()

def forces_compareson()-> None:

    t_n, r_n = read_csv('residual_value_nominal.csv')
    t_f, r_f = read_csv('residual_value_forces.csv')

    #print(r)
    
    # 
    #idx = 0
    #plt.plot(t_f[idx], r_f[idx], linestyle = '--', linewidth = 2)

    '''
    for idx in [1, 3, 4, 5, 6, 11, 12, 14]:
        t_f[idx], r_f[idx] = [], []
    '''
    for t, r in zip(t_f, r_f):
        pert_plt, = plt.plot(t, r, linestyle = '--', linewidth = 1)
    
    max_len = 500
    nom_plt, = plt.plot(t_n[0][0:max_len], r_n[0][0:max_len], linewidth = 2, color = 'r')
    #plt.plot(t_n[1][0:100], r_n[1][0:100])

    bound_plot, = plt.plot(t_n[0][0:max_len], [1 for _ in range(max_len)], linestyle = ':', linewidth = 2, color = 'k')
    plt.plot(t_n[0][0:max_len], [-1 for _ in range(max_len)], linestyle = ':', linewidth = 2, color = 'k')

    plt.title("Residual signal with external forces")
    plt.xlabel("time [s]")
    plt.ylabel("r(t)")

    plt.grid()

    plt.xlim([0, t_n[0][max_len]])

    plt.legend([pert_plt, nom_plt, bound_plot], ['With external forces', 'Nominal conditions', 'Threshold'])

    plt.savefig('residual_forces.pdf', format = "pdf")
    plt.show()

def gravity_compareson() -> None:
    t_n, r_n = read_csv('residual_value_nominal.csv')
    t_g, r_g = read_csv('residual_value_gravity.csv')

    max_len = 138

    #print(r)
    
    # 1, 3, 4, 5, 6, 11, 12, 14
    #idx = 14
    #plt.plot(t_g[idx], r_g[idx], linestyle = '--', linewidth = 2)
    for idx in [1, 3, 4, 5, 6, 11, 12, 14]:
        t_g[idx], r_g[idx] = [], []
    
    for t, r in zip(t_g, r_g):
        pert_plt, = plt.plot(t[0:max_len], r[0:max_len], linestyle = '--', linewidth = 2)
    
    nom_plt, = plt.plot(t_n[0][0:max_len], r_n[0][0:max_len], linewidth = 2, color = 'r')
    #plt.plot(t_n[1][0:100], r_n[1][0:100])

    bound_plot, = plt.plot(t_n[0][0:max_len], [1 for _ in range(max_len)], linestyle = ':', linewidth = 2, color = 'k')
    plt.plot(t_n[0][0:max_len], [-1 for _ in range(max_len)], linestyle = ':', linewidth = 2, color = 'k')

    plt.title("Residual signal with slope ground")
    plt.xlabel("time [s]")
    plt.ylabel("r(t)")

    plt.grid()

    plt.xlim([0, t_n[0][max_len]])

    plt.legend([pert_plt, nom_plt, bound_plot], ['With slop ground', 'Nominal conditions', 'Threshold'])

    plt.savefig('residual_gravity.pdf', format = 'pdf')
    plt.show()
    

def read_csv(file : str) -> tuple[list[float], list[float]]:
    

    
    tt, rr, t, r =[], [], [], []

    print("reading file")

    with open(file, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:

            if lines[0] == 'time': continue
            if float(lines[0]) == 0: 
                tt.append(t)
                rr.append(r)
                t, r = [], []
                
            t.append(float(lines[0]))
            r.append(float(lines[1]))

    tt.append(t)
    rr.append(r)
    print("file readed")

    return tt[1:], rr[1:]


if __name__ == '__main__':
    main()