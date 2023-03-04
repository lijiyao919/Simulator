import matplotlib.pyplot as plt
import matplotlib as mpl

num = [500, 600, 700, 800, 900]
time = [2, 4, 6, 8, 10]
xlabel = "The Number of Agents"
mpl.rcParams['font.size'] = 20
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 6
plt.rcParams['figure.figsize'] = [12, 9]

def plot1(ylabel, yticks=None):
    plt.figure(0)
    plt.plot(num, random, 'gp-', label="Random")
    plt.plot(num, SD_hueristic, 'co-', label="Soft SD-Heuristic")
    plt.plot(num, as_ddqn, 'b*-', label="AS-DDQN")
    plt.plot(num, soft_cdqn, 'y^-', label="Soft cDQN")
    plt.plot(num, bsac_nn, 'kh-', label="BS-AC + MLP")
    plt.plot(num, bsac_rnn_pdsa, 'rs-', label="BS-AC + TSD-Net + PDSA")
    plt.xticks(num)
    if yticks is not None:
        plt.yticks(yticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.legend()
    plt.show()

def plot2(ylabel, yticks=None):
    plt.figure(0)
    plt.plot(time, morning, 'b*-', label="Morning")
    plt.plot(time, noon, 'kh-', label="Noon")
    plt.plot(time, evening, 'rs-', label="Evening")
    plt.xticks(time)
    if yticks is not None:
        plt.yticks(yticks)
    plt.xlabel("Patient Duration (Minute)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


#Morning
#TSR
'''random = [47.48, 55.28, 62.31, 65.10, 71.57]
SD_hueristic = [69, 73.57, 79.84, 81.35, 86.89]
as_ddqn = [77, 84, 88, 89, 90.5]
soft_cdqn = [81, 88, 91.2, 92.5, 93.7]
bsac_nn = [81.5, 89.4, 92.8, 94.7, 95.86]
bsac_rnn_pdsa = [83, 90.5, 94.32, 95.84, 96.58]
plot1("TSR (%)", [50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

#ART
random = [9.2, 9.1, 8.9, 8.74, 7.66]
SD_hueristic = [7.63, 7.02, 5.28, 5.14, 4.26]
as_ddqn = [5, 4.3, 3.8, 2.5, 2]
soft_cdqn = [4.73, 3.9, 1.8, 1.7, 1.7]
bsac_nn = [4.7, 3.2, 2.6, 2.78, 2.34]
bsac_rnn_pdsa = [1.6, 1.4, 1.3, 1.3, 1.25]
plot1("ART (Minutes)")

#ARPT
random = [14.89, 16, 17.33, 20.11, 20.87]
SD_hueristic = [6.51, 9.02, 10.47, 13.41, 14.73]
as_ddqn = [4.6, 6.4, 8.3, 11.5, 13.8]
soft_cdqn = [3.7, 5.2, 7.47, 10, 12.3]
bsac_nn = [3.5, 4.7, 6.8, 9, 11.5]
bsac_rnn_pdsa = [3, 4.6, 6.34, 8.7, 11]
plot1("ARPT (Minutes)")'''


#Noon
#TSR
'''random = [42.95, 45, 50.33, 55.92, 61.34]
SD_hueristic = [56, 62, 68, 75.96, 78.56]
as_ddqn = [74, 83.5, 90, 92, 93.1]
soft_cdqn = [75, 86.7, 92, 93, 93.8]
bsac_nn = [74.7, 87.1, 93, 94.2, 95]
bsac_rnn_pdsa = [78.7, 90.63, 94.5, 95.5, 96.5]
plot1("TSR (%)", [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

#ART
random = [9.22,	9.2, 9.2, 9.03,	8.7]
SD_hueristic = [9.4, 9.14, 8.66, 8.16, 7.8]
as_ddqn = [8.4, 7, 3.9, 3.7, 2.3]
soft_cdqn = [7.1, 5.3, 4.2, 2.3, 2.2]
bsac_nn = [7.8, 5.5, 2.42, 2.1, 2.1]
bsac_rnn_pdsa = [2.2, 1.7, 1.4, 1.32, 1.26]
plot1("ART (Minutes)")

#ARPT
random = [12.15, 16.12, 17.1, 17.96, 18.77]
SD_hueristic = [6.58, 8.1, 9.62, 10.1, 12.1]
as_ddqn = [2, 2.9, 4.3, 6.2, 8.3]
soft_cdqn = [1.7, 2.2, 3.9, 5.8, 8]
bsac_nn = [1.8, 2.3, 3.6, 5.7, 8]
bsac_rnn_pdsa = [1.2, 1.7, 3.4, 5.3, 7.2]
plot1("ARPT (Minutes)")'''


#Evening
#TSR
'''random = [28.60, 31.67, 35.51, 39.35, 42.56]
SD_hueristic = [47.00, 51.20, 55.29, 58.04, 60]
as_ddqn = [60.60, 68, 73, 78, 81.00]
soft_cdqn = [61.50, 69, 77, 82, 88.00]
bsac_nn = [62.70, 70, 78, 85, 89.50]
bsac_rnn_pdsa = [64.20, 74.10, 82.70, 89.20, 91.8]
plot1("TSR (%)", [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])

#ART
random = [8.35, 8.37, 8.43, 8.44, 8.47]
SD_hueristic = [8.43, 8.1, 7.69, 7.6, 7.3]
as_ddqn = [7.6, 7, 5.3, 5.5, 4.8]
soft_cdqn = [7.9, 6.9, 6.2, 4.6, 3.8]
bsac_nn = [7.8, 7.2, 5.8, 4.6, 4.4]
bsac_rnn_pdsa = [2.2, 1.5, 1.53, 1.3, 1.15]
plot1("ART (Minutes)")

#ARPT
random = [21.79, 24.88, 26.3, 27.76, 29.5]
SD_hueristic = [8.4, 10.24, 12.3, 14.43, 16.81]
as_ddqn = [3.3, 4.5, 5.8, 7.3, 9.2]
soft_cdqn = [3.1, 4.2, 5, 6.2, 7.3]
bsac_nn = [2.9, 4, 4.7, 5.7, 6.7]
bsac_rnn_pdsa = [2.6, 3.1, 3.9, 4.95, 6.5]
plot1("ARPT (Minutes)")'''


morning = [93.8, 94.3, 96.34, 96.58, 96.58]
noon = [92.5, 93, 94.89, 95.6, 96.5]
evening = [90.1, 90.2, 90.8, 90.9, 91.8]
plot2("TSR (%)")

morning = [12.14, 12.1, 11.34, 11.3, 11]
noon = [8.3, 8, 7.42, 7.42, 7.2]
evening = [7, 6.96, 6.7, 6.7, 6.5]
plot2("ARPT (Minute)")






