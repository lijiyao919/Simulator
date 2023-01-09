import matplotlib.pyplot as plt
import matplotlib as mpl

num = [462, 616, 770]
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 6

#serving rate
random = [54, 62, 69]
demand_based = [65, 75, 82]
am_dqn = [67, 78, 85]
as_dqn = [69, 80, 86]
pr_ddqn = [73, 83, 90]

plt.figure(0)
plt.plot(num, random, 'gp-', label="Random")
plt.plot(num, demand_based, 'co-', label="Demand Heuristic")
plt.plot(num, am_dqn, 'b*-', label="AM-DQN")
plt.plot(num, as_dqn, 'k^-', label="AS-DQN")
plt.plot(num, pr_ddqn, 'rs-', label="Pr-DDQN")
plt.xticks(num)
plt.xlabel("Driver Number")
plt.ylabel("Service Rate (%)")
plt.legend()
plt.show()


#requesting time
random = [5, 4.5, 4]
demand_based = [3.7, 3, 2.6]
am_dqn = [3.5, 2.5, 1.5]
as_dqn = [3, 2.3, 1.5]
pr_ddqn = [2.5, 1.5, 1]

plt.figure(0)
plt.plot(num, random, 'gp-', label="Random")
plt.plot(num, demand_based, 'co-', label="Demand Heuristic")
plt.plot(num, am_dqn, 'b*-', label="AM-DQN")
plt.plot(num, as_dqn, 'k^-', label="AS-DQN")
plt.plot(num, pr_ddqn, 'rs-', label="Pr-DDQN")
plt.xticks(num)
plt.xlabel("Driver Number")
plt.ylabel("Requesting Time (Minutes)")
plt.legend()
plt.show()


#navigation time
random = [15, 20, 24]
demand_based = [11, 14, 18]
am_dqn = [10, 13, 17]
as_dqn = [9, 12.5, 16.5]
pr_ddqn = [6, 10, 13]

plt.figure(0)
plt.plot(num, random, 'gp-', label="Random")
plt.plot(num, demand_based, 'co-', label="Demand Heuristic")
plt.plot(num, am_dqn, 'b*-', label="AM-DQN")
plt.plot(num, as_dqn, 'k^-', label="AS-DQN")
plt.plot(num, pr_ddqn, 'rs-', label="Pr-DDQN")
plt.xticks(num)
plt.xlabel("Driver Number")
plt.ylabel("Navigaton Time (Minutes)")
plt.legend()
plt.show()