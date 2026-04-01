import matplotlib.pyplot as plt
import os

Tloss1=[]
Tloss2=[]
Oloss=[]
Ploss=[]
j=[]
gamma=[]

for i in range(1,100):
    with open(f'C:/Users/Ohay/Desktop/Shakiba/PIDL_DQ - Final - single weight/log{i-1}_NN1.txt', 'r') as f:
        contents = f.read()
        Tloss1_index = contents.find('Total Loss: ')
        Tloss1.append(float(contents[Tloss1_index + len('Total Loss: '):].split()[0]))
        # Oloss_index = contents.find('Loss_O: ')
        # Oloss.append(float(contents[Oloss_index + len('Loss_O: '):].split()[0].replace(',','')))
        # Ploss_index = contents.find('Loss_P: ')
        # Ploss.append(float(contents[Ploss_index + len('Loss_P: '):].split()[0].replace(',','')))
    if os.path.exists(f'C:/Users/Ohay/Desktop/Shakiba/PIDL_DQ - Final - single weight/log{i - 1}_NN2.txt'):
        j.append(i)
        with open(f'C:/Users/Ohay/Desktop/Shakiba/PIDL_DQ - Final - single weight/log{i - 1}_NN2.txt', 'r') as g:
            contents = g.read()
            Tloss2_index = contents.find('Total Loss: ')
            Tloss2.append(float(contents[Tloss2_index + len('Total Loss: '):].split()[0].replace(',','')))
            gamma_index = contents.find('Gamma: ')
            gamma.append(float(contents[gamma_index + len('Gamma: '):].split()[0]))

fig, ax = plt.subplots()
ax.plot(list(range(1,i+1)), Tloss1, label="Total Loss NN#1")
# ax.plot(list(range(1,i+1)), Oloss, label="Observation Loss")
# ax.plot(list(range(1,i+1)), Ploss, label="Physics Loss")
ax.plot(j, Tloss2, label="Total Loss NN#2")

ax.set_xlabel('epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss Function Values')
ax.legend(loc='upper right')
plt.show()
fig.savefig('C:/Users/Ohay/Desktop/Shakiba/PIDL_DQ - Final - single weight/Totalloss.png')

fig1, ax1 = plt.subplots()
ax1.plot(j, gamma)
ax1.set_xlabel('epoch')
ax1.set_ylabel('gamma')
ax1.set_title('Gamma values')
fig1.savefig('C:/Users/Ohay/Desktop/Shakiba/PIDL_DQ - Final - single weight/gamma.png')

