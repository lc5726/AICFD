import numpy as np 

total = np.loadtxt('total_bc_0.00_0.10',skiprows=1)
airfoil = np.loadtxt('airfoil_bc_0.00_0.10',skiprows=1)
inlet = np.loadtxt('inlet_bc_0.00_0.10',skiprows=1)
outlet = np.loadtxt('outlet_bc_0.00_0.10',skiprows=1)

# Get SDF for all boundaries

for j in range(40106):
    try:
        x_t,y_t = total[j][1],total[j][2]
        SDF_Array_inlet = []
        for i in range(1000000):
            try:
                distance = np.sqrt((x_t - inlet[i][1])**2 + (x_t - inlet[i][2])**2)
                SDF_Array_inlet.append(distance)
            except IndexError:
                break
    except IndexError:
        break
    print(min(SDF_Array_inlet))

for j in range(1):
    try:
        x_t,y_t = total[j][1],total[j][2]
        SDF_Array_outlet = []
        for i in range(1000000):
            try:
                distance = np.sqrt((x_t - outlet[i][1])**2 + (x_t - outlet[i][2])**2)
                SDF_Array_outlet.append(distance)
            except IndexError:
                break
    except IndexError:
        break
    print(min(SDF_Array_outlet))

for j in range(1):
    try:
        x_t,y_t = total[j][1],total[j][2]
        SDF_Array_airfoil = []
        for i in range(1000000):
            try:
                distance = np.sqrt((x_t - airfoil[i][1])**2 + (x_t - airfoil[i][2])**2)
                SDF_Array_airfoil.append(distance)
            except IndexError:
                break
    except IndexError:
        break
    print(min(SDF_Array_airfoil))