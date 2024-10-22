import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

#title
st.title("Lunar Dust Simulation with EDS")

#givens
q = 1.602e-14  #lunar dust charge
epsilon_s = 8.854e-12  #permittivity of free space
e_i_cnt = 10 #imaginary permittivity of cnts
e_r_cnt = 25 #real permittivity of cnts
complex_e_cnt = e_r_cnt - e_i_cnt #complex permittivity of cnts

#initially set the submit button value to none to prevent errors
subbtn = None

#user inputs for modifiable parameters
f = st.number_input("Frequency (Hz)", format="%.6f") #frequency of applied voltage
d = st.number_input("Electrode Gap Distance (m)", format="%.6f") #gap between electrodes
V_max = st.number_input("Maximum Voltage (V)", format="%.6f") #maximum voltage
R = st.number_input("Radius of Lunar Dust Particle (m)", format="%.6f") #radius of lunar dust particle
volume_fraction = st.number_input("Volume Fraction of CNTs", format="%.6f") #volume fraction of cnts
n_segments = st.slider("Number of Line Segments In Spiral Geometry", format="%.6f") #number of line segments in the spiral electrode geometry
m1_conc = st.slider("Percentage of First Material", min_value=0.0, max_value=1.0, value=0.5, step=0.01) #concentration of first material
m2_conc = st.slider("Percentage of Second Material", min_value=0.0, max_value=1.0, value=0.5, step=0.01) #concentration of second material
m3_conc = st.slider("Percentage of Third Material", min_value=0.0, max_value=1.0, value=0.5, step=0.01) #concentration of third material
m4_conc = st.slider("Percentage of Fourth Material", min_value=0.0, max_value=1.0, value=0.5, step=0.01) #concentration of fourth material
m1_er = st.slider("Real Permittivity of First Material", min_value=0.0, max_value=100.0, value=0.5,step=0.01) #real permittivity of first material
m1_ei = st.slider("Imaginary Permittivity of First Material",min_value=0.0, max_value=100.0, value=0.5, step=0.01) #imaginary permittivity of first material
m2_er = st.slider("Real Permittivity of Second Material", min_value=0.0, max_value=100.0, value=0.5,step=0.01) #real permittivity of second material
m2_ei = st.slider("Imaginary Permittivity of Second Material", min_value=0.0, max_value=100.0, value=0.5,step=0.01) #imaginary permittivity of second material
m3_er = st.slider("Real Permittivity of Third Material",min_value=0.0, max_value=100.0, value=0.5, step=0.01) #real permittivity of third material
m3_ei = st.slider("Imaginary Permittivity of Third Material", min_value=0.0, max_value=100.0, value=0.5, step=0.01)#imaginary permittivity of third material
m4_er = st.slider("Real Permittivity of Fourth Material", min_value=0.0, max_value=100.0, value=0.5, step=0.01) #real permittivity of fourth material
m4_ei = st.slider("Imaginary Permittivity of Fourth Material", min_value=0.0, max_value=100.0, value=0.5, step=0.01) #imaginary permittivity of fourth material

#submit button to submit variables
if st.button("Submit Variables"):
  subbtn is not None
else:
  subbtn is None

#when variables are submitted
if subbtn is not None: 
    if f is None or d is None or V_max is None or R is None or volume_fraction is None or m1_conc is None or m1_ei is None or m1_er is None:
        print("Error: Please submit empty values that are required.")
        subbtn = None
    elif f <= 0 or d <= 0 or V_max <= 0 or R <= 0 or volume_fraction < 0:
        st.error("Error: Please enter valid positive values for all required parameters.")
        subbtn = None
    else:
        complex_e_m1 = m1_er - m1_ei 
        complex_e_m2 = m2_er - m2_ei 
        complex_e_m3 = m3_er - m3_ei 
        complex_e_m4 = m4_er - m4_ei
        complex_e_m = m1_conc * complex_e_m1 + m2_conc * complex_e_m2 \
            + m3_conc * complex_e_m3 + m4_conc * complex_e_m4
        
        #permittivity of coating
        epsilon_p = complex_e_m * (1 + (3 * volume_fraction * (complex_e_cnt - complex_e_m)) / (complex_e_cnt + 2 * complex_e_m - volume_fraction * (complex_e_cnt - complex_e_m)))

        #time range and dt
        time_range = np.arange(0, 1.0, 0.01)
        dt = 0.01

        #voltage phases function
        def voltage_phases(w, t, V_max):
            return (V_max * np.sign(np.cos(w * t)) + 300) + \
                (V_max * np.sign(np.cos(w * t + 2 * np.pi / 3)) + 300) + \
                (V_max * np.sign(np.cos(w * t + 4 * np.pi / 3)) + 300)

        #spiral geometry parameters
        n_1, n_2, n_3 = 0.65, 0.95, 1.25
        theta_range = np.linspace(0, 4 * np.pi, n_segments // 3)

        #discretizing Archimedean spirals
        xboundary_ea = n_1 * theta_range * np.cos(theta_range)
        xboundary_eb = n_2 * theta_range * np.cos(theta_range)
        xboundary_ec = n_3 * theta_range * np.cos(theta_range)
        yboundary_ea = n_1 * theta_range * np.sin(theta_range)
        yboundary_eb = n_2 * theta_range * np.sin(theta_range)
        yboundary_ec = n_3 * theta_range * np.sin(theta_range)

        #green's function
        def greenfunction(x, y, x_i, y_i, x_j, y_j, x_k, y_k, t, w, V_max):
            r1 = np.sqrt((x + x_i)**2 + (y - y_i)**2)
            r2 = np.sqrt((x - x_j)**2 + (y - y_j)**2)
            r3 = np.sqrt((x - x_k)**2 + (y - y_k)**2)
            r4 = np.sqrt((x + x_i)**2 + (y + y_i)**2)
            r5 = np.sqrt((x - x_j)**2 + (y + y_j)**2)
            r6 = np.sqrt((x - x_k)**2 + (y + y_k)**2)

            G_13 = (1 / (2 * np.pi)) * np.log(1 / (r1 * r2 * r3 * r4 * r5 * r6)) * voltage_phases(w, t, V_max)
            G_2 = (1 / (2 * np.pi)) * np.log((r1 * r2 * r3) / (r4 * r5 * r6)) * voltage_phases(w, t, V_max)

            return G_13, G_2

        v_x_i, v_y_i = 0.0, 0.0
        accel_x_df, accel_y_df = [], []
        vel_x_df, vel_y_df = [], []
        G_phase13, G_phase2 = [], []
        E_x, E_y = [], []

        #define acceleration functions
        def acceleration_x(q, t, E_x_i, m):
            a_x_i = q * (E_x_i * np.cos(2 * np.pi * f * t)) / m
            accel_x_df.append(a_x_i)  # Store the acceleration for x
            return a_x_i

        def acceleration_y(q, t, E_y_i, m, gravity):
            a_y_i = (q * (E_y_i * np.cos(2 * np.pi * f * t)) / m) - gravity
            accel_y_df.append(a_y_i)  # Store the acceleration for y
            return a_y_i

        #updated Runge-Kutta function
        def runge_kutta(q, t, E_x_i, E_y_i, gravity, m, delta_t, v_x_i, v_y_i):
            #calculate acceleration for x
            fivx = acceleration_x(q, t, E_x_i, m)
            fiivx = acceleration_x(q, t + (delta_t / 2), E_x_i + (fivx * delta_t / 2), m)
            fiiivx = acceleration_x(q, t + (delta_t / 2), E_x_i + (fiivx * delta_t / 2), m)
            fivvx = acceleration_x(q, t + delta_t, E_x_i + (fiiivx * delta_t), m)
            
            v_x_i += (1/6) * (fivx + 2 * fiivx + 2 * fiiivx + fivvx)
            
            #calculate acceleration for y
            fivy = acceleration_y(q, t, E_y_i, m, gravity)
            fiivy = acceleration_y(q, t + (delta_t / 2), E_y_i + (fivy * delta_t / 2), m, gravity)
            fiiivy = acceleration_y(q, t + (delta_t / 2), E_y_i + (fiivy * delta_t / 2), m, gravity)
            fivvy = acceleration_y(q, t + delta_t, E_y_i + (fiiivy * delta_t), m, gravity)
            
            v_y_i += (1/6) * (fivy + 2 * fiivy + 2 * fiiivy + fivvy)
            
            vel_x_df.append(v_x_i)  #store the updated velocity for x
            vel_y_df.append(v_y_i)  #store the updated velocity for y

            return v_x_i, v_y_i

        average_F = []
        average_accelx = []
        average_accely = []
        average_velx = []
        average_vely = []
        
        results_df = pd.DataFrame(columns=['average_F', 'average_accelx', 'average_accely', 'average_velx', 'average_vely'])

        #run the simulation
        for time in time_range:
            w = 2 * np.pi * f
            m = (4/3) * np.pi * R**3 * 3000  #mass of the lunar dust particle
            coulomb_force = q * (voltage_phases(w, time, V_max) / d) * np.cos(w * time)
            gravity = m * 1.625
            v_drag = 6 * np.pi * R * 1.81e-5 * v_y_i
            image_adhesion = q**2 / (16 * np.pi * epsilon_s * epsilon_p * (0.0001 + R)**2)

            for i in range(n_segments // 3):
                x_i, y_i = xboundary_ea[i], yboundary_ea[i]
                x_j, y_j = xboundary_eb[i], yboundary_eb[i]
                x_k, y_k = xboundary_ec[i], yboundary_ec[i]

                G_13, G_2 = greenfunction(10, 10, x_i, y_i, x_j, y_j, x_k, y_k, time, w, V_max)
                G_phase13.append(G_13)
                G_phase2.append(G_2)

            #calculate electric field components
            for i in range(n_segments // 3 - 1):  #ensure you don't go out of bounds
                #calculate differences for the electric field components
                dG13x = (G_phase13[i + 1] - G_phase13[i]) / (xboundary_ea[i + 1] - xboundary_ea[i])
                dG13y = (G_phase13[i + 1] - G_phase13[i]) / (yboundary_ea[i + 1] - yboundary_ea[i])
                dG2x = (G_phase2[i + 1] - G_phase2[i]) / (xboundary_eb[i + 1] - xboundary_eb[i])
                dG2y = (G_phase2[i + 1] - G_phase2[i]) / (yboundary_eb[i + 1] - yboundary_eb[i])
                dG13x_ec = (G_phase13[i + 1] - G_phase13[i]) / (xboundary_ec[i + 1] - xboundary_ec[i])
                dG13y_ec = (G_phase13[i + 1] - G_phase13[i]) / (yboundary_ec[i + 1] - yboundary_ec[i])

                #electric field components
                E_x_i = dG13x + dG2x + dG13x_ec
                E_y_i = dG13y + dG2y + dG13y_ec
                
                E_x.append(E_x_i)
                E_y.append(E_y_i)
                runge_kutta(q, time, E_x_i, E_y_i, gravity, m, dt, v_x_i, v_y_i)
            E_x_array = np.array(E_x)
            E_y_array = np.array(E_y)
            E2 = np.sqrt(E_x_array ** 2 + E_y_array ** 2)
            dielectrophoretic_force = 2 * np.pi * R**3 * ((epsilon_p-epsilon_s)/(epsilon_p+2*epsilon_s)) * E2  
            F = coulomb_force + dielectrophoretic_force - image_adhesion - gravity - v_drag

            result_row = [np.mean(F), np.mean(accel_x_df), np.mean(accel_y_df), np.mean(vel_x_df), np.mean(vel_y_df)]
            results_df.loc[len(results_df.index)] = result_row
        formatted_df = results_df.applymap(lambda x: f"{x:.12e}")  #adjust the precision here

        st.write(formatted_df.head())
        
        
