using LinearAlgebra
using Plots
using ForwardDiff
using Interpolations, Dierckx, BenchmarkTools

println("Starting simulation...")

f = 1111 # frequency in Hz
w = 2*pi*f 
d = 0.002 
q = 1.602e-14
V_max = 300 # maximum possible 
epsilon_s = 8.854e-12 # permittivity of free space
R = 0.00005 # radius of lunar dust particle
n_segments = 25 # number of boundary segments for discretization


#scale of electrodes (calculated using Archimedes spiral)
n_1 = 0.65 
n_2 = 0.95
n_3 = 1.25

e_i_cnt = 10 # imaginary permittivity of cnts
e_r_cnt = 25 # real permittivity of cnts
complex_e_cnt = e_r_cnt - e_i_cnt # complex permittivity of cnts
complex_e_atsp = 2.3 # complex permittivity of atsp
e_i_ptfe = 0.006 # imaginary permittivity of ptfe
e_r_ptfe = 2.039 # real permittivity of ptfe
complex_e_ptfe = e_r_ptfe - e_i_ptfe #complex permittivity ptfe
complex_e_m = 0.95 * complex_e_atsp + 0.05 * complex_e_ptfe # complex permittivity of thermoplastic matrix

volume_fraction = 10 / 150 #fraction of cnts in total solution

#permittivity of coating
epsilon_p = complex_e_m * (1 + (3 * volume_fraction * (complex_e_cnt - complex_e_m)) / (complex_e_cnt + 2 * complex_e_m - volume_fraction * (complex_e_cnt - complex_e_m)))

time_range = 0:0.01:1.0
dt = 0.01

#voltage is pulsating dc current running in phases with 300 ms phase delay
function voltage_phases(w, t, V_max)
    return (V_max * sign(cos(w * t)) + 300) + (V_max * sign(cos(w * t + 2 * pi / 3)) + 300) + (V_max * sign(cos(w * t + 4 * pi / 3)) + 300)
end

#spiral geometry runs for two loops (from 0 to 4pi)
theta_range = range(0, stop=4*pi, length=div(n_segments, 3))

#discretizing Archimedean spirals in the domain
xboundary_ea = [n_1 * theta * cos(theta) for theta in theta_range]
xboundary_eb = [n_2 * theta * cos(theta) for theta in theta_range]
xboundary_ec = [n_3 * theta * cos(theta) for theta in theta_range]

#discretizing Archimedean spirals in the range
yboundary_ea = [n_1 * theta * sin(theta) for theta in theta_range]
yboundary_eb = [n_2 * theta * sin(theta) for theta in theta_range]
yboundary_ec = [n_3 * theta * sin(theta) for theta in theta_range]


println("Length of xboundary_ea: ", length(xboundary_ea))
println("Length of yboundary_ea: ", length(yboundary_ea))

#coordinates for reference point P
x_P = 10
y_P = 10

#matrices for green's function
G_phase13 = Float64[]
G_phase2 = Float64[]

#green's function
function greenfunction(x, y, x_i, y_i, x_j, y_j, x_k, y_k, t, w, V_max)
    #distances from P to P_i
    r1 = sqrt((x+x_i).^2 + (y-y_i).^2)
    r2 = sqrt((x-x_j).^2 + (y-y_j).^2)
    r3 = sqrt((x-x_k).^2 + (y-y_k).^2)
    r4 = sqrt((x+x_i).^2 + (y+y_i).^2)
    r5 = sqrt((x-x_j).^2 + (y+y_j).^2)
    r6 = sqrt((x-x_k).^2 + (y+y_k).^2)

    #voltage scaling
    G_13 = (1/(2*pi))*log(1/r1*r2*r3*r4*r5*r6) * voltage_phases(w, t, V_max)
    G_2 = ((1/(2*pi)))*log((r1*r2*r3)/r4*r5*r6) * voltage_phases(w, t, V_max)

    return G_13, G_2
end

accel_x_df = Float64[]
accel_y_df = Float64[]
vel_x_df = Float64[]
vel_y_df = Float64[]

function accelerationx(q, t, E_x_i, m)
    a_x_i = q*(E_x_i*cos(2*pi*f*t))*(1/m)
    push!(accel_x_df, a_x_i) 
    return a_x_i
end

function accelerationy(q, t, E_y_i, m, gravity)
    a_y_i = q*(E_y_i*cos(2*pi*f*t))*(1/m) - gravity
    push!(accel_y_df, a_y_i)
    return a_y_i
end

function sort_coordinates(xboundary, yboundary)
    xy_tuples = collect(zip(xboundary, yboundary))
    sorted_xy = sort(xy_tuples, by = first) 
    x_sorted, y_sorted = unzip(sorted_xy)    
    return x_sorted, y_sorted
end

function runge_kutta(q, t, E_x_i, E_y_i, gravity, m, delta_t, v_x_i, v_y_i)
    fivx = accelerationx(q, t, E_x_i, m)
    fiivx = accelerationx(q, t+(delta_t/2), E_x_i+(fivx*delta_t/2), m)
    fiiivx = accelerationx(q, t+(delta_t/2), E_x_i+(fiivx*delta_t/2), m)
    fivvx = accelerationx(q, t+delta_t, E_x_i+(fiiivx*delta_t), m)
    
    v_x_i = v_x_i + (1/6)*(fivx + 2*fiivx + 2*fiiivx + fivvx)
    
    fivy = accelerationy(q, t, E_y_i, m, gravity)
    fiivy = accelerationy(q, t+(delta_t/2), E_y_i+(fivy*delta_t/2), m, gravity)
    fiiivy = accelerationy(q, t+(delta_t/2), E_y_i+(fiivy*delta_t/2), m, gravity)
    fivvy = accelerationy(q, t+delta_t, E_y_i+(fiiivy*delta_t), m, gravity)
    
    v_y_i = v_y_i + (1/6)*(fivy + 2*fiivy + 2*fiiivy + fivvy)
    
    push!(vel_x_df, v_x_i)
    push!(vel_y_df, v_y_i)

    return v_x_i, v_y_i
end

function unzip(iterable)
    return map(x -> x[1], iterable), map(x -> x[2], iterable)
end

function ∂G13_∂x(G_13, x)
    return (-1).*ForwardDiff.derivative(x-> G_13, x)
end
function ∂G13_∂y(G_13, y)
    return (-1).*ForwardDiff.derivative(y-> G_13, y)
end
function ∂G2_∂x(G_2, x)
    return (-1).*ForwardDiff.derivative(x-> G_2, x)
end
function ∂G2_∂y(G_2, y)
    return (-1).*ForwardDiff.derivative(y-> G_2, y)
end

global v_x_i = 0.0
global v_y_i = 0.0

E_x = Float64[]
E_y = Float64[]

for time in time_range
    m = ((4/3) * pi * R.^3) * 3000
    coulomb_force = q * ((voltage_phases(w, time, V_max)) / d) * cos(w * time)
    gravity = m .* 1.625
    v_drag = 6 .* pi .* R .* 1.81e-5 .* v_y_i
    image_adhesion = q^2 / (16 * pi * epsilon_s * epsilon_p * (0.0001 + R)^2)

    for i in 1:div(n_segments, 3)
        x_i, y_i = xboundary_ea[i], yboundary_ea[i]
        x_j, y_j = xboundary_eb[i], yboundary_eb[i]
        x_k, y_k = xboundary_ec[i], yboundary_ec[i]

        G_13, G_2 = greenfunction(x_P, y_P, x_i, y_i, x_j, y_j, x_k, y_k, time, w, V_max)

        push!(G_phase13, G_13)
        push!(G_phase2, G_2)
    end

    for i in 1:7
        ∂G13x =(G_phase13[i + 1] - G_phase13[i]) / (xboundary_ea[i + 1] - xboundary_ea[i])
        ∂G13y =(G_phase13[i + 1] - G_phase13[i]) / (yboundary_ea[i + 1] - yboundary_ea[i])
        ∂G2x = (G_phase2[i + 1] - G_phase2[i]) / (xboundary_eb[i + 1] - xboundary_eb[i])
        ∂G2y = (G_phase2[i + 1] - G_phase2[i]) / (xboundary_eb[i + 1] - xboundary_eb[i])
        ∂G13x_ec = (G_phase13[i + 1] - G_phase13[i]) / (xboundary_ec[i + 1] - xboundary_ec[i])
        ∂G13y_ec = (G_phase13[i + 1] - G_phase13[i]) / (xboundary_ec[i + 1] - xboundary_ec[i])
    
        # Combine partial derivatives to get electric field components
        E_x_i = ∂G13x .+ ∂G2x .+ ∂G13x_ec
        E_y_i = ∂G13y .+ ∂G2y .+ ∂G13y_ec
    
        # Push the results into E_x and E_y
        push!(E_x, E_x_i)
        push!(E_y, E_y_i)
    end

    E2 = sqrt.(E_x .^ 2 .+ E_y .^ 2) 

    dielectrophoretic_force = 2 * pi * R.^3 * ((epsilon_p .- epsilon_s) / (epsilon_p .+ 2 * epsilon_s)) .* E2

    F = coulomb_force .+ dielectrophoretic_force .- image_adhesion .- gravity .-v_drag

    for i in 1:7
        global v_x_i, v_y_i = runge_kutta(q, time, E_x[i], E_y[i], gravity, m, dt, v_x_i, v_y_i)
    end
    average_F = mean(F)
    average_accelx = mean(accel_x_df)
    average_accely = mean(accel_y_df)
    average_velx = mean(vel_x_df)
    average_vely = mean(vel_y_df)
    println("$time, $average_F, $average_accelx, $average_accely, $average_velx, $average_vely")
end