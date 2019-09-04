"""
===========
Slider Demo
===========

Using the slider widget to control visual properties of your plot.

In this example, a slider is used to choose the frequency of a sine
wave. You can control many continuously-varying properties of your plot in
this way.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from sklearn.preprocessing import minmax_scale

mixed = False

alpha = [.5,.5,.5]

def normalize_list_numpy(list_numpy):
    normalized_list = minmax_scale(list_numpy)
    return normalized_list


def util_progress(x,scale=.2,thresh=8):
    return np.tanh(5*(x-5))
    #return np.tanh(5*(x-5)) if x < 100 else np.tanh(.5*(105-x)) 

def util_reaction(time_sec):
    return np.cos(np.radians(60*time_sec))

def util_dist(dist_m,thresh):
    x = dist_m
    #return 1.5/(1+math.exp(-1 * (x-0))) - 0.5 -1 
    return np.tanh(.8*(x-(thresh-7)))

def util_ttc(ttc_sec,thresh):
    if ttc_sec > 100:
        return 1
    x = ttc_sec
    #return 1.5/(.72+math.exp(-2 * (x-thresh))) - 1.03 -1
    return np.tanh(1.5*(x-thresh))

def get_beh_label(l1,l2,l3):
    lab = None
    if l1 < 0 and l2 < 0 and l3 > 0:
        lab = 'B1: close dist, low ttc '
    elif l1 > 0 and l2 < 0 and l3 > 0:
        lab = 'B2: close dist, high ttc'
    elif l1 > 0 and l2 > 0 and l3 < 0:
        lab = 'B3: low speed, safe dist, high ttc'
    elif l1 < 0 and l2 > 0 and l3 < 0:
        lab = 'B4: low speed, safe dist, low ttc'
    elif l1 < 0 and l2 < 0 and l3 < 0:
        lab = 'B5: low speed, close dist, low ttc'
    elif l1 > 0 and l2 < 0 and l3 < 0:
        lab = 'B6: low speed, close dist, high ttc'
    elif l1 > 0 and l2 > 0 and l3 > 0:
        lab = 'B7: safe dist, high ttc'
    elif l1 < 0 and l2 > 0 and l3 > 0:
        lab = 'B8: safe dist, low ttc'
    else:
        lab = 'Random'    
    return lab
        
l = [-1.6,6,-6]
l2 = [30,-6,6]
#fig, ax = plt.subplots(3,3)

ax11 = plt.subplot(331)
plt.tight_layout()
ax11.set_xlabel('ttc (s)')
ax11.set_ylabel('utility')
ax11.set_title('safety (ttc)')
ax12 = plt.subplot(332)
ax12.set(xlabel='dist (m)')
ax12.set_title('safety (distance gap)')
ax13 = plt.subplot(333)
ax13.set(xlabel='vel (kph)')
ax13.set_title('progress')

ax21 = plt.subplot(334)
ax21.set_ylabel('likelihood')
ax21.set_yticklabels([])
ax21.set_xlabel('utility')
ax22 = plt.subplot(335)
ax22.set_yticklabels([])
ax22.set_xlabel('utility')
ax23 = plt.subplot(336)
ax23.set_yticklabels([])
ax23.set_xlabel('utility')

ax31 = plt.subplot(337)
ax32 = plt.subplot(338)
ax32.set_title('p(vel_lc,dist | vel_s,$\Lambda)$',fontsize=8)
ax33 = plt.subplot(339)

plt.subplots_adjust(left=0.25, bottom=0.25)

delta_f = 0.1

U = np.arange(-1,1.001,0.001)

if mixed:
    P_U_1 = [(l[0]*np.exp(l[0]*(u+1))) / (np.exp(2*l[0]) -1) if l[0]!=0 else 1/2000 for u in U]
    P_U_2 = [(l2[0]*np.exp(l2[0]*(u+1))) / (np.exp(2*l2[0]) -1) if l2[0]!=0 else 1/2000 for u in U]
    P_U = [a[0]+a[1] for a in list(zip(P_U_1,P_U_2))]
else:
    P_U = [(l[0]*np.exp(l[0]*(u+1))) / (np.exp(2*l[0]) -1) if l[0]!=0 else 1/2000 for u in U]
_f10, = ax21.plot(U, P_U, lw=2)

if mixed:
    P_U_1 = [(l[1]*np.exp(l[1]*(u+1))) / (np.exp(2*l[1]) -1) if l[1]!=0 else 1/2000 for u in U]
    P_U_2 = [(l2[1]*np.exp(l2[1]*(u+1))) / (np.exp(2*l2[1]) -1) if l2[1]!=0 else 1/2000 for u in U]
    P_U = [a[0]+a[1] for a in list(zip(P_U_1,P_U_2))]
else:
    P_U = [(l[1]*np.exp(l[1]*(u+1))) / (np.exp(2*l[1]) -1) if l[1]!=0 else 1/2000 for u in U]
_f11, = ax22.plot(U, P_U, lw=2)

if mixed:
    P_U_1 = [(l[2]*np.exp(l[2]*(u+1))) / (np.exp(2*l[2]) -1) if l[2]!=0 else 1/2000 for u in U]
    P_U_2 = [(l2[2]*np.exp(l2[2]*(u+1))) / (np.exp(2*l2[2]) -1) if l2[2]!=0 else 1/2000 for u in U]
    P_U = [a[0]+a[1] for a in list(zip(P_U_1,P_U_2))]
else:
    P_U = [(l[2]*np.exp(l[2]*(u+1))) / (np.exp(2*l[2]) -1) if l[2]!=0 else 1/2000 for u in U]
_f12, = ax23.plot(U, P_U, lw=2)


ttc_inv_vals = [round(x,2) for x in np.arange(.25,10.1,.1)]
dist_vals = [round(x,2) for x in np.arange(0,10,.1)]
p_vals = [round(x,1) for x in np.arange(0.1,120,0.1)]


#util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
#util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 

y_ttc_vals = [util_ttc(ttc_sec, 3.5) for ttc_sec in ttc_inv_vals]
y_dist_vals = [util_dist(dist_m, 10) for dist_m in dist_vals]
y_p_vals = [util_progress(x) for x in p_vals]

_f20, = ax11.plot(ttc_inv_vals,y_ttc_vals)
_f21, = ax12.plot(dist_vals,y_dist_vals)
_f22, = ax13.plot(p_vals,y_p_vals)

vel_s = 30


if mixed:
    axcolor = 'lightgoldenrodyellow'
    axutil = plt.axes([0.75, 0.37, 0.08, 0.03], facecolor=axcolor)
    sutil_ttc = Slider(axutil, '$\lambda (ttc)$', 0, 30, valinit=15, valstep=delta_f)
    
    axcolor = 'lightgoldenrodyellow'
    axutil = plt.axes([0.75, 0.335, 0.08, 0.03], facecolor=axcolor)
    sutil_dist = Slider(axutil, '$\lambda$ (dist gap)', 0, 30, valinit=15, valstep=delta_f)
    
    axcolor = 'lightgoldenrodyellow'
    axutil_p = plt.axes([0.75, 0.3, 0.08, 0.03], facecolor=axcolor)
    sutil_p = Slider(axutil_p, '$\lambda$ (progress)', 0, 30, valinit=15, valstep=delta_f)
    
    axcolor = 'lightblue'
    axutil = plt.axes([0.2, 0.1, 0.10, 0.03], facecolor=axcolor)
    sv_s = Slider(axutil, 'vel subject (kph)', 0, 120, valinit=30, valstep=1)
    
    axcolor = 'lightgoldenrodyellow'
    axutil = plt.axes([0.9, 0.37, 0.08, 0.03], facecolor=axcolor)
    sutil_ttc2 = Slider(axutil, '$\lambda$ (ttc)', -30, 0, valinit=-15, valstep=delta_f)
    
    axcolor = 'lightgoldenrodyellow'
    axutil = plt.axes([0.9, 0.335, 0.08, 0.03], facecolor=axcolor)
    sutil_dist2 = Slider(axutil, '$\lambda$ (dist gap)', -30, 0, valinit=15, valstep=delta_f)
    
    axcolor = 'lightgoldenrodyellow'
    axutil_p = plt.axes([0.9, 0.3, 0.08, 0.03], facecolor=axcolor)
    sutil_p2 = Slider(axutil_p, '$\lambda$ (progress)', -30, 0, valinit=-15, valstep=delta_f)
    
    txt_label = get_beh_label(l[0], l[1], l[2])
    t1 = axutil_p.text(-.25,-1,txt_label,transform=axutil_p.transAxes)
    
    axcolor = 'lightgoldenrodyellow'
    axutil = plt.axes([0.3, 0.37, 0.08, 0.03], facecolor=axcolor)
    sutil_ttca = Slider(axutil, 'a (ttc)', 0, 1, valinit=.5, valstep=.1)
    
    axcolor = 'lightgoldenrodyellow'
    axutil = plt.axes([0.3, 0.335, 0.08, 0.03], facecolor=axcolor)
    sutil_dista = Slider(axutil, 'a (dist gap)', 0, 1, valinit=.5, valstep=.1)
    
    axcolor = 'lightgoldenrodyellow'
    axutil_p = plt.axes([0.3, 0.3, 0.08, 0.03], facecolor=axcolor)
    sutil_pa = Slider(axutil_p, 'a (progress)', 0, 1, valinit=.5, valstep=.1)

else:
    axcolor = 'lightgoldenrodyellow'
    axutil = plt.axes([0.8, 0.37, 0.10, 0.03], facecolor=axcolor)
    sutil_ttc = Slider(axutil, '$\lambda$ (ttc)', -30, 30, valinit=-1.6, valstep=delta_f)
    
    axcolor = 'lightgoldenrodyellow'
    axutil = plt.axes([0.8, 0.335, 0.10, 0.03], facecolor=axcolor)
    sutil_dist = Slider(axutil, '$\lambda$ (dist gap)', -30, 30, valinit=6, valstep=delta_f)
    
    axcolor = 'lightgoldenrodyellow'
    axutil_p = plt.axes([0.8, 0.3, 0.10, 0.03], facecolor=axcolor)
    sutil_p = Slider(axutil_p, '$\lambda$ (progress)', -30, 30, valinit=-6, valstep=delta_f)
    
    axcolor = 'lightblue'
    axutil = plt.axes([0.2, 0.1, 0.10, 0.03], facecolor=axcolor)
    sv_s = Slider(axutil, 'vel subject (kph)', 0, 120, valinit=30, valstep=1)
    
    txt_label = get_beh_label(l[0], l[1], l[2])
    t1 = axutil_p.text(-.25,-1,txt_label,transform=axutil_p.transAxes)


l_param = []


def update_prob_lambda_ttc(val):
    global l,l2
    l[0] = val
    if not mixed:
        _f10.set_ydata([(l[0]*np.exp(l[0]*(u+1))) / (np.exp(2*l[0]) -1) if l[0]!=0 else 1/2000 for u in U])
    else:
        P_U_1 = [(l[0]*np.exp(l[0]*(u+1))) / (np.exp(2*l[0]) -1) if l[0]!=0 else 1/2000 for u in U]
        P_U_2 = [(l2[0]*np.exp(l2[0]*(u+1))) / (np.exp(2*l2[0]) -1) if l2[0]!=0 else 1/2000 for u in U]
        P_U = [a[0]+a[1] for a in list(zip(P_U_1,P_U_2))]
        _f10.set_ydata(P_U)
    update_dist_action_prob(vel_s)
    txt_label = get_beh_label(l[0], l[1], l[2])
    t1.set_text(txt_label)
    plt.gcf().canvas.draw_idle()

sutil_ttc.on_changed(update_prob_lambda_ttc)

def update_prob_lambda_dist(val):
    global l,l2
    l[1] = val
    if not mixed:
        _f11.set_ydata([(l[1]*np.exp(l[1]*(u+1))) / (np.exp(2*l[1]) -1) if l[1]!=0 else 1/2000 for u in U])
    else:
        P_U_1 = [(l[1]*np.exp(l[1]*(u+1))) / (np.exp(2*l[1]) -1) if l[1]!=0 else 1/2000 for u in U]
        P_U_2 = [(l2[1]*np.exp(l2[1]*(u+1))) / (np.exp(2*l2[1]) -1) if l2[1]!=0 else 1/2000 for u in U]
        P_U = [a[0]+a[1] for a in list(zip(P_U_1,P_U_2))]
        _f11.set_ydata(P_U)
    update_dist_action_prob(vel_s)
    txt_label = get_beh_label(l[0], l[1], l[2])
    t1.set_text(txt_label)
    plt.gcf().canvas.draw_idle()
sutil_dist.on_changed(update_prob_lambda_dist)

def update_prob_lambda_prog(val):
    global l,l2
    l[2] = val
    if not mixed:
        _f12.set_ydata([(l[2]*np.exp(l[2]*(u+1))) / (np.exp(2*l[2]) -1) if l[2]!=0 else 1/2000 for u in U])
    else:
        P_U_1 = [(l[2]*np.exp(l[2]*(u+1))) / (np.exp(2*l[2]) -1) if l[2]!=0 else 1/2000 for u in U]
        P_U_2 = [(l2[2]*np.exp(l2[2]*(u+1))) / (np.exp(2*l2[2]) -1) if l2[2]!=0 else 1/2000 for u in U]
        P_U = [a[0]+a[1] for a in list(zip(P_U_1,P_U_2))]
        _f12.set_ydata(P_U)
    update_dist_action_prob(vel_s)
    txt_label = get_beh_label(l[0], l[1], l[2])
    t1.set_text(txt_label)
    plt.gcf().canvas.draw_idle()
sutil_p.on_changed(update_prob_lambda_prog)


def update_prob_lambda_ttc2(val):
    global l,l2
    l2[0] = val
    if not mixed:
        _f10.set_ydata([(l[0]*np.exp(l[0]*(u+1))) / (np.exp(2*l[0]) -1) if l[0]!=0 else 1/2000 for u in U])
    else:
        P_U_1 = [(l[0]*np.exp(l[0]*(u+1))) / (np.exp(2*l[0]) -1) if l[0]!=0 else 1/2000 for u in U]
        P_U_2 = [(l2[0]*np.exp(l2[0]*(u+1))) / (np.exp(2*l2[0]) -1) if l2[0]!=0 else 1/2000 for u in U]
        P_U = [a[0]+a[1] for a in list(zip(P_U_1,P_U_2))]
        _f10.set_ydata(P_U)
    update_dist_action_prob(vel_s)
    txt_label = get_beh_label(l[0], l[1], l[2])
    t1.set_text(txt_label)
    plt.gcf().canvas.draw_idle()



def update_prob_lambda_dist2(val):
    global l,l2
    l2[1] = val
    if not mixed:
        _f11.set_ydata([(l[1]*np.exp(l[1]*(u+1))) / (np.exp(2*l[1]) -1) if l[1]!=0 else 1/2000 for u in U])
    else:
        P_U_1 = [(l[1]*np.exp(l[1]*(u+1))) / (np.exp(2*l[1]) -1) if l[1]!=0 else 1/2000 for u in U]
        P_U_2 = [(l2[1]*np.exp(l2[1]*(u+1))) / (np.exp(2*l2[1]) -1) if l2[1]!=0 else 1/2000 for u in U]
        P_U = [a[0]+a[1] for a in list(zip(P_U_1,P_U_2))]
        _f11.set_ydata(P_U)
    update_dist_action_prob(vel_s)
    txt_label = get_beh_label(l[0], l[1], l[2])
    t1.set_text(txt_label)
    plt.gcf().canvas.draw_idle()

def update_prob_lambda_prog2(val):
    global l,l2
    l2[2] = val
    if not mixed:
        _f12.set_ydata([(l[2]*np.exp(l[2]*(u+1))) / (np.exp(2*l[2]) -1) if l[2]!=0 else 1/2000 for u in U])
    else:
        P_U_1 = [(l[2]*np.exp(l[2]*(u+1))) / (np.exp(2*l[2]) -1) if l[2]!=0 else 1/2000 for u in U]
        P_U_2 = [(l2[2]*np.exp(l2[2]*(u+1))) / (np.exp(2*l2[2]) -1) if l2[2]!=0 else 1/2000 for u in U]
        P_U = [a[0]+a[1] for a in list(zip(P_U_1,P_U_2))]
        _f12.set_ydata(P_U)
    update_dist_action_prob(vel_s)
    txt_label = get_beh_label(l[0], l[1], l[2])
    t1.set_text(txt_label)
    plt.gcf().canvas.draw_idle()


def update_alpha_ttc(val):
    global alpha
    alpha[0] = val
    update_dist_action_prob(vel_s)

def update_alpha_d(val):
    global alpha
    alpha[1] = val
    update_dist_action_prob(vel_s)
    
def update_alpha_p(val):
    global alpha
    alpha[2] = val
    update_dist_action_prob(vel_s)

if mixed:
    sutil_p2.on_changed(update_prob_lambda_prog2)
    sutil_ttc2.on_changed(update_prob_lambda_ttc2)
    sutil_dist2.on_changed(update_prob_lambda_dist2)
    
    sutil_pa.on_changed(update_alpha_p)
    sutil_ttca.on_changed(update_alpha_ttc)
    sutil_dista.on_changed(update_alpha_d)



def get_prob(vel_s,vel_lc_l,range_x_l):
    global alpha
    p_a_l = []
    ct = 0
    tot = vel_lc_l.shape[0]*range_x_l.shape[0]
    for vel_lc in np.nditer(vel_lc_l):
        for range_x in np.nditer(range_x_l):
            if vel_lc >= vel_s:
                p_a = 0
            else:
                #ct += 1
                #print(ct/tot,tot)
                #print(vel_s,vel_lc,range_x)
                u_p = util_progress(vel_lc)
                util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
                if vel_s == vel_lc:
                    u_ttc = util_ttc(range_x/(0.001),util_ttc_param)
                else:
                    u_ttc = util_ttc(range_x/(vel_s-vel_lc),util_ttc_param)
                util_dist_param = 10
                u_d = util_dist(range_x,util_dist_param)
                if not mixed:
                    _p_0 = (1/3) * (l[0]*np.exp(l[0]*(u_p+1))) / (np.exp(2*l[0]) -1) if l[0]!=0 else 1/2000
                    _p_1 = (1/3) * (l[1]*np.exp(l[1]*(u_ttc+1))) / (np.exp(2*l[1]) -1) if l[1]!=0 else 1/2000
                    _p_2 = (1/3) * (l[2]*np.exp(l[2]*(u_d+1))) / (np.exp(2*l[2]) -1) if l[2]!=0 else 1/2000
                else:
                    _p_01 = (1/3) * (l[0]*np.exp(l[0]*(u_p+1))) / (np.exp(2*l[0]) -1) if l[0]!=0 else 1/2000
                    _p_11 = (1/3) * (l[1]*np.exp(l[1]*(u_ttc+1))) / (np.exp(2*l[1]) -1) if l[1]!=0 else 1/2000
                    _p_21 = (1/3) * (l[2]*np.exp(l[2]*(u_d+1))) / (np.exp(2*l[2]) -1) if l[2]!=0 else 1/2000
                    
                    _p_02 = (1/3) * (l2[0]*np.exp(l2[0]*(u_p+1))) / (np.exp(2*l2[0]) -1) if l2[0]!=0 else 1/2000
                    _p_12 = (1/3) * (l2[1]*np.exp(l2[1]*(u_ttc+1))) / (np.exp(2*l2[1]) -1) if l2[1]!=0 else 1/2000
                    _p_22 = (1/3) * (l2[2]*np.exp(l2[2]*(u_d+1))) / (np.exp(2*l2[2]) -1) if l2[2]!=0 else 1/2000
                    
                    _p_0 = alpha[0]*_p_01 + (1-alpha[1])*_p_02
                    _p_1 = alpha[1]*_p_01 + (1-alpha[1])*_p_02
                    _p_2 = alpha[2]*_p_01 + (1-alpha[2])*_p_02
                     
                p_a = (_p_0 + _p_1 +_p_2 )/3
                
            #print(p_a,u_p,u_ttc,u_d)
            p_a_l.append(p_a)
    return np.asarray(p_a_l)

def update_dist_action_prob(val):
    global vel_s
    vel_s = val
    util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
    y_ttc_vals = [util_ttc(ttc_sec, util_ttc_param) for ttc_sec in ttc_inv_vals]
    _f20.set_ydata(y_ttc_vals)
    vel_lc_l = np.arange(120)
    range_x_l = np.arange(10)
    z = get_prob(vel_s, vel_lc_l, range_x_l)
    
    _V_LC,_R_X = np.meshgrid(vel_lc_l,range_x_l)
    Z=z.reshape(len(vel_lc_l),len(range_x_l))
    Z = normalize_list_numpy(Z)
    ax32.pcolormesh(_V_LC,_R_X,np.swapaxes(Z,0,1),shading = 'gouraud')
    ax32.set_xlabel('vel lane change (kph)')
    ax32.set_ylabel('dist (m)')

sv_s.on_changed(update_dist_action_prob)


vel_lc_l = np.arange(120)
range_x_l = np.arange(10)
z = get_prob(vel_s, vel_lc_l, range_x_l)

_V_LC,_R_X = np.meshgrid(vel_lc_l,range_x_l)
Z=z.reshape(len(vel_lc_l),len(range_x_l))
Z = normalize_list_numpy(Z)
im = ax32.pcolormesh(_V_LC,_R_X,np.swapaxes(Z,0,1),shading = 'gouraud')
ax32.set_xlabel('vel lane change (kph)')
ax32.set_ylabel('dist (m)')
fig = plt.gcf()
fig.colorbar(im, ax=ax32)
#plt.colorbar()
'''
a = np.random.random((16, 16))
ax32.imshow(a, cmap='hot', interpolation='nearest')
'''
ax31.axis('off')
ax33.axis('off')


def reset(event):
    sutil_ttc.reset()
    sutil_dist.reset()
    sutil_p.reset()

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
button.on_clicked(reset)

#plt.legend(loc='best')
plt.show()








