'''
Created on Apr 8, 2019

@author: atrisha
'''



def detect_intersections(traj,inters,l,h):
    print(' ',l,h,inters)
    m = l + ((h-l) // 2)
    if (h==l+1) and traj[l][1] is not traj[h][1]:
        inters.append(traj[h])
        return inters
    if traj[m][1] == traj[l][1]:
        return detect_intersections(traj,inters,m,h)
    elif traj[m][1] == traj[h][1]:
        return detect_intersections(traj,inters,l,m)
    else: 
        _s1 = detect_intersections(traj,inters,l,m)
        _s2 = detect_intersections(traj,inters,m,h)
        return inters

def reg_ex_example(i_string='M 21'):
    import re
    x = re.search('[M|I]\s[0-9]+',i_string)
    if x is None:
        print(False)
    else:
        print(True)
