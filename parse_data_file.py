'''
Created on Jul 2, 2018

@author: Atrisha
'''
import sqlite3
import xml.etree.ElementTree as ET
from utils import px_2_utm, get_lane
import itertools
import math
import random
import matplotlib.pyplot as plt

fps = 30
def parse_for_tracks():
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    tree = ET.parse('tracks.xml')
    root = tree.getroot()
    for track in root.findall("./log/successfull_tracking_list/tracking_history"):
        track_id = track.get('map_key')
        vehicle_type = track.get('type')
        print(track_id)
        for track_child in list(track):
            if track_child.tag == 'trajectory':
                trajectory_node = track_child
                start_frame_id = int(trajectory_node.get('first_tracking_image_id'))
                for trajectory_node_child in list(trajectory_node):
                    if trajectory_node_child.tag == 'positions':
                        positions_node = trajectory_node_child
                        all_points = positions_node.text.split(' ')
                        x_points,y_points = [],[]
                        for indx,val in enumerate(all_points):
                            if indx % 2 == 0:
                                x_points.append(val)
                            else:
                                y_points.append(str(float(val)*-1))
                        x_points_utm,y_points_utm = [],[]
                        for x,y in zip(x_points,y_points):
                            x_prime,y_prime = px_2_utm((float(x),float(y)))
                            x_points_utm.append(x_prime)
                            y_points_utm.append(y_prime)
                        frame_id = start_frame_id  
                        x_points,y_points,x_points_utm,y_points_utm = list(filter(bool, x_points)),list(filter(bool, y_points)),list(filter(bool, x_points_utm)),list(filter(bool, y_points_utm))
                        print('size of the track is',len(x_points))
                        print('last 3 elems',x_points[-1],x_points[-2],x_points[-3])
                        for i in range(len(x_points)):
                            print(i)
                            lane_id = get_lane((x_points[i],y_points[i]))
                            if i < (len(x_points) - 1): 
                                vel_mps = math.sqrt((x_points_utm[i+1] - x_points_utm[i])**2 + (y_points_utm[i+1] - y_points_utm[i])**2) * fps
                            print("INSERT INTO tracks VALUES ('"+str(vehicle_type)+"',"+str(track_id)+","+str(x_points[i])+","+str(x_points_utm[i])+","+str(frame_id)+","+str(y_points[i])+","+str(y_points_utm[i])+","+str(lane_id)+","+str(vel_mps)+")")
                            c.execute("INSERT INTO tracks VALUES ('"+str(vehicle_type)+"',"+str(track_id)+","+str(x_points[i])+","+str(x_points_utm[i])+","+str(frame_id)+","+str(y_points[i])+","+str(y_points_utm[i])+","+str(lane_id)+","+str(vel_mps)+")")
                            frame_id = frame_id + 1
                                  
    conn.commit()
    conn.close()    

def parse_for_lanes():
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    tree = ET.parse('tracks.xml')
    root = tree.getroot()

    test_lane = []
    for lanes in root.findall("./markups/lanes"):
        for lane in list(lanes):
            lane_id = lane.get('id')
            x_points,y_points = [],[]
            for lane_child in list(lane):
                for shape_child in list(lane_child):
                    if str(shape_child.tag).startswith('point'):
                        x_points.append(float(shape_child.get('x')))
                        y_points.append(float(shape_child.get('y'))*-1)
            x_points.append(x_points[0])
            y_points.append(y_points[0])
            x_y_points_tuples_utm = list(map(px_2_utm, zip(x_points,y_points)))
            x_y_points_list_utm = [list(t) for t in zip(*x_y_points_tuples_utm)]
            x_points_utm = x_y_points_list_utm[0]
            y_points_utm = x_y_points_list_utm[1]
            for i in range(len(x_points)):
                print(i)
                print("INSERT INTO lanes VALUES ('"+str(lane_id)+"',"+str(i)+","+str(x_points[i])+","+str(y_points[i])+","+str(x_points_utm[i])+","+str(y_points_utm[i])+")")
                #c.execute("INSERT INTO lanes VALUES ('"+str(lane_id)+"',"+str(i)+","+str(x_points[i])+","+str(y_points[i])+","+str(x_points_utm[i])+","+str(y_points_utm[i])+")")
            #text_indx = random.randint(0,len(x_points)-1)
            #rgb = [random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]
            #plt.text(x_points[text_indx], y_points[text_indx],lane_id,color=rgb)
            #plt.plot(x_points,y_points,color=rgb)
    #plt.show()        
    conn.commit()
    conn.close()  
parse_for_lanes()
    
def parse_for_paths():
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    fname = 'paths'

    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    path_dict = dict()
    content = [x.strip() for x in content] 
    current_key = None
    for line in content:
        if line.endswith(':'):
            current_key = line[:-1]
            path_dict[current_key] = []
        elif line.endswith(')'):
            path_dict[current_key].append(line)
    for key,value in path_dict.items():
        lane_id=key
        for pts_px in value:
            pts_px = tuple(pts_px[1:-1].split(','))
            pts_utm = px_2_utm(pts_px)
            print("INSERT INTO paths VALUES ('"+str(lane_id)+"',"+str(pts_px[0])+","+str(pts_px[1])+","+str(pts_utm[0])+","+str(pts_utm[1])+")")
            c.execute("INSERT INTO paths VALUES ('"+str(lane_id)+"',"+str(pts_px[0])+","+str(pts_px[1])+","+str(pts_utm[0])+","+str(pts_utm[1])+")")
    conn.commit()
    conn.close()          
    
def parse_for_collision_areas():
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    fname = 'collision_points'
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    current_key = 0
    for line in content:
        if ':' in line:
            current_key = current_key + 1
            lane_sets = line.split(':')
            lane_set_a = lane_sets[0].split(',')
            lane_set_b = lane_sets[1].split(',')
            lane_conflicts = list(itertools.product(lane_set_a,lane_set_b))
            for lane_conflict in lane_conflicts:
                print("INSERT INTO lane_collision_map VALUES ('"+str(current_key)+"',"+str(lane_conflict[0])+","+str(lane_conflict[1])+")")
                c.execute("INSERT INTO lane_collision_map VALUES ('"+str(current_key)+"',"+str(lane_conflict[0])+","+str(lane_conflict[1])+")")
        elif '(' in line:
            pts_px = tuple(line[1:-1].split(','))
            print("INSERT INTO collision_polygons VALUES ('"+str(current_key)+"',"+str(pts_px[0])+","+str(pts_px[1])+")")
            c.execute("INSERT INTO collision_polygons VALUES ('"+str(current_key)+"',"+str(pts_px[0])+","+str(pts_px[1])+")")
    conn.commit()
    conn.close()        
    
    
    
    
    