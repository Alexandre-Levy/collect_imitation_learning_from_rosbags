from tkinter import W
import pymap3d as pm
import json
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pathlib
from waypoint_parameterization import ParametricWaypoint
import time
import math
import argparse


def get_waypoint_by_linear_interpolation(
    observed_timestamp: np.ndarray, 
    observed_x:np.ndarray, 
    observed_y:np.ndarray, 
    observed_theta:np.ndarray,
    target_timestamp:np.ndarray, 
    delta_time=500
    ):
    
    """
    observed: known Information
    observed timesatmp, x and y are same shape
    """
    target_waypoint_list = []
    target_theta = []
    num_interpolation = len(observed_timestamp) - 1 
    # 
    for i in range(num_interpolation):
        # create latent timestamp
        n_split = int((max(observed_timestamp[i:i+2]) - min(observed_timestamp[i:i+2])) / delta_time)
        latent_timestamp = np.linspace(min(observed_timestamp[i:i+2]), max(observed_timestamp[i:i+2]), n_split)
        
        # when x is ascending, latent is ascending too.
        if observed_x[i] < observed_x[i+1]:
            latent_x = np.linspace(min(observed_x[i:i+2]), max(observed_x[i:i+2]), n_split)
        else:
            latent_x = np.linspace(min(observed_x[i:i+2]), max(observed_x[i:i+2]), n_split)[::-1]
        
        # fitting
        fitting_func = interpolate.interp1d(observed_x[i:i+2], observed_y[i:i+2])

        target_x = []
        target_y = []

        # only 
        if i == num_interpolation-1:
            target_idx = min(observed_timestamp[i:i+2]) <= target_timestamp
        elif i == 0:
            target_idx = target_timestamp < max(observed_timestamp[i:i+2])
        else:
            target_idx = (min(observed_timestamp[i:i+2]) <= target_timestamp) & (target_timestamp < max(observed_timestamp[i:i+2]))
        target_use_timestamp = target_timestamp[target_idx]

        # get the target waypoint with the closest timestamp
        for t in target_use_timestamp:
            idx = np.abs(latent_timestamp - t).argmin()
            target_x.append(latent_x[idx])
            target_theta.append(observed_theta[i+1])
            # although I don't know, there are nan sometimes.
            if np.isnan(fitting_func(latent_x[idx])).sum() > 0:
                idx = np.abs(observed_timestamp - t).argmin()
                target_y.append(observed_y[idx])
            else:
                target_y.append(fitting_func(latent_x[idx]))

        assert len(target_x) == len(target_y)
        target_waypoint = np.stack([target_x, target_y], axis=1)
        target_waypoint_list.append(target_waypoint)

    target_waypoint = np.concatenate(target_waypoint_list)
    target_theta = np.array(target_theta)
    return target_waypoint,target_theta

def rotate_waypoint(x, y, theta):
    """
    rotate waypoint by theta
    """
    theta = np.deg2rad(theta)
    x_rotated = x * np.cos(theta) - y * np.sin(theta)
    y_rotated = x * np.sin(theta) + y * np.cos(theta)
    
    return x_rotated,y_rotated

def angle_between_vector_x_axis(vector):
    """
    get angle between vector and x axis
    """
    theta = np.arctan2(vector[1], vector[0])
    return theta

def vector_from_2_points(x1, y1, x2, y2):
    """
    get vector from 2 points
    """
    x = x2 - x1
    y = y2 - y1
    return x, y
def main(args=None):

    parser = argparse.ArgumentParser(description='Bag file parser')
    parser.add_argument('--gps_positions', type=str, help='Path to gps positions json file', required=True)
    parser.add_argument('--timestamps_img', type=str, help='Path to timestamps images json file', required=True)
    parser.add_argument('--imgs_positions_folder', type=str, help='Result_folder', required=True)
    parser.add_argument('--lat_ref', type=float, help='latitude_reference', required=False, default=42.713399)
    parser.add_argument('--lon_ref', type=float, help='longitude_reference', required=False, default=1.115376)
    args = parser.parse_args()
    f = open(args.gps_positions)#'gps_positions_uab.json')
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    # Iterating through the json)
    x = []
    y = []
    z = []
    theta = []
    timestamp_gps = []
    lat, lon = args.lat_ref, args.lon_ref #42.713399, 1.115376 # 41.498629, 2.113961 #
    lat+= 1.5555981910608807e-05
    lon+= 1.1431145168216972e-06
    for i in data:
        carla_x, carla_y, carla_z = pm.geodetic2enu(data[i]['lat'], data[i]['lon'], data['0']['height'],lat, lon, data['0']['height'], ell=pm.utils.Ellipsoid('wgs72'))
        x.append(carla_x)
        y.append(-carla_y)
        z.append(carla_z)
        theta.append(data[i]['theta'])
        timestamp_gps.append(data[i]['timestamp_sec'] + data[i]['timestamp_nano']*10**(-9))
        

    f_img = open(args.timestamps_img)#'img_positions_uab.json')
    
    # returns JSON object as 
    # a dictionary
    timestamp_img = []
    data = json.load(f_img)
    for i in data:
        timestamp_img.append(data[i]['timestamp_sec'] + data[i]['timestamp_nano']*10**(-9))

    waypoints, theta_n = get_waypoint_by_linear_interpolation(
        np.array(timestamp_gps),
        np.array(x), 
        np.array(y), 
        np.array(theta),
        np.array(timestamp_img),
        delta_time=0.00001
        )

    nb_wp = len(waypoints)
    wp_dist = 0.0
    for j in range(nb_wp-1):
                wp_dist += np.linalg.norm(waypoints[j+1]-waypoints[j])
                # print(np.linalg.norm(waypoints[j+1]-waypoints[j]))
    print("dist_mean :", wp_dist/nb_wp)
    img_positions = {}
    for i in range(len(waypoints)) :
        img_positions[str(i)] = {
            "x" : waypoints[i][0],
            "y" : waypoints[i][1],
            "theta" : theta_n[i]
            # "precision" : trajectory[i][1].fix_type

        }
    pathlib.Path(args.imgs_positions_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{args.imgs_positions_folder}/measurements.json').write_text(str(img_positions).replace("'",'"'))


    w_corr = ParametricWaypoint()
    nb_pt = 3
    nb_pt_passed = 5
    nb_pt_passed_target = nb_pt + 1
    observed_timestamp = np.array(timestamp_gps)
    target_timestamp = np.array(timestamp_img)
    num_interpolation = len(observed_timestamp) - nb_pt_passed 
    x_img_positions = []
    y_img_positions = []
    theta_img_positions = []
    print(len(observed_timestamp))
    print("num_inte:",num_interpolation)
    img_positions = {}
    for i in range(0, num_interpolation, nb_pt):  #num_interpolation
        # print(i)
        x_off = x[i]
        y_off = y[i]
        theta_off = theta[i + nb_pt]
        x_rep = x[i:i + nb_pt_passed] - x_off
        y_rep = y[i:i + nb_pt_passed] - y_off
        x_new = []
        y_new = []
        target_idx = (min(observed_timestamp[i:i + nb_pt_passed_target]) <= target_timestamp) & (target_timestamp < max(observed_timestamp[i:i + nb_pt_passed_target]))
        target_use_timestamp = target_timestamp[target_idx]
        # print(x_rep)
        theta_new = angle_between_vector_x_axis(vector_from_2_points(x_rep[0], y_rep[0], x_rep[-1], y_rep[-1]))
        for j in range(len(x_rep)):
            x_rotated,y_rotated = rotate_waypoint(x_rep[j],y_rep[j],theta_new)
            x_new.append(x_rotated)
            y_new.append(y_rotated)

        dt = observed_timestamp[i + nb_pt_passed_target] - observed_timestamp[i] 
        idx = []
        
        timestamp_patch = observed_timestamp[i]
        for timestamp in target_use_timestamp:
            dt_tp = np.abs(timestamp - timestamp_patch)
            timestamp_patch += dt_tp
            idx.append(dt_tp)
        idx = np.array(idx)
        data_new = np.vstack([x_new, y_new]).T
        # print(data_new)
        t0 = time.time()
        pts_hist = w_corr.run(data_new, len(target_use_timestamp),idx, dt)
        if not isinstance(pts_hist,np.ndarray):
            # print("pts_hist is not numpy array")
            # continue
            data_new = np.vstack([y_new, - np.array(x_new)]).T
            theta_new-= 90
            t0 = time.time()
            pts_hist = w_corr.run(data_new, len(target_use_timestamp),idx, dt)
            if not isinstance(pts_hist,np.ndarray):
                # print("pts_hist is not numpy array")
                data_new = np.vstack([-np.array(y_new), np.array(x_new)]).T
                theta_new+= 180
                t0 = time.time()
                pts_hist = w_corr.run(data_new, len(target_use_timestamp),idx,dt)
                if not isinstance(pts_hist,np.ndarray):
                    # print("pts_hist is not numpy array")
                    theta_new+= 90
                    data_new = np.vstack([-np.array(x_new), -np.array(y_new)]).T
                    t0 = time.time()
                    pts_hist = w_corr.run(data_new, len(target_use_timestamp),idx,dt)

        if isinstance(pts_hist,np.ndarray):
            if pts_hist.any():
                x_new = []
                y_new = []
                # theta_new = []
                for j in range(len(pts_hist[:,0])):
                    x_rotated,y_rotated = rotate_waypoint(pts_hist[j,0],pts_hist[j,1], -theta_new)
                    # print("angle :",pts_hist[j,2])
                    
                    x_img_positions.append(x_rotated + x_off)
                    y_img_positions.append(y_rotated + y_off)
                    theta_img_positions.append(math.radians(pts_hist[j,2]) + theta_off) #(theta_off) #
                    x_new.append(x_rotated)
                    y_new.append(y_rotated)
                    # theta_new.append(theta_off + math.radians(pts_hist[j,2]))
                    

                pts_hist[:,0] = x_new + x_off 
                pts_hist[:,1] = y_new + y_off
            

                plt.plot(pts_hist[:,0], pts_hist[:,1],'x', markersize=4, color ="blue")
            else:
                for i in range(len(target_use_timestamp)):
                    x_img_positions.append(0)
                    y_img_positions.append(0)
                    theta_img_positions.append(0)
                print("Cannot infere :",i)
        else:
            for i in range(len(target_use_timestamp)):
                x_img_positions.append(0)
                y_img_positions.append(0)
                theta_img_positions.append(0)
            print("Cannot infere :",i)
    for i in range(len(x_img_positions)) :
        img_positions[str(i)] = {
            "x" : x_img_positions[i],
            "y" : y_img_positions[i],
            "theta" : theta_img_positions[i]
            # "precision" : trajectory[i][1].fix_type

        }
    pathlib.Path(f'{args.imgs_positions_folder}/measurements_quadratic.json').write_text(str(img_positions).replace("'",'"'))
    # plt.plot(x_img_positions, y_img_positions,'x', markersize=4, color ="blue")
    plt.plot(x, y, 'o', markersize=4, color ="green")
    plt.plot(waypoints[:,0], waypoints[:,1],'o', markersize=3, color ="red")
    print("waypoints theta quadratic :",theta_img_positions[20:40])
    print("waypoints theta linear :",theta_n[20:40])
    plt.show()
    # plt.savefig(f'{args.imgs_positions_folder}/interpolation.png')

if __name__ == "__main__":
    main()