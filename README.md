# collect_imitation_learning_from_rosbags


After setting environment with requirements.txt and setting up ros, run this line to extract data from the rosbags and do the interpolation of vehicle location corresponding to each image:
```
python bag_to_dataset.py  --bag_file /home/tda/Desktop/ROS/ROS_bag/alex/bombers1.bag/bombers1.bag.db3 --dataset_folder /home/tda/Desktop/ROS/ROS_bag/test_data7 --topic_image /frontcamaraUndistorted/CompressedImage --topic_gps gpsrtk/position
```

-- bag file is the ros bag that you want to process
-- dataset_folder is where you want to store it
--topic_image and --topic_gps are the topics for images and gps

All the images will be stored in the folder /img, the gps positions collected in gps_positions.json , the images timestamps collected in img_timestamps.json then we will infere imges positions in measurements_quadratic.json (quadratic interpollation) and measurements.json(linear interpollation)
