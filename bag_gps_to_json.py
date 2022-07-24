import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import pathlib
import numpy as np
import cv2
import argparse


class BagFileParser:
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of: type_of for id_of, name_of, type_of in topics_data}
        self.topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}
        self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of, type_of in topics_data}

    def __del__(self):
        self.conn.close()

    def get_sizes(self, topic_name):
        topic_id = self.topic_id[topic_name]
        size = self.cursor.execute("SELECT COUNT(data) FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()[
            0
        ][0]
        return size

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):

        # print(self.topic_id.keys())
        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)
        ).fetchall()

        # Deserialise all and timestamp them
        return [(timestamp, deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp, data in rows]

    def get_n_messages(self, topic_name, limit=10, offset=10):

        # print(self.topic_id.keys())
        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = {} LIMIT {} OFFSET {}".format(
                topic_id, limit, offset
            )
        ).fetchall()

        # Deserialise all and timestamp them
        return [(timestamp, deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp, data in rows]


def main(args=None):

    # read args
    parser = argparse.ArgumentParser(description="Bag file parser")
    parser.add_argument("--bag_file", type=str, help="Path to bag file", required=True)
    parser.add_argument("--result_folder", type=str, help="Result_folder", required=True)
    parser.add_argument("--topic_image", type=str, help="Topic_image", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch_size", required=False, default=10)
    parser.add_argument("--topic_gps", type=str, help="Topic_gps", required=True)
    args = parser.parse_args()

    bag_file = (
        args.bag_file
    )  # '/home/tda/Desktop/ROS/ROS_bag/alex/bombers1.bag/bombers1.bag.db3' # '/home/tda/Desktop/ROS/ROS_bag/ubx_files/alos7/alos7.db3' #

    # create folder if it doesn't exist
    pathlib.Path(args.result_folder).mkdir(parents=True, exist_ok=True)

    parser = BagFileParser(bag_file)
    # # print(parser.topic_id)
    # # trajectory = parser.get_messages("gpsrtk/position")
    if(args.topic_gps is not None):
        trajectory = parser.get_messages(args.topic_gps)
        gps_positions = {}
        for i in range(len(trajectory)):
            gps_positions[str(i)] = {
                "lat": trajectory[i][1].lat,
                "lon": trajectory[i][1].lon,
                "height": trajectory[i][1].height,
                "theta": trajectory[i][1].heading,
                "timestamp_sec": trajectory[i][1].header.stamp.sec,
                "timestamp_nano": trajectory[i][1].header.stamp.nanosec
                # "precision" : trajectory[i][1].fix_type
            }
        # print(gps_positions)
        pathlib.Path(f"{args.result_folder}/gps_positions.json").write_text(str(gps_positions).replace("'", '"'))

    pathlib.Path(f"{args.result_folder}/img").mkdir(parents=True, exist_ok=True)
    size = parser.get_sizes(args.topic_image)
    img_timestamps = {}

    for i in range(0, size, args.batch_size):
        messages = parser.get_n_messages(args.topic_image, args.batch_size, i)

        # trajectory = parser.get_messages(args.topic_image)
        

        for j in range(len(messages)):
            id = i + j
            img_timestamps[str(id)] = {
                "timestamp_sec": trajectory[j][1].header.stamp.sec,
                "timestamp_nano": trajectory[j][1].header.stamp.nanosec
                # "precision" : trajectory[i][1].fix_type
            }

            decoded_img = cv2.imdecode(np.frombuffer(np.asarray(messages[j][1].data), dtype=np.uint8), 1)

            if decoded_img is not None:
                decoded_img = cv2.imdecode(np.frombuffer(np.asarray(messages[j][1].data), dtype=np.uint8), 1)[
                    ..., ::-1
                ]
                cv2.imwrite("%s/img/%04d.png" % (args.result_folder, id), cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB))
            else:
                decoded_img = None
                print("None image")
        # print(gps_positions)
    pathlib.Path(f"{args.result_folder}/img_timestamps.json").write_text(str(img_timestamps).replace("'", '"'))


if __name__ == "__main__":
    main()
