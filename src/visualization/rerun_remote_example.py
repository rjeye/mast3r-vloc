"""
🚀 Ensure you're running the Rerun Viewer and the Rerun TCP server before running this script.
This can be done by running the following command in a separate terminal on your remote machine:
        rerun --serve
💻 Set up port forwarding by running the following command on your local machine:
        ssh -L 9876:localhost:9876 -L 9877:localhost:9877 -L 9090:localhost:9090 -N user@remote_server
🌐 Once the ports are forwarded, open the following URL in your browser to access the web viewer:
        http://localhost:9090?url=ws://localhost:9877
"""


import time
import argparse
import rerun as rr
import numpy as np

def main():
    # Initialize Rerun and connect to the running Rerun TCP server
    rr.init("rerun_example_image_logging", spawn=False)
    rr.connect_tcp()  # Connect to the TCP server
    
    print(__doc__)
    
    # Create a red image (100x100 pixels)
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    image1[:, :] = [255, 0, 0]  # Red color (R, G, B)

    # Create a green image (100x100 pixels)
    image2 = np.zeros((100, 100, 3), dtype=np.uint8)
    image2[:, :] = [0, 255, 0]  # Green color (R, G, B)

    # Log the images to the Rerun Viewer
    rr.log("image3", rr.Image(image1))
    rr.log("image2", rr.Image(image2))

    print("Images logged. Keep the script running to maintain the connection.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Disconnecting...")
        rr.disconnect()  # Disconnect gracefully on script termination

if __name__ == "__main__":
    main()