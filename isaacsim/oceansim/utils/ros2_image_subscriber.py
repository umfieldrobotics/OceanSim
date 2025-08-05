import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            '/oceansim/robot/uw_img',
            self.image_callback,
            qos)
            
        self.get_logger().info('Compressed Image Subscriber Node has been started')

    def image_callback(self, msg):
        try:
            # debug
            self.get_logger().info(f'Received compressed image: format={msg.format}, '
                                 f'data_size={len(msg.data)}')

            # convert
            np_arr = np.frombuffer(msg.data, np.uint8)
            # decode jpeg img
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # return BGR format

            if cv_image is None:
                self.get_logger().error('Failed to decode compressed image')
                return

            # debug
            self.get_logger().info(f'Image shape: {cv_image.shape}, dtype: {cv_image.dtype}, '
                                 f'min: {cv_image.min()}, max: {cv_image.max()}')

            # imshow
            cv2.imshow('Underwater Image', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing compressed image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()