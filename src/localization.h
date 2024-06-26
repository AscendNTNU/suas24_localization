#pragma once
#include <map>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/service.hpp>
#include <rclcpp/timer.hpp>
#include <sensor_msgs/msg/detail/image__struct.hpp>
#include <std_msgs/msg/detail/int32_multi_array__struct.hpp>
#include <string>
#include <suas24_interfaces/msg/detail/visualization_imgs__struct.hpp>
#include <suas24_interfaces/srv/detail/debug__struct.hpp>
#include <tf2_msgs/msg/detail/tf_message__struct.hpp>
#include <vector>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "image_geometry/pinhole_camera_model.h"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "std_msgs/msg/string.hpp"
#include <std_msgs/msg/int32_multi_array.hpp>
//#include "suas23_common/suas23_common.hpp"
//#include "suas23_interfaces/srv/drop_point_info.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include <suas24_interfaces/msg/detail/classification__struct.hpp>
//#include <suas24_interfaces/msg/classification.hpp>
#include <suas24_interfaces/msg/detail/classification__struct.hpp>

#include <suas24_interfaces/srv/detail/drop_point_info__struct.hpp>
#include <suas24_interfaces/msg/visualization_imgs.hpp>
#include <suas24_interfaces/srv/debug.hpp>
//#include "vision_msgs/msg/detection2_d_array.hpp"

// TODO - MAVROS position

struct DetectionLocation {
  float x, y, z;
  float confidence;
};

struct DropPointImage {
  cv::Mat image;
  int min_x, min_y;
};

class DetectionEstimator : public rclcpp::Node {
 public:
  DetectionEstimator();
  

 private:
  // Publishers
  std::string otopic_points;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr
      points_publisher;
    
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr visualization_points_publisher;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr json_data_publisher;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub;
  
  int standard_objects_size;

  rclcpp::Publisher<suas24_interfaces::msg::VisualizationImgs>::SharedPtr visualization_heatmap_publisher;
  rclcpp::TimerBase::SharedPtr timer_;
  void visualization_callback();

  void publish_json_data(tf2_msgs::msg::TFMessage::SharedPtr msg);
  rclcpp::TimerBase::SharedPtr json_timer;

  // Subscribers
  std::string itopic_classifications, itopic_camera_info;
//   rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr
//       detections_subscriber;
  rclcpp::Subscription<suas24_interfaces::msg::Classification>::SharedPtr detections_subscriber;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_subscriber;
  void detections_callback(
      //vision_msgs::msg::Detection2DArray::ConstSharedPtr msg);
      suas24_interfaces::msg::Classification::ConstSharedPtr msg);
  void camera_info_callback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
  image_geometry::PinholeCameraModel camera_model;

  //kok
  //StandardObject parse_standard_object(const std::string &object);

  // Services
  std::string oserv_drop_points;
  rclcpp::Service<suas24_interfaces::srv::DropPointInfo>::SharedPtr
      drop_points_service;
  void drop_points_callback(
      suas24_interfaces::srv::DropPointInfo::Request::SharedPtr req,
      suas24_interfaces::srv::DropPointInfo::Response::SharedPtr resp);

  std::array<geometry_msgs::msg::PointStamped, 5> get_distinct_drop_points();

  // TF2
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener;

  bool m_gimbal_mode = 0;
  float m_gimbal_pitch_offset = 0.0;
  float m_gimbal_roll_offset = 0.0;

  void apply_gimbal_correction(geometry_msgs::msg::TransformStamped& transform_drone_to_ground);

  // cvars
  std::string frame_camera, frame_ground;
  float confidence_threshold = 0.0;
  float spatial_resolution;
  bool m_use_drone_pos;

  bool m_select_distinct_points;

  // Filtering
  cv::Mat kernel;
  static void save_debug_image(cv::Mat image_f32, const std::string& name);
  DropPointImage get_drop_point_image(int object_index);
  cv::Point3f get_drop_point(int object_index);
  std::vector<std::vector<DetectionLocation>> detection_points;

  // Standard objects
  //std::vector<suas23_common::StandardObject> standard_objects;


  //debug shit
  bool m_should_publish_json;
  bool m_should_publish_heatmap;
  bool debug;
  std::string debug_string;
  rclcpp::Service<suas24_interfaces::srv::Debug>::SharedPtr debug_service; 
  void debug_callback(
    suas24_interfaces::srv::Debug::Request::SharedPtr req,
    suas24_interfaces::srv::Debug::Response::SharedPtr res
  );



  //refactor
  sensor_msgs::msg::Image cv_mat_to_ros(const cv::Mat& mat);
};
