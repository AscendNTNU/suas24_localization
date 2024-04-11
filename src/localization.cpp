/*
 * This file is part of the suas24_perception project.
 *
 * Original Author: @MStarvik, @gronnmann
 * Original Repository: https://github.com/AscendNTNU/object_position_estimator.git
 * License:
 *
 * Modifications made by @ulrikisdahl, @NicolaiAdilfor suas24_localization.
 * https://github.com/AscendNTNU/suas24_localization.git
 *
 */


#include "localization.h"

#include <chrono>
#include <cstdint>
#include <geometry_msgs/msg/detail/point_stamped__struct.hpp>
#include <geometry_msgs/msg/detail/transform_stamped__struct.hpp>
#include <sensor_msgs/msg/detail/image__struct.hpp>
#include <std_msgs/msg/detail/string__struct.hpp>
#include <string>
#include <suas24_interfaces/msg/detail/classification__struct.hpp>
#include <suas24_interfaces/msg/detail/visualization_imgs__struct.hpp>
#include <suas24_interfaces/srv/detail/debug__struct.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <array>
#include <optional>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
//#include <suas24_interfaces/msg/detail/classification__struct.hpp>
#include <suas24_interfaces/msg/classification.hpp>
#include <suas24_interfaces/srv/detail/drop_point_info__struct.hpp>
#include <suas24_interfaces/srv/debug.hpp>


int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectionEstimator>());
  rclcpp::shutdown();
  return 0;
}

// TODO - check all int/float conversions
DetectionEstimator::DetectionEstimator()
    : rclcpp::Node("object_position_estimator") {
  tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener = std::make_unique<tf2_ros::TransformListener>(*tf_buffer);

  standard_objects_size = 5;

  detection_points.resize(5); //must correspond to size of 'confidence' in suas24_interfaces::msg::Classification

  this->declare_parameter<std::string>("frame_camera", "camera"); //for tf2 transforms
  this->declare_parameter<std::string>("frame_ground", "local_enu"); //for tf2 transforms
  //populate the fields
  this->get_parameter<std::string>("frame_camera", frame_camera); //for tf2 transforms
  this->get_parameter<std::string>("frame_ground", frame_ground); //for tf2 transforms

  //send drop points of objects
  this->declare_parameter<std::string>("oserv_drop_points", "~/drop_points"); //drop point service name
  this->get_parameter<std::string>("oserv_drop_points", oserv_drop_points);
  drop_points_service = create_service<suas24_interfaces::srv::DropPointInfo>(
      oserv_drop_points,
      std::bind(&DetectionEstimator::drop_points_callback, this,
                std::placeholders::_1, std::placeholders::_2));

  this->declare_parameter<std::string>("otopic_points", "/perception/object_position");
  this->get_parameter<std::string>("otopic_points", otopic_points);
  points_publisher =
      create_publisher<geometry_msgs::msg::PointStamped>(otopic_points, 10); //publisher local-ENU

  this->declare_parameter<std::string>("itopic_classifications", "/classification");
  this->get_parameter<std::string>("itopic_classifications", itopic_classifications);

  this->declare_parameter<float>("spatial_resolution", 0.1);
  this->declare_parameter<float>("confidence_threshold", 0.0);
  this->get_parameter<float>("spatial_resolution", spatial_resolution);
  this->get_parameter<float>("confidence_threshold", confidence_threshold);

  
  this->declare_parameter<std::string>("itopic_camera_info", "/perception/camera_info");
  this->get_parameter<std::string>("itopic_camera_info", itopic_camera_info);
  camera_info_subscriber = create_subscription<sensor_msgs::msg::CameraInfo>(
      itopic_camera_info, 10,
      std::bind(&DetectionEstimator::camera_info_callback, this,
                std::placeholders::_1));

  debug = true;

  visualization_heatmap_publisher = create_publisher<suas24_interfaces::msg::VisualizationImgs>("/viz/heatmap", 10);
  timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&DetectionEstimator::visualization_callback, this));

  // The kernel size is an odd value corresponding to 5 meters on the ground
  const int kernel_size = std::round(5.0 / spatial_resolution);
  kernel = cv::getGaussianKernel(kernel_size + (1 - kernel_size % 2), 0, CV_32F);
}


/*
* Saves the center coordinates of the detection (in local-ENU) whenever a detection is made
*/
void DetectionEstimator::detections_callback(
    suas24_interfaces::msg::Classification::ConstSharedPtr detection_msg) {

  std::optional<geometry_msgs::msg::PointStamped> central_detection;
  float central_detection_deviation;
  RCLCPP_INFO(this->get_logger(), "RCL1"); 

  geometry_msgs::msg::TransformStamped transform_drone_to_ground;
  geometry_msgs::msg::TransformStamped transform_ground_to_drone;

  try {
    transform_drone_to_ground = tf_buffer->lookupTransform(
        frame_ground, frame_camera, detection_msg->header.stamp,
        rclcpp::Duration::from_seconds(1));
    transform_ground_to_drone = tf_buffer->lookupTransform(
        frame_camera, frame_ground, detection_msg->header.stamp,
        rclcpp::Duration::from_seconds(1));
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "Failure %s",
                ex.what());  // Print exception which was caught
    return;
  }

  RCLCPP_INFO(this->get_logger(), "RCL2");
  int x_cam = detection_msg->center_x;
  int y_cam = detection_msg->center_y; 

  double drone_x = transform_drone_to_ground.transform.translation.x;
  double drone_y = transform_drone_to_ground.transform.translation.y;
  double drone_z = transform_drone_to_ground.transform.translation.z;

  //auto cameraVectorCV = camera_model.projectPixelTo3dRay(cv::Point2d(x_cam, y_cam));
  cv::Point3d cameraVectorCV; //replacement for segfaulting error in projectPixelTo3dRay
  cameraVectorCV.x = (x_cam - camera_model.cx() -camera_model.Tx()) / camera_model.fx();
  cameraVectorCV.y = (y_cam - camera_model.cy() -camera_model.Ty()) / camera_model.fy();
  cameraVectorCV.z = 1.0f;

  geometry_msgs::msg::PointStamped cameraVector;
  cameraVector.point.x = drone_z * cameraVectorCV.x;  // Stretch to correct distance
  cameraVector.point.y = drone_z * cameraVectorCV.y;
  cameraVector.point.z = drone_z * cameraVectorCV.z;
  cameraVector.header.stamp = detection_msg->header.stamp;
  cameraVector.header.frame_id = frame_camera;

  RCLCPP_INFO(this->get_logger(), "RCL3");

  geometry_msgs::msg::PointStamped cameraVectorGround;

  // Transfer from camera frame to body frame
  tf2::doTransform(cameraVector, cameraVectorGround,
                    transform_drone_to_ground);

  // Save the current detection in each of the standard object indecies along with the confidence
  RCLCPP_INFO(this->get_logger(), "RCL4");
  
  bool over_threshold = false; 
  for (int i = 0; i < standard_objects_size; i++) {
    const float score = detection_msg->confidence[i].conf_global;
    if (score < confidence_threshold) {
      continue;
    }
    over_threshold = true;

    detection_points.at(i).push_back(
        {static_cast<float>(cameraVectorGround.point.x),
          static_cast<float>(cameraVectorGround.point.y),
          static_cast<float>(cameraVectorGround.point.z),
          static_cast<float>(score)});
  }
  RCLCPP_INFO(this->get_logger(), "RCL5");

  if (over_threshold) {
    const float deviation =
        std::pow(static_cast<float>(x_cam) - camera_model.cx(), 2) +
        std::pow(static_cast<float>(y_cam) - camera_model.cy(), 2);
    if (!central_detection.has_value() ||
        deviation < central_detection_deviation) {
      central_detection = cameraVectorGround;
      central_detection_deviation = deviation;
    }
  }
  RCLCPP_INFO(this->get_logger(), "RCL6");

  RCLCPP_INFO(this->get_logger(), "DROP LOCATION: x: %f, y: %f, z: %f",
              cameraVectorGround.point.x, cameraVectorGround.point.y,
              cameraVectorGround.point.z);

  if (central_detection.has_value()) {
    points_publisher->publish(central_detection.value());
    RCLCPP_INFO(this->get_logger(), "RCL PUBLISHED");
  }
}

void DetectionEstimator::camera_info_callback(
    sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Initializing camera model");
  camera_model.fromCameraInfo(msg);

  camera_info_subscriber.reset();

  detections_subscriber = create_subscription<suas24_interfaces::msg::Classification>(
                          itopic_classifications, 10,
                          std::bind(&DetectionEstimator::detections_callback, this, std::placeholders::_1));
}

DropPointImage DetectionEstimator::get_drop_point_image(const int object_index) {
  if (object_index >= standard_objects_size + 1) {
    RCLCPP_ERROR(this->get_logger(), "Object index out of range");
    return {
        cv::Mat::zeros(1, 1, CV_32F),
        0,
        0,
    };
  }

  const std::string object_id =
      (object_index >= standard_objects_size)
          ? "emergent"
          : "standard_object"; //TODO: get actual specific object name indexed by object_index

  if (detection_points.at(object_index).empty()) {
    RCLCPP_ERROR(this->get_logger(), "No detections for object: %s",
                 object_id.c_str());
    return {
        cv::Mat::zeros(1, 1, CV_32F),
        0,
        0,
    };
  }

  // TODO - better ways to find min max
  float x_min = std::numeric_limits<float>::max();
  float x_max = std::numeric_limits<float>::lowest();
  float y_min = std::numeric_limits<float>::max();
  float y_max = std::numeric_limits<float>::lowest();

  for (auto detection : detection_points.at(object_index)) {
    x_min = std::min(x_min, detection.x);
    x_max = std::max(x_max, detection.x);
    y_min = std::min(y_min, detection.y);
    y_max = std::max(y_max, detection.y);
  }

  int width = std::abs(std::ceil((x_max - x_min) / spatial_resolution));
  int height = std::abs(std::ceil((y_max - y_min) / spatial_resolution));
  RCLCPP_INFO(this->get_logger(), "HEIGHT %i", height);

  width = std::max(width, 1);
  height = std::max(height, 1);

  cv::Mat img = cv::Mat::zeros(height, width, CV_32F);

  // RCLCPP_INFO(this->get_logger(), "Initializing matrix for %s: %i x %i",
  //             object_id.c_str(), width, height);

  for (const auto& detection : detection_points.at(object_index)) {
    const int x_img = (detection.x - x_min) / spatial_resolution;
    const int y_img = (detection.y - y_min) / spatial_resolution;

    img.at<float>(y_img, x_img) += 255;
  }

  cv::Mat img_filtered(height, width, CV_32F);
  cv::sepFilter2D(img, img_filtered, -1, kernel, kernel);

  if (debug) {
    save_debug_image(img_filtered, "droppoint_" + object_id + ".png");
  }

  return {img_filtered, static_cast<int>(x_min), static_cast<int>(y_min)};
}

cv::Point3f DetectionEstimator::get_drop_point(const int object_index) {
  if (object_index >= standard_objects_size + 1) {
    RCLCPP_ERROR(this->get_logger(), "Object index out of range");
    return {0, 0, 0};
  }

  DropPointImage dropPointImage = get_drop_point_image(object_index);
  cv::Mat img = dropPointImage.image;

  // USe cv minMaxLoc
  cv::Point2i min_loc;
  cv::Point2i max_loc;

  double min_val;
  double max_val;

  cv::minMaxLoc(img, &min_val, &max_val, &min_loc, &max_loc);

  // In meters relative to local_enu
  const float x =
      static_cast<float>(max_loc.x) * spatial_resolution + dropPointImage.min_x;
  const float y =
      static_cast<float>(max_loc.y) * spatial_resolution + dropPointImage.min_y;

  return {x, y, static_cast<float>(max_val)};
}

void DetectionEstimator::drop_points_callback(
    suas24_interfaces::srv::DropPointInfo::Request::SharedPtr req,
    suas24_interfaces::srv::DropPointInfo::Response::SharedPtr resp) {
  bool success = false;

  std::array<geometry_msgs::msg::Point, 5> drop_points;
  std::array<int, 5> confidences;

  for (int i = 0; i < standard_objects_size + 1; i++) {
    if (!success && !detection_points.at(i).empty()) {
      success = true;
    }

    const auto dropPoint = get_drop_point(i);

    drop_points[i].x = dropPoint.x;
    drop_points[i].y = dropPoint.y;
    drop_points[i].z = 0;

    confidences[i] = dropPoint.z;

    const std::string object_id = (i >= standard_objects_size)
                                      ? "emergent"
                                      : "standard object"; //TODO get the correct object name
    RCLCPP_INFO(this->get_logger(), "Drop point for %s: %f, %f  (conf: %f)",
                object_id.c_str(), dropPoint.x, dropPoint.y, dropPoint.z);
  }

  resp->drop_points = drop_points;
  resp->success = success;
}

void DetectionEstimator::save_debug_image(cv::Mat image_f32,
                                          const std::string& name) {
  double min_val, max_val;
  cv::minMaxLoc(image_f32, &min_val, &max_val);

  cv::Mat image_u8;
  image_f32.convertTo(image_u8, CV_8U, 255.0 / (max_val - min_val),
                      -min_val * 255.0 / (max_val - min_val));

  cv::imwrite(name, image_u8);
}


/*
* Publishes the heatmap for debugging purposes
*/
void DetectionEstimator::visualization_callback(){
  if(!detection_points.at(0).empty()){
    suas24_interfaces::msg::VisualizationImgs viz_msg;
    std::vector<sensor_msgs::msg::Image> imgs;

    //append each objects heatmap image to the message 
    for (int i = 0; i < 5; i++) { 
      DropPointImage img_struct = get_drop_point_image(i);
      cv::Mat img = img_struct.image;

      sensor_msgs::msg::Image sensor_img = cv_mat_to_ros(img);
      imgs.push_back(sensor_img); 
    }

    viz_msg.images = imgs;
    visualization_heatmap_publisher->publish(viz_msg);
    RCLCPP_INFO(this->get_logger(), "Published viz");
  }
}




//Refactor:
sensor_msgs::msg::Image
DetectionEstimator::cv_mat_to_ros(const cv::Mat& mat_f32) {
    cv::Mat mat;
    
    double min_val, max_val;
    cv::minMaxLoc(mat_f32, &min_val, &max_val);
    mat_f32.convertTo(mat, CV_8U, 255.0 / (max_val - min_val),
                      -min_val * 255.0 / (max_val - min_val));
    
    sensor_msgs::msg::Image ros_img;

    ros_img.width = mat.cols;
    ros_img.height = mat.rows;

    ros_img.encoding = "8UC1";

    ros_img.is_bigendian = false;

    ros_img.header = std_msgs::msg::Header();

    ros_img.step = ros_img.width * mat.elemSize();

    size_t size = ros_img.step * ros_img.height;
    RCLCPP_INFO(this->get_logger(), "Matrix size: %zu", size);
    ros_img.data.resize(size);

    uchar *ros_data_ptr = reinterpret_cast<uchar *>(&ros_img.data[0]);
    uchar *cv_data_ptr = mat.data;

    memcpy(ros_data_ptr, cv_data_ptr, size);

    return ros_img;
}


