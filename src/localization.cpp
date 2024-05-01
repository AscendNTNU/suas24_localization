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
#include "fstream"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <geometry_msgs/msg/detail/point_stamped__struct.hpp>
#include <geometry_msgs/msg/detail/transform_stamped__struct.hpp>
#include <rclcpp/time.hpp>
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
#include <tf2_msgs/msg/detail/tf_message__struct.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
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
  this->declare_parameter<std::string>("oserv_drop_points", "perception/drop_points"); //drop point service name
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

  this->declare_parameter<bool>("debug_json_data", false);
  this->get_parameter<bool>("debug_json_data", m_should_publish_json);

  this->declare_parameter<bool>("debug_viz_heatmap", false);
  this->get_parameter<bool>("debug_viz_heatmap", m_should_publish_heatmap);

  this->declare_parameter<bool>("gimbal_mode", true);
  this->get_parameter<bool>("gimbal_mode", m_gimbal_mode);

  this->declare_parameter<float>("gimbal_pitch_offset", 0.0);
  this->get_parameter<float>("gimbal_pitch_offset", m_gimbal_pitch_offset);

  this->declare_parameter<float>("gimbal_roll_offset", 0.0);
  this->get_parameter<float>("gimbal_roll_offset", m_gimbal_roll_offset);

  if (m_gimbal_mode) {
    RCLCPP_INFO(this->get_logger(), "Configured with gimbal mode");
    RCLCPP_INFO(this->get_logger(), 
      "Using gimbal offsets (roll, pitch) = (%f, %f)", 
      m_gimbal_roll_offset, m_gimbal_pitch_offset
    );
  }

  debug = true;

  if (m_should_publish_heatmap) {
      visualization_heatmap_publisher = create_publisher<suas24_interfaces::msg::VisualizationImgs>("/viz/heatmap", 10);
      timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&DetectionEstimator::visualization_callback, this));
  }

  if (m_should_publish_json) {
    json_data_publisher = create_publisher<std_msgs::msg::String>("/viz/flythrough", 10);
    tf_sub = create_subscription<tf2_msgs::msg::TFMessage>("/tf", 10, 
      std::bind(&DetectionEstimator::publish_json_data, this, std::placeholders::_1));
  }

  // The kernel size is an odd value corresponding to 5 meters on the ground
  const int kernel_size = std::round(5.0 / spatial_resolution);
  kernel = cv::getGaussianKernel(kernel_size + (1 - kernel_size % 2), 0, CV_32F);
}

cv::Point2d rectifyPoint(const cv::Point2d& uv_raw, const cv::Matx33d& K, const cv::Mat_<double>& D, const cv::Matx33d& R, const cv::Matx34d& P) {
    cv::Point2f raw32 = uv_raw, rect32;
   const cv::Mat src_pt(1, 1, CV_32FC2, &raw32.x);
   cv::Mat dst_pt(1, 1, CV_32FC2, &rect32.x);
   cv::undistortPoints(src_pt, dst_pt, K, D, R, P);
   return rect32;
}

std::vector<std::array<float, 4>> detections_buffer;

constexpr float GIMBAL_ROLL = -0.05235988;
constexpr float GIMBAL_PITCH = -0.087 / 2;

void 
DetectionEstimator::apply_gimbal_correction(
    geometry_msgs::msg::TransformStamped& transform_drone_to_ground) 
{
  // This is code for ignoring pitch and roll
  // if we assume the gimbal makes the camera point perfectly downward
  // except for an offset given by gimbal_pitch_offset and/or gimbal_roll_offset

  // NB: Will make tf_static redundant, so it has to be set here in newQuat basically

  tf2::Quaternion quat;
  tf2::fromMsg(transform_drone_to_ground.transform.rotation, quat);
  double roll, pitch, yaw;
  tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);

  // Create a new quaternion with roll and pitch set to zero, only preserving yaw
  tf2::Quaternion newQuat;
  newQuat.setRPY(m_gimbal_roll_offset, M_PI + m_gimbal_pitch_offset, yaw);

  transform_drone_to_ground.transform.rotation = tf2::toMsg(newQuat);
}

void DetectionEstimator::publish_json_data(tf2_msgs::msg::TFMessage::SharedPtr msg) {
  if (!camera_model.initialized())return;
  geometry_msgs::msg::TransformStamped transform_drone_to_ground;
  auto transform_time = msg->transforms.at(0).header.stamp;
  try {
    transform_drone_to_ground = tf_buffer->lookupTransform(
        frame_ground, frame_camera, transform_time,
        rclcpp::Duration::from_seconds(0.2));
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "Failure %s",
                ex.what());  // Print exception which was caught
    return;
  }

  if (m_gimbal_mode) {
      apply_gimbal_correction(transform_drone_to_ground);
  }

  double drone_x = transform_drone_to_ground.transform.translation.x;
  double drone_y = transform_drone_to_ground.transform.translation.y;
  double drone_z = transform_drone_to_ground.transform.translation.z;

  std::stringstream stream;

  stream << "{\"cam_corners\": [";

  // Update the transform_drone_to_ground with the new rotation that ignores roll and pitch
  for (int y = 0; y < 3496; y += 3495) {
    for (int x = 0; x < 4656; x += 4655) {
        cv::Point3d cameraVectorCV; //replacement for segfaulting error in projectPixelTo3dRay
        auto uv_rect = rectifyPoint(cv::Point2d(x, y), 
          camera_model.intrinsicMatrix(), 
          camera_model.distortionCoeffs(), 
          camera_model.rotationMatrix(), 
          camera_model.projectionMatrix());


        cameraVectorCV.x = (float)(uv_rect.x - camera_model.cx()) / camera_model.fx() / 1.0;
        cameraVectorCV.y = (float)(uv_rect.y - camera_model.cy()) / camera_model.fy() / 1.0;
        cameraVectorCV.z = 1.0f;

        geometry_msgs::msg::PointStamped cameraVector;
        cameraVector.point.x = drone_z * cameraVectorCV.x;  // Stretch to correct distance
        cameraVector.point.y = drone_z * cameraVectorCV.y;
        cameraVector.point.z = drone_z * cameraVectorCV.z;
        cameraVector.header.stamp = transform_time;
        cameraVector.header.frame_id = frame_camera;

        geometry_msgs::msg::PointStamped cameraVectorGround;

        // Transfer from camera frame to body frame
        tf2::doTransform(cameraVector, cameraVectorGround,
                          transform_drone_to_ground);
        
        double local_e = cameraVectorGround.point.x;
        double local_n = cameraVectorGround.point.y;
        double local_u = cameraVectorGround.point.z;

        if (x+y != 0)stream << ",";
        stream << "[";
        
        stream << local_e << ", " << local_n << ", " << local_u << "]";
    }
  }
  stream << "], \"drone_pos\": [" << drone_x << ", " << drone_y << "]";

  if (detections_buffer.size()) {
    stream << ", \"detections\": [";

    for (int i = 0; i < detections_buffer.size(); ++i) {
      if (i > 0)stream << ", ";
      stream << "[";

      for (int j = 0; j < 4; ++j) {
        if (j > 0)stream << ", ";
        stream << detections_buffer[i][j];
      }
      stream << "]";
    }
    stream << "]";
    detections_buffer.clear();
  }

  stream << "}";
  std_msgs::msg::String pub_msg;
  pub_msg.data = stream.str();
  json_data_publisher->publish(pub_msg);
}



/*
* Saves the center coordinates of the detection (in local-ENU) whenever a detection is made
*/
void DetectionEstimator::detections_callback(
    suas24_interfaces::msg::Classification::ConstSharedPtr detection_msg) {

  std::optional<geometry_msgs::msg::PointStamped> central_detection;
  float central_detection_deviation;

  geometry_msgs::msg::TransformStamped transform_drone_to_ground;

  try {
    transform_drone_to_ground = tf_buffer->lookupTransform(
        frame_ground, frame_camera, detection_msg->header.stamp,
        rclcpp::Duration::from_seconds(0));
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "Failure %s",
                ex.what());  // Print exception which was caught
    return;
  }
  
  if (m_gimbal_mode) {
      apply_gimbal_correction(transform_drone_to_ground);
  }

  int x_cam = detection_msg->center_x;
  int y_cam = detection_msg->center_y; 

  double drone_x = transform_drone_to_ground.transform.translation.x;
  double drone_y = transform_drone_to_ground.transform.translation.y;
  double drone_z = transform_drone_to_ground.transform.translation.z;

  cv::Point3d cameraVectorCV; //replacement for segfaulting error in projectPixelTo3dRay

  // Unsure if we need rectification???
  auto uv_rect = rectifyPoint(cv::Point2d(x_cam, y_cam), 
    camera_model.intrinsicMatrix(), 
    camera_model.distortionCoeffs(), 
    camera_model.rotationMatrix(), 
    camera_model.projectionMatrix());

  cameraVectorCV.x = (uv_rect.x - camera_model.cx() - camera_model.Tx()) / camera_model.fx(); // / 10.0;
  cameraVectorCV.y = (uv_rect.y - camera_model.cy() - camera_model.Ty()) / camera_model.fy(); // / 10.0;
  cameraVectorCV.z = 1.0f;

  geometry_msgs::msg::PointStamped cameraVector;
  cameraVector.point.x = drone_z * cameraVectorCV.x;  // Stretch to correct distance
  cameraVector.point.y = drone_z * cameraVectorCV.y;
  cameraVector.point.z = drone_z * cameraVectorCV.z;
  cameraVector.header.stamp = detection_msg->header.stamp;
  cameraVector.header.frame_id = frame_camera;

  geometry_msgs::msg::PointStamped cameraVectorGround;

  // Transfer from camera frame to body frame
  tf2::doTransform(cameraVector, cameraVectorGround,
                    transform_drone_to_ground);

  if (m_should_publish_json) {
      detections_buffer.push_back({
        (float)cameraVectorGround.point.x, 
        (float)cameraVectorGround.point.y, 
        (float)drone_x, 
        (float)drone_y
      });
  }

  // Save the current detection in each of the standard object indecies along with the confidence
  bool over_threshold = false; 
  for (int i = 0; i < standard_objects_size; i++) {
    const float score = detection_msg->confidence[i].conf_global;
    RCLCPP_INFO(this->get_logger(), "DETECTION %d, score: %f", i, score);
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

  RCLCPP_INFO(this->get_logger(), "DROP LOCATION: x: %f, y: %f, z: %f, score: ",
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

  width = std::max(width, 1);
  height = std::max(height, 1);

  cv::Mat img = cv::Mat::zeros(height, width, CV_32F);

  // RCLCPP_INFO(this->get_logger(), "Initializing matrix for %s: %i x %i",
  //             object_id.c_str(), width, height);

  for (const auto& detection : detection_points.at(object_index)) {
    const int x_img = (detection.x - x_min) / spatial_resolution;
    const int y_img = (detection.y - y_min) / spatial_resolution;

    img.at<float>(y_img, x_img) += detection.confidence;
  }

  cv::Mat img_filtered(height, width, CV_32F);
  cv::sepFilter2D(img, img_filtered, -1, kernel, kernel);

  if (debug) {
    save_debug_image(img_filtered, "droppoint_v3_" + object_id + ".png");
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

  std::array<geometry_msgs::msg::PointStamped, 5> drop_points;

  for (int i = 0; i < standard_objects_size/*standard_objects_size + 1*/; i++) {
    if (!success && !detection_points.at(i).empty()) {
      success = true;
    }

    const auto dropPoint = get_drop_point(i);

    drop_points[i].point.x = dropPoint.x;
    drop_points[i].point.y = dropPoint.y;
    drop_points[i].point.z = 25.0;

    drop_points[i].header.stamp = now();
    drop_points[i].header.frame_id = frame_ground;

    RCLCPP_INFO(this->get_logger(), "GIVING DROP LOCATION OBJECT %d: %f %f", i, dropPoint.x, dropPoint.y);

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
      // break; for duplicates
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


