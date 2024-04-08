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

#include <cstdint>
#include <std_msgs/msg/detail/string__struct.hpp>
#include <string>
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

  detection_points.resize(5); //must correspond to size of 'confidence' in suas24_interfaces::msg::Classification

  this->declare_parameter<std::string>("frame_camera", "camera");
  this->declare_parameter<std::string>("frame_ground", "local_enu");
  //populate the fields
  this->get_parameter<std::string>("frame_camera", frame_camera);
  this->get_parameter<std::string>("frame_ground", frame_ground);

  //send drop points of objects
  this->declare_parameter<std::string>("oserv_drop_points", "~/drop_points");
  this->get_parameter<std::string>("oserv_drop_points", oserv_drop_points);
  drop_points_service = create_service<suas24_interfaces::srv::DropPointInfo>(
      oserv_drop_points,
      std::bind(&DetectionEstimator::drop_points_callback, this,
                std::placeholders::_1, std::placeholders::_2));

  debug_service = create_service<suas24_interfaces::srv::Debug>(
    "debug_service",
    std::bind(&DetectionEstimator::debug_callback, this,
              std::placeholders::_1, std::placeholders::_2)
  );

  //publish local-ENU coordinates
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

  //get intrinsic camera parameters
  this->declare_parameter<std::string>("itopic_camera_info", "/perception/camera_info");
  this->get_parameter<std::string>("itopic_camera_info", itopic_camera_info);
  camera_info_subscriber = create_subscription<sensor_msgs::msg::CameraInfo>(
      itopic_camera_info, 10,
      std::bind(&DetectionEstimator::camera_info_callback, this,
                std::placeholders::_1));

  debug = true;


  //publish images for debugging
  bool debug_heatmap = true;
  if(debug_heatmap == true){ //rework this
    visualization_heatmap_publisher = create_publisher<suas24_interfaces::msg::VisualizationImgs>("/viz/heatmap", 10);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&DetectionEstimator::visualization_callback, this));
  }

  // The kernel size is an odd value corresponding to 5 meters on the ground
  const int kernel_size = std::round(5.0 / spatial_resolution);
  kernel =
      cv::getGaussianKernel(kernel_size + (1 - kernel_size % 2), 0, CV_32F);
}

void DetectionEstimator::debug_callback(suas24_interfaces::srv::Debug::Request::SharedPtr req, suas24_interfaces::srv::Debug::Response::SharedPtr res){
    std_msgs::msg::String msg;
    debug_string = "";
    // for (const auto& detection : detection_points.at(req->index)){
    //   debug_string += std::to_string(detection.confidence) + " ";
    // }

    float x_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest();

    for (auto detection : detection_points.at(2)) {
      x_min = std::min(x_min, detection.x);
      //x_min = -30;
      x_max = std::max(x_max, detection.x);
      //x_max = 30;
      y_min = std::min(y_min, detection.y);
      //y_min = -30;
      y_max = std::max(y_max, detection.y);
      //y_max = 30;
    }
    
    debug_string = "x_min:" + std::to_string(x_min) + ", x_max:" + std::to_string(x_max) + ", y_min:" + std::to_string(y_min) + ", y_max:" + std::to_string(y_max); 

    msg.data = debug_string;
    res->response = msg;
    RCLCPP_INFO(this->get_logger(), "DEBUG SERVICE");
}

void DetectionEstimator::visualization_callback(){
  
  if (!detection_points.at(0).empty()){ //dont publish until we have detection points
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


void DetectionEstimator::detections_callback(
    //vision_msgs::msg::Detection2DArray::ConstSharedPtr msg) {
    suas24_interfaces::msg::Classification::ConstSharedPtr msg){
    //trenger: 
    // - drone x,y,z —> /perception/camera_info
    // - detection.header.stamp -> /classiciation
    // - bbox x, y -> /classification

  std::cout << "START DETECTION CALLBACK" << std::endl;
  RCLCPP_INFO(this->get_logger(), "detections callback");
  std::optional<geometry_msgs::msg::PointStamped> central_detection;
  float central_detection_deviation;
  std::vector<suas24_interfaces::msg::Classification::ConstSharedPtr> detections;
  detections.push_back(msg);
  for (auto detection : detections/*msg->detections*/) { //HMMMMM
    geometry_msgs::msg::TransformStamped transform_drone_to_ground;
    geometry_msgs::msg::TransformStamped transform_ground_to_drone;

    try {
      transform_drone_to_ground = tf_buffer->lookupTransform(
          frame_ground, frame_camera, detection->header.stamp, //might fail
          rclcpp::Duration::from_seconds(1));
      transform_ground_to_drone = tf_buffer->lookupTransform(
          frame_camera, frame_ground, detection->header.stamp,
          rclcpp::Duration::from_seconds(1));
    } catch (tf2::TransformException& ex) {
      RCLCPP_WARN(this->get_logger(), "Failure %s",
                  ex.what());  // Print exception which was caught
      return;
    }

    // int x_cam = detection.bbox.center.x;
    // int y_cam = detection.bbox.center.y;
    int x_cam =  detection->center_x;
    int y_cam = detection->center_y;

    double drone_x = transform_drone_to_ground.transform.translation.x;
    double drone_y = transform_drone_to_ground.transform.translation.y;
    double drone_z = transform_drone_to_ground.transform.translation.z;


    // auto cameraVectorCV =
    //     camera_model.projectPixelTo3dRay(cv::Point2d(x_cam, y_cam));
    cv::Point3d cameraVectorCV;
    cameraVectorCV.x = (x_cam - camera_model.cx() -camera_model.Tx()) / camera_model.fx();
    cameraVectorCV.y = (y_cam - camera_model.cy() -camera_model.Ty()) / camera_model.fy();
    cameraVectorCV.z = 1.0f;

    geometry_msgs::msg::PointStamped cameraVector;
    cameraVector.point.x =
        drone_z * cameraVectorCV.x;  // Stretch to correct distance
    cameraVector.point.y = drone_z * cameraVectorCV.y;
    cameraVector.point.z = drone_z * cameraVectorCV.z;
    cameraVector.header.stamp = detection->header.stamp;
    cameraVector.header.frame_id = frame_camera;

    // print before transform
    // RCLCPP_INFO(this->get_logger(), "Camera vector x: %f, y: %f, z: %f",
    // cameraVector.point.x, cameraVector.point.y, cameraVector.point.z);

    geometry_msgs::msg::PointStamped cameraVectorGround;

    tf2::doTransform(cameraVector, cameraVectorGround,
                     transform_drone_to_ground); //Transforms from drone frame to the ground frame (local ENU)

    bool over_threshold = false;
    //for (int i = 0; i < standard_objects.size() + 1; i++) { //one for each standard object
    for (int i = 0; i < 5; i++) {
      //const float score = detection.results.at(i).score; // tilsvarende Classification.confidence[i]
      const float score = detection->confidence[i].conf_global;
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

    RCLCPP_INFO(this->get_logger(), "DROP LOCATION: x: %f, y: %f, z: %f",
                cameraVectorGround.point.x, cameraVectorGround.point.y,
                cameraVectorGround.point.z); //local ENU coords
  }

  // points_publisher->publish(central_detection.value());
  if (central_detection.has_value()) {
    RCLCPP_INFO(this->get_logger(), "Publishing localization points");
    points_publisher->publish(central_detection.value());
  }
}

void DetectionEstimator::camera_info_callback(
    sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Initializing camera model");
  camera_model.fromCameraInfo(msg);

  camera_info_subscriber.reset();  // unsubscribe

  std::cout << "CAMERA INFO" << std::endl;
  detections_subscriber =
      create_subscription<suas24_interfaces::msg::Classification>( //******
          itopic_classifications, 10,
          std::bind(&DetectionEstimator::detections_callback, this,
                    std::placeholders::_1));
}

DropPointImage DetectionEstimator::get_drop_point_image(
    const int object_index) {
  //if (object_index >= standard_objects.size() + 1) {
    if (object_index >= 5) {
    RCLCPP_ERROR(this->get_logger(), "Object index out of range");
    return {
        cv::Mat::zeros(1, 1, CV_32F),
        0,
        0,
    };
  }

  const std::string object_id =
      //(object_index >= standard_objects.size())
      (object_index >= 5)
          ? "emergent"
        //   : standard_objects.at(object_index).to_string();
          : "test";

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
    //x_min = -30;
    x_max = std::max(x_max, detection.x);
    //x_max = 30;
    y_min = std::min(y_min, detection.y);
    //y_min = -30;
    y_max = std::max(y_max, detection.y);
    //y_max = 30;
  }
  //x_min:-6.709982, x_max:2.614120, y_min:9.370054, y_max:31.942196
  

  int width = std::abs(std::ceil((x_max - x_min) / spatial_resolution));
  int height = std::abs(std::ceil((y_max - y_min) / spatial_resolution));
  
  if(debug){ //fix the image size to make it easier to visualize
    width = 90;
    height = 160;
  }


  width = std::max(width, 1);
  height = std::max(height, 1);

  cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);

//   RCLCPP_INFO(this->get_logger(), "Initializing matrix for %s: %i x %i", 
//               object_id.c_str(), width, height);

  for (const auto& detection : detection_points.at(object_index)) {
    const int x_img = (detection.x - x_min) / spatial_resolution;
    const int y_img = (detection.y - y_min) / spatial_resolution;
    //TODO check row col
    RCLCPP_INFO(this->get_logger(), "CONF: %f", detection.confidence);
    img.at<uint>(x_img, y_img) = 255;// * detection.confidence; //Scale white pixel by the confidence value
  }

  cv::Mat img_filtered(height, width, CV_32F);
  cv::sepFilter2D(img, img_filtered, -1, kernel, kernel);

  if (debug) {
    save_debug_image(img_filtered, "droppoint_" + object_id + std::to_string(object_index) + ".png");
  }

  return {img_filtered, static_cast<int>(x_min), static_cast<int>(y_min)};
}

cv::Point3f DetectionEstimator::get_drop_point(const int object_index) {
//   if (object_index >= standard_objects.size() + 1) {
    if (object_index >= 5) {
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

  float max_loc_f_x = static_cast<float>(max_loc.x) * spatial_resolution;
  float max_loc_f_y = static_cast<float>(max_loc.y) * spatial_resolution;
  cv::Point2f max_loc_f = {max_loc_f_x, max_loc_f_y};
  max_loc_f.x += dropPointImage.min_x;
  max_loc_f.y += dropPointImage.min_y;

  return {max_loc_f.x, max_loc_f.y, static_cast<float>(max_val)};
}

void DetectionEstimator::drop_points_callback(
    suas24_interfaces::srv::DropPointInfo::Request::SharedPtr req,
    suas24_interfaces::srv::DropPointInfo::Response::SharedPtr resp) {
  bool success = false;

  std::array<geometry_msgs::msg::Point, 5> drop_points;
  std::array<int, 5> confidences;

//   for (int i = 0; i < standard_objects.size() + 1; i++) {
    for (int i = 0; i < 5; i++) {
    if (!success && !detection_points.at(i).empty()) {
      success = true;
    }

    const auto dropPoint = get_drop_point(i);

    drop_points[i].x = dropPoint.x;
    drop_points[i].y = dropPoint.y;
    drop_points[i].z = 0;

    confidences[i] = dropPoint.z;

    // const std::string object_id = (i >= standard_objects.size())
    //                                   ? "emergent"
    //                                   : standard_objects.at(i).to_string();
    const std::string object_id = (i >= 5 ? "emergent" : "standard_obj");
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


//Refactor:
sensor_msgs::msg::Image
DetectionEstimator::cv_mat_to_ros(const cv::Mat& mat) {
    sensor_msgs::msg::Image ros_img;

    ros_img.width = mat.cols;
    ros_img.height = mat.rows;

    ros_img.encoding = "bgr8";

    ros_img.is_bigendian = false;

    ros_img.header = std_msgs::msg::Header();

    ros_img.step = ros_img.width * mat.elemSize();

    size_t size = ros_img.step * ros_img.height;
    ros_img.data.resize(size);

    uchar *ros_data_ptr = reinterpret_cast<uchar *>(&ros_img.data[0]);
    uchar *cv_data_ptr = mat.data;

    memcpy(ros_data_ptr, cv_data_ptr, size);

    return ros_img;
}


//suas24_interfaces.srv.Debug_Response(response=std_msgs.msg.String(data='x_min:-6.709982, x_max:2.614120, y_min:9.370054, y_max:31.942196'))
