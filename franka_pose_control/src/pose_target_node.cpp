// Copyright 2026 The franka_pose_control Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/duration.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit_msgs/action/execute_trajectory.hpp>
#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/srv/get_cartesian_path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace {
volatile std::sig_atomic_t shutdown_requested = 0;

void signalHandler(int /*signal*/) {
  shutdown_requested = 1;
}

bool isFinite(double value) {
  return std::isfinite(value);
}

double durationSeconds(const builtin_interfaces::msg::Duration& duration) {
  return static_cast<double>(duration.sec) + static_cast<double>(duration.nanosec) * 1e-9;
}

double positionDistance(const geometry_msgs::msg::Pose& lhs, const geometry_msgs::msg::Pose& rhs) {
  const double dx = lhs.position.x - rhs.position.x;
  const double dy = lhs.position.y - rhs.position.y;
  const double dz = lhs.position.z - rhs.position.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double quaternionNorm(const geometry_msgs::msg::Quaternion& quaternion) {
  return std::sqrt(quaternion.x * quaternion.x + quaternion.y * quaternion.y +
                   quaternion.z * quaternion.z + quaternion.w * quaternion.w);
}

void normalizeQuaternion(geometry_msgs::msg::Quaternion& quaternion) {
  const double norm = quaternionNorm(quaternion);
  quaternion.x /= norm;
  quaternion.y /= norm;
  quaternion.z /= norm;
  quaternion.w /= norm;
}

double orientationDistance(const geometry_msgs::msg::Quaternion& lhs,
                           const geometry_msgs::msg::Quaternion& rhs) {
  const double dot = std::abs(lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w);
  return 2.0 * std::acos(std::clamp(dot, 0.0, 1.0));
}

}  // namespace

class PoseTargetNode : public rclcpp::Node {
 public:
  using MoveGroup = moveit_msgs::action::MoveGroup;
  using MoveGroupGoalHandle = rclcpp_action::ClientGoalHandle<MoveGroup>;
  using ExecuteTrajectory = moveit_msgs::action::ExecuteTrajectory;
  using ExecuteGoalHandle = rclcpp_action::ClientGoalHandle<ExecuteTrajectory>;
  using GetCartesianPath = moveit_msgs::srv::GetCartesianPath;

  PoseTargetNode() : Node("franka_pose_target") {
    planning_group_ = declare_parameter<std::string>("planning_group", "fr3_arm");
    end_effector_link_ = declare_parameter<std::string>("end_effector_link", "fr3_hand_tcp");
    reference_frame_ = declare_parameter<std::string>("reference_frame", "base");
    controller_name_ = declare_parameter<std::string>("controller_name", "fr3_arm_controller");
    current_pose_topic_ = declare_parameter<std::string>(
        "current_pose_topic", "/franka_robot_state_broadcaster/current_pose");
    move_group_action_ = declare_parameter<std::string>("move_group_action", "/move_action");
    execute_action_ =
        declare_parameter<std::string>("execute_trajectory_action", "/execute_trajectory");
    cartesian_path_service_name_ =
        declare_parameter<std::string>("cartesian_path_service", "/compute_cartesian_path");

    velocity_scaling_ = declare_parameter<double>("velocity_scaling", 0.05);
    acceleration_scaling_ = declare_parameter<double>("acceleration_scaling", 0.05);
    planning_time_ = declare_parameter<double>("planning_time", 5.0);
    planning_attempts_ = declare_parameter<int>("planning_attempts", 5);
    position_tolerance_ = declare_parameter<double>("position_tolerance", 0.005);
    orientation_tolerance_ = declare_parameter<double>("orientation_tolerance", 0.02);
    max_translation_ = declare_parameter<double>("max_translation_per_goal", 0.05);
    max_rotation_ = declare_parameter<double>("max_rotation_per_goal", 0.35);
    max_current_pose_age_ = declare_parameter<double>("max_current_pose_age", 1.0);
    max_plan_age_ = declare_parameter<double>("max_plan_age", 10.0);
    max_start_translation_drift_ = declare_parameter<double>("max_start_translation_drift", 0.01);
    max_start_rotation_drift_ = declare_parameter<double>("max_start_rotation_drift", 0.10);
    workspace_min_ = declare_parameter<std::vector<double>>("workspace_min", {0.10, -0.60, 0.10});
    workspace_max_ = declare_parameter<std::vector<double>>("workspace_max", {0.85, 0.60, 0.90});
    home_joint_names_ = declare_parameter<std::vector<std::string>>(
        "home_joint_names", {"fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
                             "fr3_joint5", "fr3_joint6", "fr3_joint7"});
    home_joint_positions_ = declare_parameter<std::vector<double>>(
        "home_joint_positions", {0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.7854});
    home_joint_tolerance_ = declare_parameter<double>("home_joint_tolerance", 0.005);
    cartesian_eef_step_ = declare_parameter<double>("cartesian_eef_step", 0.005);
    cartesian_min_fraction_ = declare_parameter<double>("cartesian_min_fraction", 1.0);
    cartesian_revolute_jump_threshold_ =
        declare_parameter<double>("cartesian_revolute_jump_threshold", 0.20);
    max_cartesian_translation_ = declare_parameter<double>("max_cartesian_translation", 0.20);
    max_cartesian_rotation_ = declare_parameter<double>("max_cartesian_rotation", 0.35);
    max_cartesian_speed_ = declare_parameter<double>("max_cartesian_speed", 0.05);

    validateParameters();

    move_group_client_ = rclcpp_action::create_client<MoveGroup>(this, move_group_action_);
    execute_client_ = rclcpp_action::create_client<ExecuteTrajectory>(this, execute_action_);
    cartesian_path_client_ = create_client<GetCartesianPath>(cartesian_path_service_name_);
    planning_ready_service_ = create_service<std_srvs::srv::Trigger>(
        "~/planning_ready", [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                                   std_srvs::srv::Trigger::Response::SharedPtr response) {
          response->success = move_group_client_->action_server_is_ready() &&
                              cartesian_path_client_->service_is_ready();
          response->message = response->success ? "planning interfaces ready" :
              "waiting for MoveGroup and Cartesian path interfaces";
        });

    const auto state_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    current_pose_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        current_pose_topic_, state_qos,
        std::bind(&PoseTargetNode::currentPoseCallback, this, std::placeholders::_1));
    target_pose_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "~/target_pose", rclcpp::QoS(1),
        std::bind(&PoseTargetNode::targetPoseCallback, this, std::placeholders::_1));
    linear_target_pose_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "~/linear_target_pose", rclcpp::QoS(1),
        std::bind(&PoseTargetNode::linearTargetPoseCallback, this, std::placeholders::_1));

    plan_home_service_ = create_service<std_srvs::srv::Trigger>(
        "~/plan_home", std::bind(&PoseTargetNode::planHomeCallback, this, std::placeholders::_1,
                                 std::placeholders::_2));
    execute_service_ = create_service<std_srvs::srv::Trigger>(
        "~/execute", std::bind(&PoseTargetNode::executeCallback, this, std::placeholders::_1,
                               std::placeholders::_2));
    stop_service_ = create_service<std_srvs::srv::Trigger>(
        "~/stop", std::bind(&PoseTargetNode::stopCallback, this, std::placeholders::_1,
                            std::placeholders::_2));

    display_trajectory_publisher_ = create_publisher<moveit_msgs::msg::DisplayTrajectory>(
        "/display_planned_path", rclcpp::QoS(1).reliable().transient_local());

    RCLCPP_INFO(get_logger(),
                "Ready in PLAN-ONLY mode. Use %s/target_pose for PTP or "
                "%s/linear_target_pose for a Cartesian line; call %s/execute only after "
                "inspecting the plan. PTP limits: translation %.3f m, rotation %.3f rad, speed "
                "%.2f.",
                get_fully_qualified_name(), get_fully_qualified_name(),
                get_fully_qualified_name(), max_translation_, max_rotation_, velocity_scaling_);
  }

  void requestSafeStop(const std::string& reason) {
    MoveGroupGoalHandle::SharedPtr planning_handle;
    ExecuteGoalHandle::SharedPtr execute_handle;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cached_plan_.reset();
      if (state_ == State::kIdle || state_ == State::kPlanReady) {
        state_ = State::kIdle;
        RCLCPP_WARN(get_logger(), "Cleared cached plan: %s", reason.c_str());
        return;
      }
      state_ = State::kStopping;
      planning_handle = planning_goal_handle_;
      execute_handle = execute_goal_handle_;
    }

    RCLCPP_WARN(get_logger(), "Requesting motion cancellation: %s", reason.c_str());
    if (planning_handle) {
      (void)move_group_client_->async_cancel_goal(planning_handle);
    }
    if (execute_handle) {
      (void)execute_client_->async_cancel_goal(execute_handle);
    }
  }

  bool motionActive() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ == State::kPlanning || state_ == State::kExecuting || state_ == State::kStopping;
  }

 private:
  enum class State { kIdle, kPlanning, kPlanReady, kExecuting, kStopping };

  struct CachedPlan {
    moveit_msgs::msg::RobotTrajectory trajectory;
    moveit_msgs::msg::RobotState start_state;
    geometry_msgs::msg::PoseStamped measured_start_pose;
    rclcpp::Time created_at;
  };

  void validateParameters() const {
    const auto in_unit_interval = [](double value) { return value > 0.0 && value <= 1.0; };
    if (!in_unit_interval(velocity_scaling_) || !in_unit_interval(acceleration_scaling_)) {
      throw std::invalid_argument("velocity_scaling and acceleration_scaling must be in (0, 1]");
    }
    if (!in_unit_interval(cartesian_min_fraction_)) {
      throw std::invalid_argument("cartesian_min_fraction must be in (0, 1]");
    }
    if (cartesian_path_service_name_.empty()) {
      throw std::invalid_argument("cartesian_path_service must not be empty");
    }
    if (workspace_min_.size() != 3 || workspace_max_.size() != 3) {
      throw std::invalid_argument("workspace_min and workspace_max must each contain 3 values");
    }
    for (size_t i = 0; i < 3; ++i) {
      if (!isFinite(workspace_min_[i]) || !isFinite(workspace_max_[i]) ||
          workspace_min_[i] >= workspace_max_[i]) {
        throw std::invalid_argument("each workspace_min value must be below workspace_max");
      }
    }
    if (home_joint_names_.empty() || home_joint_names_.size() != home_joint_positions_.size()) {
      throw std::invalid_argument(
          "home_joint_names and home_joint_positions must have the same non-zero size");
    }
    for (size_t i = 0; i < home_joint_names_.size(); ++i) {
      if (home_joint_names_[i].empty() || !isFinite(home_joint_positions_[i])) {
        throw std::invalid_argument("Home joint names must be non-empty and positions finite");
      }
      if (std::count(home_joint_names_.begin(), home_joint_names_.end(), home_joint_names_[i]) != 1) {
        throw std::invalid_argument("home_joint_names must not contain duplicates");
      }
    }
    if (planning_time_ <= 0.0 || planning_attempts_ < 1 || position_tolerance_ <= 0.0 ||
        orientation_tolerance_ <= 0.0 || max_translation_ <= 0.0 || max_rotation_ <= 0.0 ||
        max_current_pose_age_ <= 0.0 || max_plan_age_ <= 0.0 ||
        max_start_translation_drift_ <= 0.0 || max_start_rotation_drift_ <= 0.0 ||
        home_joint_tolerance_ <= 0.0 || cartesian_eef_step_ <= 0.0 ||
        cartesian_revolute_jump_threshold_ <= 0.0 || max_cartesian_translation_ <= 0.0 ||
        max_cartesian_rotation_ <= 0.0 || max_cartesian_speed_ <= 0.0) {
      throw std::invalid_argument("all timing, tolerance, and safety limits must be positive");
    }
  }

  void currentPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr message) {
    if (message->header.frame_id != reference_frame_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Ignoring current pose in frame '%s'; expected '%s'.",
                           message->header.frame_id.c_str(), reference_frame_.c_str());
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    current_pose_ = *message;
  }

  bool poseIsFinite(const geometry_msgs::msg::Pose& pose) const {
    return isFinite(pose.position.x) && isFinite(pose.position.y) && isFinite(pose.position.z) &&
           isFinite(pose.orientation.x) && isFinite(pose.orientation.y) &&
           isFinite(pose.orientation.z) && isFinite(pose.orientation.w);
  }

  bool currentPoseIsFresh(const geometry_msgs::msg::PoseStamped& pose, std::string& reason) const {
    const rclcpp::Time stamp(pose.header.stamp, get_clock()->get_clock_type());
    const double age = (now() - stamp).seconds();
    if (!isFinite(age) || age < -0.5 || age > max_current_pose_age_) {
      reason =
          "current TCP pose is stale or has an incompatible clock (age=" + std::to_string(age) +
          " s)";
      return false;
    }
    return true;
  }

  bool validateTarget(geometry_msgs::msg::PoseStamped& target,
                      const geometry_msgs::msg::PoseStamped& current,
                      double translation_limit, double rotation_limit,
                      std::string& reason) const {
    if (target.header.frame_id != reference_frame_) {
      reason = "target frame must be '" + reference_frame_ + "'";
      return false;
    }
    if (!poseIsFinite(target.pose)) {
      reason = "target contains NaN or infinity";
      return false;
    }
    const double quaternion_norm = quaternionNorm(target.pose.orientation);
    if (!isFinite(quaternion_norm) || quaternion_norm < 1e-6) {
      reason = "target quaternion is degenerate";
      return false;
    }
    normalizeQuaternion(target.pose.orientation);

    const std::vector<double> position = {target.pose.position.x, target.pose.position.y,
                                          target.pose.position.z};
    for (size_t i = 0; i < 3; ++i) {
      if (position[i] < workspace_min_[i] || position[i] > workspace_max_[i]) {
        reason = "target lies outside the configured workspace bounding box";
        return false;
      }
    }

    const double translation = positionDistance(target.pose, current.pose);
    const double rotation = orientationDistance(target.pose.orientation, current.pose.orientation);
    if (translation > translation_limit) {
      reason = "translation " + std::to_string(translation) + " m exceeds limit " +
               std::to_string(translation_limit) + " m";
      return false;
    }
    if (rotation > rotation_limit) {
      reason = "rotation " + std::to_string(rotation) + " rad exceeds limit " +
               std::to_string(rotation_limit) + " rad";
      return false;
    }
    return true;
  }

  MoveGroup::Goal makePlanningGoal() const {
    MoveGroup::Goal goal;
    // An empty diff retains the current state from MoveIt's planning scene.
    goal.request.start_state.is_diff = true;
    goal.request.group_name = planning_group_;
    goal.request.num_planning_attempts = planning_attempts_;
    goal.request.allowed_planning_time = planning_time_;
    goal.request.max_velocity_scaling_factor = velocity_scaling_;
    goal.request.max_acceleration_scaling_factor = acceleration_scaling_;
    goal.request.workspace_parameters.header.frame_id = reference_frame_;
    goal.request.workspace_parameters.min_corner.x = workspace_min_[0];
    goal.request.workspace_parameters.min_corner.y = workspace_min_[1];
    goal.request.workspace_parameters.min_corner.z = workspace_min_[2];
    goal.request.workspace_parameters.max_corner.x = workspace_max_[0];
    goal.request.workspace_parameters.max_corner.y = workspace_max_[1];
    goal.request.workspace_parameters.max_corner.z = workspace_max_[2];
    goal.planning_options.plan_only = true;
    goal.planning_options.look_around = false;
    goal.planning_options.replan = false;
    return goal;
  }

  bool submitPlanningGoal(MoveGroup::Goal goal,
                          const geometry_msgs::msg::PoseStamped& measured_start_pose,
                          const std::string& description, std::string& reason) {
    if (!move_group_client_->wait_for_action_server(std::chrono::seconds(0))) {
      reason = "MoveGroup action '" + move_group_action_ + "' is unavailable";
      return false;
    }

    uint64_t sequence;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ == State::kPlanning || state_ == State::kExecuting || state_ == State::kStopping) {
        reason = "node is busy or stopping";
        return false;
      }
      cached_plan_.reset();
      state_ = State::kPlanning;
      planning_goal_handle_.reset();
      sequence = ++plan_sequence_;
      pending_start_pose_ = measured_start_pose;
    }

    RCLCPP_INFO(get_logger(), "%s No motion will occur yet.", description.c_str());

    rclcpp_action::Client<MoveGroup>::SendGoalOptions options;
    options.goal_response_callback = [this,
                                      sequence](const MoveGroupGoalHandle::SharedPtr& goal_handle) {
      bool cancel_immediately = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sequence != plan_sequence_) {
          return;
        }
        if (!goal_handle) {
          state_ = State::kIdle;
          RCLCPP_ERROR(get_logger(), "MoveGroup rejected the planning request.");
          return;
        }
        planning_goal_handle_ = goal_handle;
        cancel_immediately = state_ == State::kStopping;
      }
      if (cancel_immediately) {
        (void)move_group_client_->async_cancel_goal(goal_handle);
      }
    };
    options.result_callback = [this, sequence](const MoveGroupGoalHandle::WrappedResult& result) {
      moveit_msgs::msg::DisplayTrajectory display;
      bool publish_display = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sequence != plan_sequence_) {
          return;
        }
        planning_goal_handle_.reset();
        if (state_ == State::kStopping) {
          state_ = State::kIdle;
          cached_plan_.reset();
          RCLCPP_WARN(get_logger(), "Discarded planning result because stop was requested.");
          return;
        }
        if (result.code != rclcpp_action::ResultCode::SUCCEEDED || !result.result ||
            result.result->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
          const int error_code = result.result ? result.result->error_code.val : 0;
          state_ = State::kIdle;
          cached_plan_.reset();
          RCLCPP_ERROR(get_logger(), "Planning did not succeed (action=%d, MoveIt=%d).",
                       static_cast<int>(result.code), error_code);
          return;
        }

        cached_plan_ = CachedPlan{result.result->planned_trajectory,
                                  result.result->trajectory_start, pending_start_pose_, now()};
        state_ = State::kPlanReady;
        display.model_id = "fr3";
        display.trajectory_start = cached_plan_->start_state;
        display.trajectory.push_back(cached_plan_->trajectory);
        publish_display = true;
      }
      if (publish_display) {
        display_trajectory_publisher_->publish(display);
        RCLCPP_WARN(get_logger(),
                    "Plan ready and displayed in RViz. Inspect it, then explicitly call ~/execute "
                    "within %.1f seconds.",
                    max_plan_age_);
      }
    };
    (void)move_group_client_->async_send_goal(goal, options);
    return true;
  }

  bool getFreshCurrentPose(geometry_msgs::msg::PoseStamped& current, std::string& reason) const {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ == State::kPlanning || state_ == State::kExecuting || state_ == State::kStopping) {
        reason = "node is busy or stopping";
        return false;
      }
      if (!current_pose_) {
        reason = "no measured TCP pose received yet";
        return false;
      }
      current = *current_pose_;
    }
    return currentPoseIsFresh(current, reason);
  }

  void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr message) {
    geometry_msgs::msg::PoseStamped current;
    std::string reason;
    if (!getFreshCurrentPose(current, reason)) {
      RCLCPP_ERROR(get_logger(), "Rejected target: %s", reason.c_str());
      return;
    }

    geometry_msgs::msg::PoseStamped target = *message;
    if (!validateTarget(target, current, max_translation_, max_rotation_, reason)) {
      RCLCPP_ERROR(get_logger(), "Rejected target: %s", reason.c_str());
      return;
    }

    auto goal = makePlanningGoal();
    goal.request.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(
        end_effector_link_, target, position_tolerance_, orientation_tolerance_));
    const std::string description =
        "Planning only to Cartesian target [" + std::to_string(target.pose.position.x) + ", " +
        std::to_string(target.pose.position.y) + ", " +
        std::to_string(target.pose.position.z) + "] in '" + reference_frame_ + "'.";
    if (!submitPlanningGoal(std::move(goal), current, description, reason)) {
      RCLCPP_ERROR(get_logger(), "Rejected target: %s", reason.c_str());
    }
  }

  void linearTargetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr message) {
    geometry_msgs::msg::PoseStamped current;
    std::string reason;
    if (!getFreshCurrentPose(current, reason)) {
      RCLCPP_ERROR(get_logger(), "Rejected linear target: %s", reason.c_str());
      return;
    }

    geometry_msgs::msg::PoseStamped target = *message;
    if (!validateTarget(target, current, max_cartesian_translation_, max_cartesian_rotation_,
                        reason)) {
      RCLCPP_ERROR(get_logger(), "Rejected linear target: %s", reason.c_str());
      return;
    }
    if (!cartesian_path_client_->service_is_ready()) {
      RCLCPP_ERROR(get_logger(), "Rejected linear target: Cartesian path service '%s' is unavailable.",
                   cartesian_path_service_name_.c_str());
      return;
    }

    auto request = std::make_shared<GetCartesianPath::Request>();
    request->header.frame_id = reference_frame_;
    request->header.stamp = now();
    request->start_state.is_diff = true;
    request->group_name = planning_group_;
    request->link_name = end_effector_link_;
    request->waypoints.push_back(target.pose);
    request->max_step = cartesian_eef_step_;
    request->jump_threshold = 0.0;
    request->prismatic_jump_threshold = 0.0;
    request->revolute_jump_threshold = cartesian_revolute_jump_threshold_;
    request->avoid_collisions = true;
    request->max_velocity_scaling_factor = velocity_scaling_;
    request->max_acceleration_scaling_factor = acceleration_scaling_;
    request->cartesian_speed_limited_link = end_effector_link_;
    request->max_cartesian_speed = max_cartesian_speed_;

    uint64_t sequence;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ == State::kPlanning || state_ == State::kExecuting ||
          state_ == State::kStopping) {
        RCLCPP_ERROR(get_logger(),
                     "Rejected linear target: node state changed while validating it.");
        return;
      }
      cached_plan_.reset();
      state_ = State::kPlanning;
      planning_goal_handle_.reset();
      sequence = ++plan_sequence_;
      pending_start_pose_ = current;
    }

    RCLCPP_INFO(get_logger(),
                "Computing collision-checked Cartesian line to [%.4f, %.4f, %.4f] with %.4f m "
                "steps. No motion will occur yet.",
                target.pose.position.x, target.pose.position.y, target.pose.position.z,
                cartesian_eef_step_);

    (void)cartesian_path_client_->async_send_request(
        request, [this, sequence](rclcpp::Client<GetCartesianPath>::SharedFuture future) {
          GetCartesianPath::Response::SharedPtr result;
          try {
            result = future.get();
          } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (sequence == plan_sequence_) {
              state_ = State::kIdle;
              cached_plan_.reset();
              RCLCPP_ERROR(get_logger(), "Cartesian path service failed: %s", error.what());
            }
            return;
          }

          moveit_msgs::msg::DisplayTrajectory display;
          bool publish_display = false;
          {
            std::lock_guard<std::mutex> lock(mutex_);
            if (sequence != plan_sequence_) {
              return;
            }
            if (state_ == State::kStopping) {
              state_ = State::kIdle;
              cached_plan_.reset();
              RCLCPP_WARN(get_logger(),
                          "Discarded Cartesian path because stop was requested.");
              return;
            }

            const auto& trajectory = result->solution.joint_trajectory;
            bool timing_is_valid = trajectory.points.size() >= 2;
            double previous_time = -1.0;
            for (const auto& point : trajectory.points) {
              const double point_time = durationSeconds(point.time_from_start);
              if (!isFinite(point_time) || point_time <= previous_time ||
                  point.positions.size() != trajectory.joint_names.size()) {
                timing_is_valid = false;
                break;
              }
              previous_time = point_time;
            }
            timing_is_valid = timing_is_valid && previous_time > 0.0;

            if (result->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS ||
                !isFinite(result->fraction) ||
                result->fraction + 1e-6 < cartesian_min_fraction_ || !timing_is_valid) {
              state_ = State::kIdle;
              cached_plan_.reset();
              RCLCPP_ERROR(get_logger(),
                           "Cartesian planning rejected (MoveIt=%d, fraction=%.6f, "
                           "timed_trajectory=%s). Required fraction is %.6f.",
                           result->error_code.val, result->fraction,
                           timing_is_valid ? "yes" : "no", cartesian_min_fraction_);
              return;
            }

            cached_plan_ = CachedPlan{result->solution, result->start_state,
                                      pending_start_pose_, now()};
            state_ = State::kPlanReady;
            display.model_id = "fr3";
            display.trajectory_start = cached_plan_->start_state;
            display.trajectory.push_back(cached_plan_->trajectory);
            publish_display = true;
          }
          if (publish_display) {
            display_trajectory_publisher_->publish(display);
            RCLCPP_WARN(get_logger(),
                        "Complete Cartesian line ready (fraction %.6f) and displayed in RViz. "
                        "Inspect it, then explicitly call ~/execute within %.1f seconds.",
                        result->fraction, max_plan_age_);
          }
        });
  }

  void planHomeCallback(const std_srvs::srv::Trigger::Request::SharedPtr& /*request*/,
                        std_srvs::srv::Trigger::Response::SharedPtr response) {
    geometry_msgs::msg::PoseStamped current;
    std::string reason;
    if (!getFreshCurrentPose(current, reason)) {
      response->success = false;
      response->message = "Home planning rejected: " + reason;
      return;
    }

    auto goal = makePlanningGoal();
    moveit_msgs::msg::Constraints home_constraints;
    home_constraints.name = "configured_home_joint_pose";
    for (size_t i = 0; i < home_joint_names_.size(); ++i) {
      moveit_msgs::msg::JointConstraint constraint;
      constraint.joint_name = home_joint_names_[i];
      constraint.position = home_joint_positions_[i];
      constraint.tolerance_above = home_joint_tolerance_;
      constraint.tolerance_below = home_joint_tolerance_;
      constraint.weight = 1.0;
      home_constraints.joint_constraints.push_back(constraint);
    }
    goal.request.goal_constraints.push_back(home_constraints);

    if (!submitPlanningGoal(std::move(goal), current,
                            "Planning only to the configured Home joint pose.", reason)) {
      response->success = false;
      response->message = "Home planning rejected: " + reason;
      return;
    }
    response->success = true;
    response->message =
        "Home planning request accepted; inspect the RViz trajectory before calling ~/execute";
  }

  void executeCallback(const std_srvs::srv::Trigger::Request::SharedPtr& /*request*/,
                       std_srvs::srv::Trigger::Response::SharedPtr response) {
    if (!execute_client_->wait_for_action_server(std::chrono::seconds(0))) {
      response->success = false;
      response->message = "ExecuteTrajectory action is unavailable";
      return;
    }

    CachedPlan plan;
    uint64_t sequence;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ != State::kPlanReady || !cached_plan_) {
        response->success = false;
        response->message = "no reviewed plan is ready";
        return;
      }
      if (!current_pose_) {
        response->success = false;
        response->message = "no measured TCP pose is available";
        return;
      }
      plan = *cached_plan_;
      const auto current = *current_pose_;

      std::string reason;
      if (!currentPoseIsFresh(current, reason)) {
        cached_plan_.reset();
        state_ = State::kIdle;
        response->success = false;
        response->message = reason;
        return;
      }
      const double plan_age = (now() - plan.created_at).seconds();
      if (!isFinite(plan_age) || plan_age < 0.0 || plan_age > max_plan_age_) {
        cached_plan_.reset();
        state_ = State::kIdle;
        response->success = false;
        response->message = "plan expired; publish the target again";
        return;
      }
      if (positionDistance(current.pose, plan.measured_start_pose.pose) >
              max_start_translation_drift_ ||
          orientationDistance(current.pose.orientation, plan.measured_start_pose.pose.orientation) >
              max_start_rotation_drift_) {
        cached_plan_.reset();
        state_ = State::kIdle;
        response->success = false;
        response->message = "robot moved since planning; publish the target again";
        return;
      }

      state_ = State::kExecuting;
      execute_goal_handle_.reset();
      sequence = ++execute_sequence_;
    }

    ExecuteTrajectory::Goal goal;
    goal.trajectory = plan.trajectory;
    if (!controller_name_.empty()) {
      goal.controller_names.push_back(controller_name_);
    }

    rclcpp_action::Client<ExecuteTrajectory>::SendGoalOptions options;
    options.goal_response_callback = [this,
                                      sequence](const ExecuteGoalHandle::SharedPtr& goal_handle) {
      bool cancel_immediately = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sequence != execute_sequence_) {
          return;
        }
        if (!goal_handle) {
          state_ = State::kIdle;
          cached_plan_.reset();
          RCLCPP_ERROR(get_logger(), "ExecuteTrajectory rejected the trajectory.");
          return;
        }
        execute_goal_handle_ = goal_handle;
        cancel_immediately = state_ == State::kStopping;
      }
      if (cancel_immediately) {
        (void)execute_client_->async_cancel_goal(goal_handle);
      }
    };
    options.result_callback = [this, sequence](const ExecuteGoalHandle::WrappedResult& result) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (sequence != execute_sequence_) {
        return;
      }
      execute_goal_handle_.reset();
      state_ = State::kIdle;
      cached_plan_.reset();
      const int error_code = result.result ? result.result->error_code.val : 0;
      if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result &&
          error_code == moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        RCLCPP_INFO(get_logger(), "Trajectory execution completed successfully.");
      } else if (result.code == rclcpp_action::ResultCode::CANCELED) {
        RCLCPP_WARN(get_logger(), "Trajectory execution was canceled.");
      } else {
        RCLCPP_ERROR(get_logger(), "Trajectory execution failed (action=%d, MoveIt=%d).",
                     static_cast<int>(result.code), error_code);
      }
    };
    (void)execute_client_->async_send_goal(goal, options);

    response->success = true;
    response->message = "execution request accepted; call ~/stop or press Ctrl+C to cancel";
    RCLCPP_WARN(get_logger(), "EXECUTING reviewed plan. Use ~/stop or Ctrl+C to cancel.");
  }

  void stopCallback(const std_srvs::srv::Trigger::Request::SharedPtr& /*request*/,
                    std_srvs::srv::Trigger::Response::SharedPtr response) {
    requestSafeStop("~/stop service called");
    response->success = true;
    response->message = "stop/cancel requested and cached plan cleared";
  }

  mutable std::mutex mutex_;
  State state_{State::kIdle};
  std::optional<geometry_msgs::msg::PoseStamped> current_pose_;
  std::optional<CachedPlan> cached_plan_;
  geometry_msgs::msg::PoseStamped pending_start_pose_;
  uint64_t plan_sequence_{0};
  uint64_t execute_sequence_{0};
  MoveGroupGoalHandle::SharedPtr planning_goal_handle_;
  ExecuteGoalHandle::SharedPtr execute_goal_handle_;

  std::string planning_group_;
  std::string end_effector_link_;
  std::string reference_frame_;
  std::string controller_name_;
  std::string current_pose_topic_;
  std::string move_group_action_;
  std::string execute_action_;
  std::string cartesian_path_service_name_;
  double velocity_scaling_;
  double acceleration_scaling_;
  double planning_time_;
  int planning_attempts_;
  double position_tolerance_;
  double orientation_tolerance_;
  double max_translation_;
  double max_rotation_;
  double max_current_pose_age_;
  double max_plan_age_;
  double max_start_translation_drift_;
  double max_start_rotation_drift_;
  std::vector<double> workspace_min_;
  std::vector<double> workspace_max_;
  std::vector<std::string> home_joint_names_;
  std::vector<double> home_joint_positions_;
  double home_joint_tolerance_;
  double cartesian_eef_step_;
  double cartesian_min_fraction_;
  double cartesian_revolute_jump_threshold_;
  double max_cartesian_translation_;
  double max_cartesian_rotation_;
  double max_cartesian_speed_;

  rclcpp_action::Client<MoveGroup>::SharedPtr move_group_client_;
  rclcpp_action::Client<ExecuteTrajectory>::SharedPtr execute_client_;
  rclcpp::Client<GetCartesianPath>::SharedPtr cartesian_path_client_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      linear_target_pose_subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_home_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr planning_ready_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr execute_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_service_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_trajectory_publisher_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  auto node = std::make_shared<PoseTargetNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  std::thread executor_thread([&executor]() { executor.spin(); });

  using namespace std::chrono_literals;
  while (rclcpp::ok() && shutdown_requested == 0) {
    std::this_thread::sleep_for(100ms);
  }

  node->requestSafeStop("process shutdown requested");
  const auto cancellation_deadline = std::chrono::steady_clock::now() + 2s;
  while (node->motionActive() && std::chrono::steady_clock::now() < cancellation_deadline) {
    std::this_thread::sleep_for(50ms);
  }
  if (node->motionActive()) {
    RCLCPP_ERROR(node->get_logger(),
                 "Cancellation was not acknowledged within 2 seconds; use the robot safety "
                 "controls if motion has not stopped.");
  }

  executor.cancel();
  if (executor_thread.joinable()) {
    executor_thread.join();
  }
  rclcpp::shutdown();
  return 0;
}
