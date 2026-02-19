"""
KAHU Stack Launch File
======================
Starts the full KAHU perception stack:
  - Static transform: map → halo_a
  - EchoFlow Lite node (radar target detection)
  - Foxglove bridge (visualization)

Usage:
    source /opt/ros/jazzy/setup.bash
    source ~/ros2_ws/install/setup.bash
    ros2 launch kahu_stack.launch.py

Optional overrides:
    ros2 launch kahu_stack.launch.py intensity_threshold:=0.8 dbscan_eps:=200.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Launch Arguments ────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            "intensity_threshold",
            default_value="0.1",
            description="Minimum intensity to keep a radar return (0.0–1.0)",
        ),
        DeclareLaunchArgument(
            "dbscan_eps",
            default_value="500.0",
            description="DBSCAN cluster radius in metres",
        ),
        DeclareLaunchArgument(
            "dbscan_min_samples",
            default_value="2",
            description="Minimum points to form a DBSCAN cluster",
        ),
        DeclareLaunchArgument(
            "max_output_points",
            default_value="50",
            description="Maximum target points published per cycle",
        ),
        DeclareLaunchArgument(
            "accumulate_sectors",
            default_value="164",
            description="Number of sectors to accumulate before clustering (~1 full rotation)",
        ),
        DeclareLaunchArgument(
            "foxglove_port",
            default_value="8765",
            description="WebSocket port for Foxglove bridge",
        ),
    ]

    # ── Nodes ────────────────────────────────────────────────────────────────

    # 1. Static transform: map → halo_a
    #    Publishes at 10 Hz so Foxglove always has a valid TF tree.
    #    Without this, PointCloud2 messages with frame_id="halo_a" won't
    #    render in Foxglove when Fixed Frame is set to "map".
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_halo_a",
        # New-style args: x y z yaw pitch roll parent_frame child_frame
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--yaw", "0", "--pitch", "0", "--roll", "0",
                   "--frame-id", "map",
                   "--child-frame-id", "halo_a"],
        output="screen",
    )

    # 2. EchoFlow Lite — radar target detection
    echoflow_lite = Node(
        package="echoflow_lite",
        executable="echoflow_lite_node",
        name="echoflow_lite_node",
        parameters=[{
            "input_topic":          "/aura/perception/sensors/halo_a/data",
            "output_topic":         "/aura/perception/sensors/halo_a/lite/targets",
            "output_frame":         "halo_a",
            "intensity_threshold":  LaunchConfiguration("intensity_threshold"),
            "dbscan_eps":           LaunchConfiguration("dbscan_eps"),
            "dbscan_min_samples":   LaunchConfiguration("dbscan_min_samples"),
            "max_output_points":    LaunchConfiguration("max_output_points"),
            "accumulate_sectors":   LaunchConfiguration("accumulate_sectors"),
        }],
        output="screen",
    )

    # 3. Foxglove bridge — WebSocket visualization
    foxglove_bridge = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        parameters=[{
            "port": LaunchConfiguration("foxglove_port"),
        }],
        output="screen",
    )

    return LaunchDescription(args + [static_tf, echoflow_lite, foxglove_bridge])
