#!/usr/bin/env python3
"""
EchoFlow Lite - Lightweight Radar Target Tracking

A Pi-friendly ROS2 node that:
1. Subscribes to RadarSector messages
2. Applies intensity thresholding
3. Converts polar → cartesian coordinates
4. Accumulates points across N sectors (full rotation)
5. Clusters points with DBSCAN
6. Publishes centroids as PointCloud2

Usage:
    ros2 run echoflow_lite echoflow_lite_node
"""

import math
import struct
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from marine_sensor_msgs.msg import RadarSector

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class EchoFlowLiteNode(Node):
    """Lightweight radar target detection and tracking."""

    def __init__(self):
        super().__init__("echoflow_lite_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("input_topic",         "/aura/perception/sensors/halo_a/data")
        self.declare_parameter("output_topic",        "/aura/perception/sensors/halo_a/lite/targets")
        self.declare_parameter("output_frame",        "halo_a")
        self.declare_parameter("intensity_threshold", 0.15)   # 0.0–1.0
        self.declare_parameter("dbscan_eps",          500.0)  # metres
        self.declare_parameter("dbscan_min_samples",  2)
        self.declare_parameter("max_output_points",   50)
        self.declare_parameter("accumulate_sectors",  164)    # ~1 full rotation

        self.input_topic         = self.get_parameter("input_topic").value
        self.output_topic        = self.get_parameter("output_topic").value
        self.output_frame        = self.get_parameter("output_frame").value
        self.intensity_threshold = self.get_parameter("intensity_threshold").value
        self.dbscan_eps          = self.get_parameter("dbscan_eps").value
        self.dbscan_min_samples  = self.get_parameter("dbscan_min_samples").value
        self.max_output_points   = self.get_parameter("max_output_points").value
        self.accumulate_sectors  = self.get_parameter("accumulate_sectors").value

        # ── QoS ───────────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ── Pub/Sub ───────────────────────────────────────────────────────────
        self.radar_sub = self.create_subscription(
            RadarSector,
            self.input_topic,
            self._on_radar_sector,
            qos,
        )
        self.targets_pub = self.create_publisher(PointCloud2, self.output_topic, qos)

        # ── Accumulation buffer ───────────────────────────────────────────────
        self._point_buffer  = []   # list of Nx3 arrays
        self._sector_count  = 0
        self._last_header   = None  # header from most recent sector for timestamps

        # ── Stats ─────────────────────────────────────────────────────────────
        self.sectors_received  = 0
        self.total_points_in   = 0
        self.total_points_out  = 0
        self.total_clusters    = 0

        self.create_timer(5.0, self._log_stats)

        self.get_logger().info("=" * 60)
        self.get_logger().info("EchoFlow Lite")
        self.get_logger().info(f"  Input:               {self.input_topic}")
        self.get_logger().info(f"  Output:              {self.output_topic}")
        self.get_logger().info(f"  Intensity threshold: {self.intensity_threshold}")
        self.get_logger().info(f"  DBSCAN eps:          {self.dbscan_eps}m")
        self.get_logger().info(f"  DBSCAN min_samples:  {self.dbscan_min_samples}")
        self.get_logger().info(f"  Accumulate sectors:  {self.accumulate_sectors}")
        self.get_logger().info(f"  Max output points:   {self.max_output_points}")
        self.get_logger().info(f"  sklearn available:   {HAS_SKLEARN}")
        self.get_logger().info("=" * 60)

    # ── Callback ──────────────────────────────────────────────────────────────

    def _on_radar_sector(self, msg: RadarSector):
        """Accumulate sectors, then cluster and publish once per full rotation."""
        self.sectors_received += 1
        self._last_header = msg.header

        points = self._extract_points(msg)
        self.total_points_in += len(points)

        if len(points) > 0:
            self._point_buffer.append(points)

        self._sector_count += 1

        # Wait until we have a full rotation's worth of sectors
        if self._sector_count < self.accumulate_sectors:
            return

        # ── Flush buffer ──────────────────────────────────────────────────────
        self._sector_count = 0

        if not self._point_buffer:
            return

        all_points = np.vstack(self._point_buffer)  # Nx3 [x, y, intensity]
        self._point_buffer = []

        self.get_logger().debug(
            f"Buffer flush: {len(all_points)} pts from {self.accumulate_sectors} sectors"
        )

        # ── Cluster ───────────────────────────────────────────────────────────
        if HAS_SKLEARN and len(all_points) >= self.dbscan_min_samples:
            centroids = self._cluster_dbscan(all_points)
        else:
            # Fallback: use all accumulated points (no sklearn)
            centroids = all_points

        # Cap output size — keep highest intensity
        if len(centroids) > self.max_output_points:
            centroids = centroids[
                np.argsort(centroids[:, 2])[-self.max_output_points:]
            ]

        self.total_points_out += len(centroids)

        cloud_msg = self._create_pointcloud2(centroids, self._last_header)
        self.targets_pub.publish(cloud_msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_points(self, msg: RadarSector) -> np.ndarray:
        """Extract above-threshold points. Returns Nx3 [x, y, intensity]."""
        if not msg.intensities:
            return np.empty((0, 3), dtype=np.float32)

        num_range_cells = len(msg.intensities[0].echoes) if msg.intensities else 0
        if num_range_cells == 0:
            return np.empty((0, 3), dtype=np.float32)

        range_resolution = msg.range_max / num_range_cells

        points = []
        for spoke_idx, echo in enumerate(msg.intensities):
            angle = msg.angle_start + spoke_idx * msg.angle_increment
            for range_idx, intensity in enumerate(echo.echoes):
                if intensity >= self.intensity_threshold:
                    r   = (range_idx + 0.5) * range_resolution
                    x   = r * math.cos(angle)
                    y   = r * math.sin(angle)
                    points.append([x, y, float(intensity)])

        if not points:
            return np.empty((0, 3), dtype=np.float32)

        return np.array(points, dtype=np.float32)

    def _cluster_dbscan(self, points: np.ndarray) -> np.ndarray:
        """Cluster on x,y; return centroids Nx3 [x, y, max_intensity]."""
        xy = points[:, :2]
        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples
        ).fit_predict(xy)

        unique_labels = set(labels) - {-1}
        noise_count   = int(np.sum(labels == -1))

        self.get_logger().info(
            f"DBSCAN: {len(points)} pts → {len(unique_labels)} clusters, "
            f"{noise_count} noise (eps={self.dbscan_eps}m)"
        )

        self.total_clusters += len(unique_labels)

        if not unique_labels:
            # Everything is noise — return raw points so Foxglove still shows something
            return points[:min(len(points), self.max_output_points)]

        centroids = []
        for label in unique_labels:
            mask    = labels == label
            cluster = points[mask]
            centroids.append([
                float(np.mean(cluster[:, 0])),
                float(np.mean(cluster[:, 1])),
                float(np.max(cluster[:, 2])),
            ])

        return np.array(centroids, dtype=np.float32)

    def _create_pointcloud2(self, points: np.ndarray, header: Header) -> PointCloud2:
        """Pack Nx3 points into a PointCloud2 message (x, y, z=0, intensity)."""
        msg              = PointCloud2()
        msg.header       = Header()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.output_frame

        msg.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step   = 16
        msg.height       = 1
        msg.width        = len(points)
        msg.row_step     = msg.point_step * msg.width
        msg.is_dense     = True

        data = bytearray(msg.row_step)
        for i, pt in enumerate(points):
            struct.pack_into('<ffff', data, i * msg.point_step,
                             pt[0], pt[1], 0.0, pt[2])
        msg.data = bytes(data)
        return msg

    def _log_stats(self):
        if self.sectors_received > 0:
            self.get_logger().info(
                f"Stats: {self.sectors_received} sectors, "
                f"{self.total_points_in} pts in, "
                f"{self.total_points_out} pts out, "
                f"{self.total_clusters} clusters"
            )
        self.sectors_received = 0
        self.total_points_in  = 0
        self.total_points_out = 0
        self.total_clusters   = 0


def main(args=None):
    rclpy.init(args=args)
    node = EchoFlowLiteNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
