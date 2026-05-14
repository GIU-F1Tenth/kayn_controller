"""
KAYN Controller : ROS2 Node

Subscribes:
    /odom                                  nav_msgs/Odometry
    /horizon_mapper/reference_trajectory   giu_f1t_interfaces/VehicleStateArray
    /horizon_mapper/path_ready             std_msgs/Bool

Publishes:
    /kayn/drive             ackermann_msgs/AckermannDriveStamped
    /kayn/mode              std_msgs/String
    /kayn/cross_track_error std_msgs/Float32
    /kayn/curvature         std_msgs/Float32
    /kayn/diagnostics       diagnostic_msgs/DiagnosticArray
"""
