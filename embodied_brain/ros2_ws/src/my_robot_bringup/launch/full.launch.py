"""full.launch.py — 一键拉起整个具身脑.

启动顺序 (各子 launch 内部已配好, 不同步等待):
    1. URDF + robot_state_publisher           ← my_robot_description
    2. fake_odom (临时, 等 STM32 烧固件后改)   ← my_robot_agents
    3. 4 路传感器驱动 (LD14 + Astra + USB cam + serial F407)
                                              ← my_robot_drivers
    4. depth_to_laserscan + slam_toolbox      ← my_robot_navigation
    5. (Phase 9 默认不跑 Nav2, 等真车阶段再开)
    6. 烧结炉 OCR + monitor + alert_dispatcher ← my_robot_agents
    7. dispatch_server + ai_brain_bridge + telemetry  ← my_robot_bridge

参数:
    use_fake_odom (bool):  默认 true (没 STM32 时); Phase 6 改 false 走真 serial_F407
    use_serial_f407 (bool): 默认 false (没烧固件), 与 use_fake_odom 互斥
    use_slam (bool):       默认 true
    use_nav2 (bool):       默认 false (Phase 9 不开 Nav2, 等真车闭环)
    map (str):             use_nav2=true 且 use_slam=false 时加载的地图 YAML
    stub_mode (bool):      默认 true (dispatch_server 安全 stub; 真闭环设 false)
    use_lab_fsd_shadow (bool): 默认 true (常驻 shadow planner, 不接管 cmd_vel)
    allow_origin_placeholder_targets (bool): 默认 false (真 Nav2 模式拒绝未标定 0,0,0 目标)
    use_furnace (bool):    默认 false (没接云台之前 OCR 没图像源, 只在测试时打开)
    use_bridge (bool):     默认 true (跨网通信常驻)
    ai_brain_url (str):    AI 脑 dashboard 地址
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_desc  = FindPackageShare('my_robot_description')
    pkg_drv   = FindPackageShare('my_robot_drivers')
    pkg_nav   = FindPackageShare('my_robot_navigation')
    pkg_ag    = FindPackageShare('my_robot_agents')
    pkg_brdg  = FindPackageShare('my_robot_bridge')
    pkg_perc  = FindPackageShare('my_robot_perception')

    declare_use_fake_odom = DeclareLaunchArgument(
        'use_fake_odom', default_value=os.environ.get('EB_USE_FAKE_ODOM', 'true'),
        description='STM32 烧固件后 /etc/embodied_brain.env 设 EB_USE_FAKE_ODOM=false'
    )
    declare_use_serial = DeclareLaunchArgument(
        'use_serial_f407', default_value=os.environ.get('EB_USE_SERIAL_F407', 'false'),
        description='STM32F407 烧好后 env 设 EB_USE_SERIAL_F407=true; 跟 fake_odom 互斥'
    )
    declare_use_lidar = DeclareLaunchArgument(
        'use_lidar', default_value=os.environ.get('EB_USE_LIDAR', 'true'))
    declare_use_depth = DeclareLaunchArgument(
        'use_depth_camera', default_value=os.environ.get('EB_USE_DEPTH_CAM', 'true'),
        description='Astra Pro depth+rgb. 跟 K3 USB cam 抢带宽时可设 false (EB_USE_DEPTH_CAM=false)')
    declare_use_lift_cam = DeclareLaunchArgument('use_lift_camera', default_value='false',
        description='Phase 6 取料时打开')
    pt_camera_source_default = os.environ.get('EB_PT_CAMERA_SOURCE', '/dev/PT_CAM')
    pt_camera_requested = os.environ.get('EB_USE_PT_CAMERA', 'false').strip().lower() in {
        '1', 'true', 'yes', 'on'
    }
    pt_camera_source_available = (
        '://' in pt_camera_source_default
        or pt_camera_source_default.isdigit()
        or os.path.exists(pt_camera_source_default)
    )
    pt_camera_default = 'true' if pt_camera_requested and pt_camera_source_available else 'false'
    declare_use_pt_cam = DeclareLaunchArgument('use_pt_camera',
        default_value=pt_camera_default,
        description='Phase 8 K3 USB cam 接好后打开 (替代米家云台). 也可 /etc/embodied_brain.env 设 EB_USE_PT_CAMERA=true')
    declare_pt_cam_source = DeclareLaunchArgument('pt_camera_source',
        default_value=pt_camera_source_default,
        description='K3 USB cam 设备路径')
    declare_use_slam = DeclareLaunchArgument('use_slam', default_value='true')
    declare_use_nav2 = DeclareLaunchArgument(
        'use_nav2',
        default_value=os.environ.get('EB_USE_NAV2', 'false'),
        description='Nav2 navigation; finals wrapper enables it through EB_USE_NAV2=true')
    declare_use_state_estimator = DeclareLaunchArgument(
        'use_state_estimator',
        default_value=os.environ.get('EB_USE_STATE_ESTIMATOR', 'true'),
        description='Fuse /wheel_odom and validity-gated /imu into authoritative /odom')
    declare_map = DeclareLaunchArgument(
        'map', default_value=os.environ.get('EB_MAP_YAML', ''),
        description='保存地图 YAML；use_nav2=true 且 use_slam=false 时必须提供')
    declare_use_collision_monitor = DeclareLaunchArgument(
        'use_collision_monitor',
        default_value=os.environ.get('EB_USE_COLLISION_MONITOR', 'false'),
        description='Nav2 官方 Collision Monitor: /cmd_vel -> software safety -> /cmd_vel_safe')
    declare_stub_mode = DeclareLaunchArgument(
        'stub_mode',
        default_value=os.environ.get('EB_DISPATCH_STUB_MODE', 'true'),
        description='dispatch_server 是否保持安全 stub; 真 Nav2 dispatch 设 false'
    )
    declare_use_lab_fsd_shadow = DeclareLaunchArgument(
        'use_lab_fsd_shadow',
        default_value=os.environ.get('EB_USE_LAB_FSD_SHADOW', 'true'),
        description='启动 Lab-FSD shadow planner/vision bridge, 只观察/辅助, 不接管 /cmd_vel'
    )
    declare_allow_origin = DeclareLaunchArgument(
        'allow_origin_placeholder_targets',
        default_value=os.environ.get('EB_ALLOW_ORIGIN_PLACEHOLDER_TARGETS', 'false'),
        description='真 Nav2 dispatch 是否允许非 home 点位仍为 0,0,0 占位'
    )
    declare_execute_pickup_actuators = DeclareLaunchArgument(
        'execute_pickup_actuators',
        default_value=os.environ.get('EB_EXECUTE_PICKUP_ACTUATORS', 'false'),
        description='实物 pickup_flow 是否调用 F407 升降台/电磁铁服务; 未标定时保持 false'
    )
    declare_pickup_height = DeclareLaunchArgument(
        'pickup_height_m', default_value=os.environ.get('EB_PICKUP_HEIGHT_M', '0.05'))
    declare_transport_height = DeclareLaunchArgument(
        'transport_height_m', default_value=os.environ.get('EB_TRANSPORT_HEIGHT_M', '0.08'))
    declare_place_height = DeclareLaunchArgument(
        'place_height_m', default_value=os.environ.get('EB_PLACE_HEIGHT_M', '0.05'))
    declare_use_physical_evidence_gate = DeclareLaunchArgument(
        'use_physical_evidence_gate',
        default_value=os.environ.get('EB_USE_PHYSICAL_EVIDENCE_GATE', 'false'),
        description='启动独立物理证据门；无真实传感器适配器时保持 false')
    declare_physical_evidence_mode = DeclareLaunchArgument(
        'physical_evidence_mode',
        default_value=os.environ.get('EB_PHYSICAL_EVIDENCE_MODE', 'disabled'),
        description='disabled|report_only|required；生产默认 disabled')
    declare_use_d2l = DeclareLaunchArgument('use_depth_scan', default_value='true')
    declare_use_apriltag = DeclareLaunchArgument(
        'use_apriltag', default_value=os.environ.get('EB_USE_APRILTAG', 'false'),
        description='AprilTag CPU 检测 (CHRISTIAN RAUCH 包). 贴 tag 后开. 可由 EB_USE_APRILTAG=true 控')
    declare_use_yolo_world = DeclareLaunchArgument(
        'use_yolo_world', default_value=os.environ.get('EB_USE_YOLO_WORLD', 'true'),
        description='BPU YOLO-World 开放词检测 (Round 4 Day 1). 默认开, 订 /pt_camera/image_raw 发 /hobot_yolo_world')
    declare_use_edgesam = DeclareLaunchArgument(
        'use_edgesam', default_value=os.environ.get('EB_USE_EDGESAM', 'false'),
        description='BPU EdgeSAM 像素级分割 (Round 4 Day 2-3). 默认关, 串 yolo_world 出框 → 出 mask, 占 BPU ~30%. EB_USE_EDGESAM=true 开')
    declare_use_xfeat = DeclareLaunchArgument(
        'use_xfeat', default_value=os.environ.get('EB_USE_XFEAT', 'false'),
        description='BPU XFeat 视觉特征 (Round 4 D1 Phase 3). 985KB/17ms/57FPS. EB_USE_XFEAT=true 开')
    declare_use_mppi = DeclareLaunchArgument(
        'use_mppi', default_value=os.environ.get('EB_USE_MPPI', 'false'),
        description='BPU MPPI cost MLP. 默认只发 /mppi/cmd_vel_proposed, 不接管 /cmd_vel')
    declare_mppi_direct = DeclareLaunchArgument(
        'mppi_publish_direct_cmd_vel',
        default_value=os.environ.get('EB_MPPI_PUBLISH_DIRECT_CMD_VEL', 'false'),
        description='MPPI 是否允许直接发布 /cmd_vel. 真车默认 false, 需安全评估后显式打开')
    declare_mppi_proposed_topic = DeclareLaunchArgument(
        'mppi_cmd_vel_topic',
        default_value=os.environ.get('EB_MPPI_CMD_VEL_TOPIC', '/mppi/cmd_vel_proposed'),
        description='MPPI proposed velocity topic')
    declare_use_bottle_ocr = DeclareLaunchArgument(
        'use_bottle_ocr', default_value=os.environ.get('EB_USE_BOTTLE_OCR', 'false'),
        description='BPU PP-OCRv4 det 6ms + PaddleOCR rec CPU (Round 4 A4 Phase 4). 订 /lift_camera/image_raw. EB_USE_BOTTLE_OCR=true 开')
    declare_use_audio = DeclareLaunchArgument(
        'use_audio', default_value=os.environ.get('EB_USE_AUDIO', 'false'),
        description='M260C 麦阵: RNNoise 降噪 + ODAS DOA 声源定位 (Round 4 E1/E2/E3 Phase 4). EB_USE_AUDIO=true 开')
    declare_alsa_device = DeclareLaunchArgument(
        'alsa_device', default_value=os.environ.get('EB_ALSA_DEVICE', 'hw:2,0'),
        description='M260C ALSA 设备号. 若重启后变了, 用 aplay -l 查 XFMDPV/XFM-DP 所在 card. EB_ALSA_DEVICE=hw:X,0')
    declare_use_furnace = DeclareLaunchArgument('use_furnace', default_value='false',
        description='Phase 7 云台拉流后打开')
    declare_use_voice = DeclareLaunchArgument(
        'use_voice', default_value=os.environ.get('EB_USE_VOICE', 'false'),
        description='本地中文 ASR (SenseVoice) + TTS (Piper) on M260C 麦阵 (Round 4 Day 4 B1+B2). EB_USE_VOICE=true 开')
    declare_use_bridge = DeclareLaunchArgument('use_bridge', default_value='true')
    declare_ai_brain = DeclareLaunchArgument(
        'ai_brain_url',
        default_value=os.environ.get('EB_AI_BRAIN_URL', 'http://192.0.2.103:8888')
    )

    # ===================== 1. URDF + RSP =====================
    # 不用 display.launch.py (它含 rviz2, X5 server 没装).
    # 直接启 robot_state_publisher + joint_state_publisher.
    urdf_path = PathJoinSubstitution([pkg_desc, 'urdf', 'my_robot.urdf.xacro'])
    robot_description = ParameterValue(
        Command(['xacro ', urdf_path]),
        value_type=str,
    )
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'publish_frequency': 30.0,
        }],
    )
    jsp = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
    )

    # Final actuator input is derived from the monitor switch, not from a
    # second user-controlled flag.  This prevents a launch-time safety bypass.
    actuator_cmd_vel_topic = PythonExpression([
        "'/cmd_vel_safe' if '", LaunchConfiguration('use_collision_monitor'),
        "'.lower() in ('1', 'true', 'yes', 'on') else '/cmd_vel'",
    ])
    f407_odom_topic = PythonExpression([
        "'/wheel_odom' if '", LaunchConfiguration('use_state_estimator'),
        "'.lower() in ('1', 'true', 'yes', 'on') else '/odom'",
    ])
    f407_publish_tf = PythonExpression([
        "'false' if '", LaunchConfiguration('use_state_estimator'),
        "'.lower() in ('1', 'true', 'yes', 'on') else 'true'",
    ])

    # ===================== 2. fake_odom (临时) =====================
    fake_odom = Node(
        package='my_robot_agents',
        executable='fake_odom',
        name='fake_odom',
        output='screen',
        remappings=[('cmd_vel', actuator_cmd_vel_topic)],
        condition=IfCondition(LaunchConfiguration('use_fake_odom')),
    )

    # ===================== 3. 传感器驱动 =====================
    sensors = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_drv, 'launch', 'sensors.launch.py'])
        ),
        launch_arguments={
            'use_lidar': LaunchConfiguration('use_lidar'),
            'use_depth_camera': LaunchConfiguration('use_depth_camera'),
            'use_lift_camera': LaunchConfiguration('use_lift_camera'),
            'use_pt_camera': LaunchConfiguration('use_pt_camera'),
            'pt_camera_source': LaunchConfiguration('pt_camera_source'),
            'use_serial_f407': LaunchConfiguration('use_serial_f407'),
            'cmd_vel_topic': actuator_cmd_vel_topic,
            'f407_odom_topic': f407_odom_topic,
            'f407_publish_tf': f407_publish_tf,
        }.items(),
    )

    state_estimator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'state_estimator.launch.py'])
        ),
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('use_state_estimator'), "'.lower() in ('1','true','yes','on') and '",
            LaunchConfiguration('use_serial_f407'), "'.lower() in ('1','true','yes','on')",
        ])),
    )

    # ===================== 4. SLAM + Nav2 =====================
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'slam.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('use_slam')),
    )
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'localization.launch.py'])
        ),
        launch_arguments={
            'map': LaunchConfiguration('map'),
        }.items(),
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('use_nav2'), "' == 'true' and '",
            LaunchConfiguration('use_slam'), "' == 'false'",
        ])),
    )
    d2l = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'depth_to_laserscan.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('use_depth_scan')),
    )
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments={
            'use_collision_monitor': LaunchConfiguration('use_collision_monitor'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_nav2')),
    )
    lab_fsd_shadow = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'lab_fsd_shadow.launch.py'])
        ),
        launch_arguments={
            'ai_brain_url': LaunchConfiguration('ai_brain_url'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_lab_fsd_shadow')),
    )
    apriltag = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'apriltag.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('use_apriltag')),
    )

    # ===== Round 4 Day 1: BPU YOLO-World 开放词检测 =====
    yolo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_perc, 'launch', 'yolo_world.launch.py'])
        ),
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('use_yolo_world'), "'.lower() in ['true','1','yes'] and '",
            LaunchConfiguration('use_pt_camera'), "'.lower() in ['true','1','yes']",
        ])),
    )

    # ===== Round 4 Day 2-3: BPU EdgeSAM 像素级分割 (串 yolo_world) =====
    edgesam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_perc, 'launch', 'edgesam.launch.py'])
        ),
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('use_edgesam'), "'.lower() in ['true','1','yes'] and '",
            LaunchConfiguration('use_yolo_world'), "'.lower() in ['true','1','yes'] and '",
            LaunchConfiguration('use_pt_camera'), "'.lower() in ['true','1','yes']",
        ])),
    )

    # ===== Round 4 Phase 3 D1: BPU XFeat 视觉特征 (VIO 基础) =====
    xfeat = Node(
        package='my_robot_agents',
        executable='xfeat',
        name='xfeat_node',
        output='screen',
        parameters=[{'image_topic': '/lift_camera/image_raw'}],
        condition=IfCondition(LaunchConfiguration('use_xfeat')),
    )

    # ===== Round 4 Phase 3 C2: BPU MPPI cost MLP =====
    mppi = Node(
        package='my_robot_agents',
        executable='mppi',
        name='mppi_node',
        output='screen',
        parameters=[{
            'cmd_vel_topic': LaunchConfiguration('mppi_cmd_vel_topic'),
            'publish_direct_cmd_vel': ParameterValue(
                LaunchConfiguration('mppi_publish_direct_cmd_vel'),
                value_type=bool,
            ),
        }],
        condition=IfCondition(LaunchConfiguration('use_mppi')),
    )

    # ===== Round 4 Phase 4 A4: BPU PP-OCRv4 试剂瓶标签 OCR =====
    bottle_ocr = Node(
        package='my_robot_agents',
        executable='bottle_ocr_bpu',
        name='bottle_ocr_bpu_node',
        output='screen',
        parameters=[{'use_bpu': True, 'use_rec': True,
                     'image_topic': '/lift_camera/image_raw'}],
        condition=IfCondition(LaunchConfiguration('use_bottle_ocr')),
    )

    # ===== Round 4 Phase 4 E1/E2/E3: M260C 麦阵 RNNoise + ODAS DOA =====
    audio_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_ag, 'launch', 'audio_pipeline.launch.py'])
        ),
        launch_arguments={'alsa_device': LaunchConfiguration('alsa_device')}.items(),
        condition=IfCondition(LaunchConfiguration('use_audio')),
    )

    # ===================== 5. 烧结炉 OCR Agent =====================
    furnace = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_ag, 'launch', 'furnace_monitor.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('use_furnace')),
    )

    # ===== Round 4 Day 4: 本地中文语音 (SenseVoice ASR + Piper TTS) =====
    voice_in = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_ag, 'launch', 'voice_input.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('use_voice')),
    )
    voice_out = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_ag, 'launch', 'voice_output.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('use_voice')),
    )

    # ===================== 6. 跨网桥 =====================
    bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_brdg, 'launch', 'bridge.launch.py'])
        ),
        launch_arguments={
            'ai_brain_url': LaunchConfiguration('ai_brain_url'),
            'stub_mode': LaunchConfiguration('stub_mode'),
            'use_nav2': LaunchConfiguration('use_nav2'),
            'allow_origin_placeholder_targets': LaunchConfiguration('allow_origin_placeholder_targets'),
            'execute_pickup_actuators': LaunchConfiguration('execute_pickup_actuators'),
            'pickup_height_m': LaunchConfiguration('pickup_height_m'),
            'transport_height_m': LaunchConfiguration('transport_height_m'),
            'place_height_m': LaunchConfiguration('place_height_m'),
            'use_physical_evidence_gate': LaunchConfiguration('use_physical_evidence_gate'),
            'physical_evidence_mode': LaunchConfiguration('physical_evidence_mode'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_bridge')),
    )

    # ===================== 7. command_interpreter (常驻 service) =====================
    cmd_interp = Node(
        package='my_robot_agents',
        executable='command_interpreter',
        name='command_interpreter',
        output='screen',
        parameters=[{'backend': 'rule'}],  # 默认 rule, 后期可改 remote
    )

    # ===================== 8. location_visualizer (rviz Marker) =====================
    location_viz = Node(
        package='my_robot_agents',
        executable='location_visualizer',
        name='location_visualizer',
        output='screen',
    )

    return LaunchDescription([
        declare_use_fake_odom, declare_use_serial,
        declare_use_lidar, declare_use_depth, declare_use_lift_cam,
        declare_use_pt_cam, declare_pt_cam_source,
        declare_use_slam, declare_use_nav2, declare_use_state_estimator, declare_map,
        declare_use_collision_monitor, declare_stub_mode,
        declare_use_lab_fsd_shadow, declare_allow_origin,
        declare_execute_pickup_actuators, declare_pickup_height,
        declare_transport_height, declare_place_height,
        declare_use_physical_evidence_gate, declare_physical_evidence_mode,
        declare_use_d2l,
        declare_use_apriltag,
        declare_use_yolo_world,
        declare_use_edgesam,
        declare_use_xfeat,
        declare_use_mppi, declare_mppi_direct, declare_mppi_proposed_topic,
        declare_use_bottle_ocr,
        declare_use_audio, declare_alsa_device,
        declare_use_furnace, declare_use_voice, declare_use_bridge, declare_ai_brain,
        GroupAction([
            rsp, jsp,
            fake_odom,
            sensors,
            state_estimator,
            slam, localization, d2l, nav2, lab_fsd_shadow,
            apriltag,
            yolo_world,
            edgesam,
            xfeat,
            mppi,
            bottle_ocr,
            audio_pipeline,
            furnace,
            voice_in, voice_out,
            bridge,
            cmd_interp,
            location_viz,
        ]),
    ])
