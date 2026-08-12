"""command_interpreter_node — ROS2 service /interpret_command.

把 command_interpreter 三档抽象层包成 ROS2 service.
其他节点 (例如 ai_brain_bridge 收到自然语言指令时, 或语音 ASR 后) 调它得到结构化 task.

Service:
    /interpret_command (my_robot_msgs/InterpretCommand)
        request:  utterance, context_json
        response: success, reason, backend_used, task_id, task_type, bottle_id,
                  from_location, to_location, priority, timeout_s, raw_response

参数 (ros params):
    backend (str): 'rule' (默认) | 'local' | 'remote'
    local_endpoint (str): 本地 llama-server 地址, 默认 http://127.0.0.1:9100/v1/chat/completions
    local_model (str): llama 模型名, 默认 'qwen3-0.6b'

启动:
    ros2 run my_robot_agents command_interpreter
    # 切到远程 LLM (AI 脑)
    ros2 run my_robot_agents command_interpreter --ros-args -p backend:=remote
"""
from __future__ import annotations

import json

import rclpy
from rclpy.node import Node

from my_robot_msgs.srv import InterpretCommand

from .command_interpreter import (
    make_interpreter,
    CommandInterpreter,
    LocalLLMInterpreter,
    RemoteLLMInterpreter,
)


class CommandInterpreterNode(Node):
    def __init__(self):
        super().__init__('command_interpreter')

        self.declare_parameter('backend', 'rule')
        self.declare_parameter('local_endpoint',
                               'http://127.0.0.1:9100/v1/chat/completions')
        self.declare_parameter('local_model', 'qwen3-0.6b')

        backend = self.get_parameter('backend').value
        kwargs = {}
        if backend == 'local':
            kwargs['endpoint'] = self.get_parameter('local_endpoint').value
            kwargs['model'] = self.get_parameter('local_model').value

        try:
            self._interpreter: CommandInterpreter = make_interpreter(backend, **kwargs)
        except ValueError as e:
            self.get_logger().error(f'invalid backend: {e}')
            raise

        self.create_service(
            InterpretCommand,
            'interpret_command',
            self._on_request,
        )

        self.get_logger().info(
            f'command_interpreter ready, backend={backend} '
            f'(切换: ros2 param set /command_interpreter backend rule/local/remote)'
        )

    def _on_request(self, request, response):
        ctx = None
        if request.context_json:
            try:
                ctx = json.loads(request.context_json)
            except Exception:
                pass

        result = self._interpreter.parse(request.utterance, ctx)

        response.success = result.success
        response.reason = result.reason
        response.backend_used = result.backend_used
        response.task_id = result.task_id
        response.task_type = result.task_type
        response.bottle_id = result.bottle_id
        response.from_location = result.from_location
        response.to_location = result.to_location
        response.priority = result.priority
        response.timeout_s = result.timeout_s
        response.raw_response = result.raw_response

        self.get_logger().info(
            f'[{result.backend_used}] "{request.utterance}" → '
            f'success={result.success} type={result.task_type} '
            f'bottle={result.bottle_id} from={result.from_location} to={result.to_location}'
        )
        return response


def main():
    rclpy.init()
    node = CommandInterpreterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
