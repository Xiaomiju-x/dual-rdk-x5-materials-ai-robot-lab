"""alert_dispatcher — 收 /alarm, 按 channels 位掩码分发到 TTS / Email / WeChat / Log.

订阅:
    /alarm (my_robot_msgs/Alarm)

通道:
    CHAN_TTS    1 → POST AI 脑 dashboard:8888/api/say (跨网, Phase 4 完整接通; 暂时 fallback 本地 espeak)
    CHAN_EMAIL  2 → smtplib 发邮件给张丹 (复用 AI 脑 SMTP creds, 走环境变量)
    CHAN_WECHAT 4 → 企业微信 webhook (HTTP POST)
    CHAN_LOG    8 → 写入 /tmp/embodied_brain_alarms.log + ROS2 INFO

环境变量 (rdk@192.0.2.85 ~/.bashrc 配):
    EB_SMTP_HOST       默认 smtp.qq.com
    EB_SMTP_PORT       默认 465 (SSL)
    EB_SMTP_USER       发件邮箱, e.g. someone@qq.com
    EB_SMTP_PASS       授权码 (不是密码)
    EB_SMTP_TO         张丹老师邮箱, e.g. zhang_dan@todo.todo
    EB_WECHAT_WEBHOOK  企业微信群机器人 webhook URL
    EB_AI_BRAIN_URL    默认 http://192.0.2.103:8888 (AI 脑 dashboard)
    EB_TTS_ENDPOINT    默认 /api/say (AI 脑接口路径)

参数 (ros params):
    enable_tts (bool, 默认 true)
    enable_email (bool, 默认 true)
    enable_wechat (bool, 默认 true)
    enable_log (bool, 默认 true)
    log_file (str, 默认 /tmp/embodied_brain_alarms.log)
"""
from __future__ import annotations

import base64
import os
import smtplib
import threading
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import rclpy
from rclpy.node import Node

from my_robot_msgs.msg import Alarm

# 通道位掩码 (跟 furnace_monitor_agent 约定)
CHAN_TTS    = 1
CHAN_EMAIL  = 2
CHAN_WECHAT = 4
CHAN_LOG    = 8

LEVEL_NAMES = {1: 'INFO', 2: 'WARNING', 3: 'CRITICAL'}
SOURCE_NAMES = {
    1: 'TEMPERATURE_OUT_OF_RANGE',
    2: 'PV_SV_DEVIATION',
    3: 'POWER_LIGHT_OFF',
    4: 'FIRE_OR_SMOKE',
    99: 'OTHER',
}


class AlertDispatcher(Node):
    def __init__(self):
        super().__init__('alert_dispatcher')

        self.declare_parameter('enable_tts', True)
        self.declare_parameter('enable_email', True)
        self.declare_parameter('enable_wechat', True)
        self.declare_parameter('enable_log', True)
        self.declare_parameter('log_file', '/tmp/embodied_brain_alarms.log')

        self.enable_tts    = bool(self.get_parameter('enable_tts').value)
        self.enable_email  = bool(self.get_parameter('enable_email').value)
        self.enable_wechat = bool(self.get_parameter('enable_wechat').value)
        self.enable_log    = bool(self.get_parameter('enable_log').value)
        self.log_file      = self.get_parameter('log_file').value

        # 环境变量
        self.smtp_host = os.environ.get('EB_SMTP_HOST', 'smtp.qq.com')
        self.smtp_port = int(os.environ.get('EB_SMTP_PORT', '465'))
        self.smtp_user = os.environ.get('EB_SMTP_USER', '')
        self.smtp_pass = os.environ.get('EB_SMTP_PASS', '')
        self.smtp_to   = os.environ.get('EB_SMTP_TO', '')
        self.wechat_webhook = os.environ.get('EB_WECHAT_WEBHOOK', '')
        self.ai_brain_url   = os.environ.get('EB_AI_BRAIN_URL', 'http://192.0.2.103:8888')
        self.tts_endpoint   = os.environ.get('EB_TTS_ENDPOINT', '/api/say')

        # 缺配置时给 warning, 但不退出
        if self.enable_email and not (self.smtp_user and self.smtp_pass and self.smtp_to):
            self.get_logger().warn(
                'EMAIL enabled but EB_SMTP_USER/EB_SMTP_PASS/EB_SMTP_TO not set in env'
            )
        if self.enable_wechat and not self.wechat_webhook:
            self.get_logger().warn('WECHAT enabled but EB_WECHAT_WEBHOOK not set in env')

        # 订阅
        self.create_subscription(Alarm, '/alarm', self._on_alarm, 10)

        self.get_logger().info(
            f'alert_dispatcher started '
            f'tts={self.enable_tts} email={self.enable_email} '
            f'wechat={self.enable_wechat} log={self.enable_log}'
        )

    def _on_alarm(self, msg: Alarm):
        ch = msg.channels

        # 各通道 fire-and-forget 异步派发, 不要阻塞 ROS2 spin
        if (ch & CHAN_LOG) and self.enable_log:
            threading.Thread(target=self._log, args=(msg,), daemon=True).start()
        if (ch & CHAN_TTS) and self.enable_tts:
            threading.Thread(target=self._tts, args=(msg,), daemon=True).start()
        if (ch & CHAN_EMAIL) and self.enable_email:
            threading.Thread(target=self._email, args=(msg,), daemon=True).start()
        if (ch & CHAN_WECHAT) and self.enable_wechat:
            threading.Thread(target=self._wechat, args=(msg,), daemon=True).start()

    # ============ 各通道 ============

    def _log(self, msg: Alarm):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(self._format_line(msg) + '\n')
        except Exception as e:
            self.get_logger().error(f'log write failed: {e}')

    def _tts(self, msg: Alarm):
        text = self._format_voice(msg)

        # 优先调 AI 脑 (它有 M260C 麦克阵列 + TTS)
        try:
            import requests
            url = self.ai_brain_url + self.tts_endpoint
            r = requests.post(url, json={'text': text, 'priority': 'high'}, timeout=3)
            if r.status_code == 200:
                self.get_logger().info(f'TTS via AI brain OK: {text}')
                return
            self.get_logger().warn(f'TTS via AI brain failed status={r.status_code}, fallback espeak')
        except Exception as e:
            self.get_logger().warn(f'TTS via AI brain error {e}, fallback espeak')

        # Fallback: 本地 espeak (车载脑没接喇叭就静默)
        try:
            import subprocess
            subprocess.run(['espeak', '-v', 'zh', text],
                          timeout=10, check=False,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            self.get_logger().info(f'espeak not installed, TTS skipped: {text}')
        except Exception as e:
            self.get_logger().warn(f'espeak failed: {e}')

    def _email(self, msg: Alarm):
        if not (self.smtp_user and self.smtp_pass and self.smtp_to):
            return

        try:
            mime = MIMEMultipart()
            mime['From'] = self.smtp_user
            mime['To'] = self.smtp_to
            mime['Subject'] = f'[具身脑报警-{LEVEL_NAMES.get(msg.level, "?")}] {msg.title}'

            body = self._format_email_body(msg)
            mime.attach(MIMEText(body, 'plain', 'utf-8'))

            # 附件: snapshot
            if msg.snapshot_b64:
                try:
                    img_bytes = base64.b64decode(msg.snapshot_b64)
                    img = MIMEImage(img_bytes, _subtype='jpeg')
                    img.add_header('Content-Disposition', 'attachment',
                                   filename='furnace_snapshot.jpg')
                    mime.attach(img)
                except Exception as e:
                    self.get_logger().warn(f'snapshot attach failed: {e}')

            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=10) as s:
                s.login(self.smtp_user, self.smtp_pass)
                s.sendmail(self.smtp_user, [self.smtp_to], mime.as_string())

            self.get_logger().info(f'email sent to {self.smtp_to}')
        except Exception as e:
            self.get_logger().error(f'email failed: {e}')

    def _wechat(self, msg: Alarm):
        if not self.wechat_webhook:
            return
        try:
            import requests
            payload = {
                'msgtype': 'markdown',
                'markdown': {
                    'content': self._format_wechat_markdown(msg),
                },
            }
            r = requests.post(self.wechat_webhook, json=payload, timeout=5)
            if r.status_code == 200:
                self.get_logger().info('wechat webhook OK')
            else:
                self.get_logger().warn(f'wechat status={r.status_code} body={r.text[:200]}')
        except Exception as e:
            self.get_logger().error(f'wechat failed: {e}')

    # ============ 格式化 ============

    def _format_line(self, msg: Alarm) -> str:
        ts = datetime.fromtimestamp(
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ).strftime('%Y-%m-%d %H:%M:%S')
        return (f'[{ts}] [{LEVEL_NAMES.get(msg.level, "?")}] '
                f'[{SOURCE_NAMES.get(msg.source, str(msg.source))}] '
                f'{msg.title} :: {msg.description}')

    def _format_voice(self, msg: Alarm) -> str:
        # TTS 简短, 避免冗长
        return f'{LEVEL_NAMES.get(msg.level, "")}级报警: {msg.title}'

    def _format_email_body(self, msg: Alarm) -> str:
        ts = datetime.fromtimestamp(
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ).strftime('%Y-%m-%d %H:%M:%S')
        return (
            f'触发时间: {ts}\n'
            f'报警级别: {LEVEL_NAMES.get(msg.level, "?")}\n'
            f'报警来源: {SOURCE_NAMES.get(msg.source, str(msg.source))}\n'
            f'\n'
            f'标题: {msg.title}\n'
            f'\n'
            f'详细描述:\n{msg.description}\n'
            f'\n'
            f'上游 topic: {msg.ros2_topic}\n'
            f'\n'
            f'-- \n'
            f'此邮件由 embodied_brain 车载脑 (192.0.2.85) 自动发送.\n'
            f'附件为触发现场截图 (如果有).\n'
        )

    def _format_wechat_markdown(self, msg: Alarm) -> str:
        level = LEVEL_NAMES.get(msg.level, '?')
        color = {'INFO': 'info', 'WARNING': 'warning', 'CRITICAL': 'red'}.get(level, 'comment')
        ts = datetime.fromtimestamp(
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ).strftime('%Y-%m-%d %H:%M:%S')
        return (
            f'## <font color="{color}">具身脑报警 - {level}</font>\n'
            f'**{msg.title}**\n'
            f'\n'
            f'> {msg.description}\n'
            f'\n'
            f'- 来源: {SOURCE_NAMES.get(msg.source, str(msg.source))}\n'
            f'- 时间: {ts}\n'
            f'- 设备: 车载脑 RDK X5 (192.0.2.85)\n'
        )


def main():
    rclpy.init()
    node = AlertDispatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
