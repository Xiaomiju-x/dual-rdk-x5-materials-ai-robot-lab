"""smtp_self_test.py — 验 EB_SMTP_* 凭据是否能登录 QQ SMTP 并发出邮件.

发件人和收件人都用 EB_SMTP_USER (自己发给自己), 不打扰任何人.
跑法: 让脚本通过 systemd EnvironmentFile 注入环境, 或:
    sudo bash -c 'set -a; . /etc/embodied_brain.env; set +a; python3 smtp_self_test.py'
"""
import os
import smtplib
import ssl
import sys
from email.mime.text import MIMEText
from email.utils import formatdate


def main() -> int:
    user = os.environ.get('EB_SMTP_USER', '').strip()
    pwd = os.environ.get('EB_SMTP_PASS', '').strip()
    host = os.environ.get('EB_SMTP_HOST', 'smtp.qq.com').strip()
    port = int(os.environ.get('EB_SMTP_PORT', '465'))

    if not (user and pwd):
        print('FAIL: EB_SMTP_USER or EB_SMTP_PASS empty')
        return 2

    print(f'host={host} port={port} user={user} pass=***({len(pwd)}chars)')

    msg = MIMEText(
        '具身脑车载脑 X5 (192.0.2.85) SMTP 通道自测.\n'
        '凭据加载成功, 发件 → 收件 → 服务器 OK.\n'
        '后续 alert_dispatcher 触发 CRITICAL 报警时会经此通道发到 EB_SMTP_TO.\n'
        '\n-- embodied_brain --',
        'plain', 'utf-8',
    )
    msg['Subject'] = '[embodied_brain] SMTP 通道自测就绪'
    msg['From'] = user
    msg['To'] = user
    msg['Date'] = formatdate(localtime=True)

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=10) as s:
            s.login(user, pwd)
            s.sendmail(user, [user], msg.as_string())
        print('OK: SMTP login + sendmail 成功. 收件箱看一眼.')
        return 0
    except smtplib.SMTPAuthenticationError as e:
        print(f'FAIL auth: {e} — 检查 EB_SMTP_PASS 是不是 QQ 邮箱授权码 (16字符, 非密码)')
        return 3
    except Exception as e:
        print(f'FAIL: {type(e).__name__}: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
