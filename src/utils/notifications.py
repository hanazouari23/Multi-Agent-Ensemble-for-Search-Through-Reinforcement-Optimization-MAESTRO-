"""Email notification helpers for the MAESTRO pipeline.

Configuration is read from environment variables (usually via ``.env.txt``):

- ``SMTP_USER``: sender email address (e.g. ``your_email@gmail.com``)
- ``SMTP_PASS``: SMTP / app password for the sender
- ``SMTP_HOST``: SMTP server host (default: ``smtp.gmail.com``)
- ``SMTP_PORT``: SMTP server port (default: ``587``)
- ``NOTIFICATION_EMAIL``: recipient address (default: ``hana.zouari23@gmail.com``)

For Gmail you must use an **App Password**; regular account passwords are no
longer accepted by Google's SMTP servers. Generate one at:
https://myaccount.google.com/apppasswords
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def send_email(subject: str, body: str, to_email: str | None = None) -> bool:
    """
    Send a plain-text email notification.

    If SMTP credentials are not configured, the call is logged and skipped
    instead of raising, so the pipeline keeps running.

    Parameters
    ----------
    subject : str
        Email subject line.
    body : str
        Plain-text email body.
    to_email : str, optional
        Override recipient. Defaults to ``NOTIFICATION_EMAIL`` env var or
        ``hana.zouari23@gmail.com``.

    Returns
    -------
    bool
        True if the email was sent, False if it was skipped or failed.
    """
    to_email = to_email or os.getenv("NOTIFICATION_EMAIL", "hana.zouari23@gmail.com")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    try:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        smtp_port = 587

    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")

    if not smtp_user or not smtp_pass:
        logger.warning(
            "SMTP_USER/SMTP_PASS not configured; skipping email: %s", subject
        )
        return False

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_email], msg.as_string())

        logger.info("Email sent to %s: %s", to_email, subject)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to send email notification: %s", exc)
        return False
