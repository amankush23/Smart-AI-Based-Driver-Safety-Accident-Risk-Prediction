"""
OTP Service — Generates, stores, verifies, and emails 6-digit OTPs.

If SMTP is not configured, the OTP is printed to the console so developers
can test the flow locally without requiring a real email service.
"""

import random
import smtplib
import string
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import (
    OTP_EXPIRY_MINUTES,
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASS,
    SMTP_PORT,
    SMTP_USER,
)
from database.mongo import (
    create_otp_request,
    delete_otp_request,
    get_otp_request,
)
from utils.logger import get_logger

logger = get_logger("services.otp")


def _generate_otp(length: int = 6) -> str:
    """Generate a random numeric OTP string."""
    return "".join(random.choices(string.digits, k=length))


def _send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an email via SMTP. Returns True on success."""
    if not SMTP_HOST or not SMTP_USER:
        # Development fallback — log OTP to console
        logger.warning("SMTP not configured. OTP will be logged to console only.")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())

        logger.info(f"OTP email sent to {to_email}")
        return True
    except Exception as exc:
        logger.error(f"SMTP error sending OTP email: {exc}")
        return False


def _otp_email_body(otp: str) -> str:
    return f"""
    <html>
    <body style="font-family:Poppins,sans-serif;background:#0f0f1a;color:#fff;padding:40px;">
        <div style="max-width:480px;margin:0 auto;background:#151528;border-radius:16px;
                    padding:40px;border:1px solid rgba(0,229,255,0.2);">
            <h2 style="color:#00e5ff;margin-bottom:8px;">Driver Safety System</h2>
            <p style="color:#a0a0b8;margin-bottom:28px;">Password Reset Request</p>
            <p style="color:#fff;margin-bottom:16px;">
                Your one-time password (OTP) for resetting your account password:
            </p>
            <div style="background:#0f0f1a;border-radius:12px;padding:24px;
                        text-align:center;margin-bottom:28px;
                        border:1px solid rgba(0,229,255,0.3);">
                <span style="font-size:2.4rem;font-weight:800;letter-spacing:12px;
                             color:#00e5ff;">{otp}</span>
            </div>
            <p style="color:#a0a0b8;font-size:0.85rem;">
                This OTP expires in <strong>{OTP_EXPIRY_MINUTES} minutes</strong>.
                If you did not request this, ignore this email.
            </p>
        </div>
    </body>
    </html>
    """


def request_otp(email: str) -> dict:
    """
    Generate and store an OTP for the given email.
    Sends an email if SMTP is configured; otherwise logs to console.

    Returns:
        dict with 'sent' (bool) and 'otp' (str, only when SMTP not configured)
    """
    otp = _generate_otp()
    record_id = create_otp_request(email, otp, expiry_minutes=OTP_EXPIRY_MINUTES)

    if record_id is None:
        logger.error(f"Failed to store OTP for {email} — database unavailable")
        return {"sent": False, "error": "Database unavailable"}

    sent = _send_email(
        to_email=email,
        subject="Driver Safety System — Password Reset OTP",
        html_body=_otp_email_body(otp),
    )

    if not sent:
        # Development mode: log OTP so it can be used without a real email server
        logger.info(f"[DEV] OTP for {email}: {otp}  (expires in {OTP_EXPIRY_MINUTES} min)")
        return {"sent": False, "dev_otp": otp}

    return {"sent": True}


def verify_otp(email: str, otp_code: str) -> bool:
    """
    Verify that the provided OTP matches the stored record and has not expired.

    Returns True if valid; does NOT delete the OTP (saved for reset-password step).
    """
    record = get_otp_request(email)
    if not record:
        return False

    if record.get("otp_code") != otp_code:
        return False

    expiry = record.get("expiry_time")
    if expiry is None:
        return False

    # expiry_time may be a datetime or an isoformat string
    if isinstance(expiry, str):
        expiry = datetime.fromisoformat(expiry)

    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) > expiry:
        delete_otp_request(email)
        return False

    return True


def consume_otp(email: str) -> None:
    """Remove the OTP after a successful password reset."""
    delete_otp_request(email)
