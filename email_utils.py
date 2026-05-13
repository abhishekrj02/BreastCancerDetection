import threading
from flask import current_app
from flask_mail import Message
from extensions import mail


def _send_async(app, msg):
    with app.app_context():
        try:
            mail.send(msg)
        except Exception as e:
            print(f"[Email] Failed to send: {e}")


def send_prediction_email(app, user, prediction):
    """Send analysis result email to patient after prediction."""
    if not app.config.get("MAIL_USERNAME"):
        return
    diag_emoji = "✅" if prediction.diagnosis == "Benign" else "⚠️"
    subject = f"{diag_emoji} Your BreastGuard AI Analysis Results — {prediction.diagnosis}"
    body = f"""
    <html><body style="font-family:Arial,sans-serif;color:#2C3E50;max-width:600px;margin:auto;">
      <div style="background:linear-gradient(135deg,#FDE8F4,#fff);padding:32px;border-radius:12px;">
        <h1 style="color:#E91E8C;margin:0 0 8px;">BreastGuard AI</h1>
        <p style="color:#7F8C8D;margin:0 0 24px;">Your mammogram analysis is ready.</p>

        <div style="background:#fff;border-radius:10px;padding:24px;box-shadow:0 4px 20px rgba(0,0,0,0.08);">
          <h2 style="color:#1A2940;margin:0 0 16px;">Hello, {user.username}!</h2>
          <p>Your mammogram has been analyzed. Here are your results:</p>

          <div style="background:{'#E8F8EF' if prediction.diagnosis=='Benign' else '#FDECEB'};
                      border-radius:8px;padding:16px;margin:16px 0;text-align:center;">
            <h3 style="color:{'#27AE60' if prediction.diagnosis=='Benign' else '#E74C3C'};
                       font-size:1.6rem;margin:0;">{prediction.diagnosis}</h3>
            <p style="margin:4px 0 0;color:#7F8C8D;">Primary Diagnosis</p>
          </div>

          <table style="width:100%;border-collapse:collapse;">
            <tr style="background:#F5F7FA;">
              <td style="padding:10px;font-weight:bold;color:#1A2940;">Benign Probability</td>
              <td style="padding:10px;color:#27AE60;font-weight:bold;">{prediction.benign:.1f}%</td>
            </tr>
            <tr>
              <td style="padding:10px;font-weight:bold;color:#1A2940;">Malignant Probability</td>
              <td style="padding:10px;color:#E74C3C;font-weight:bold;">{prediction.malignant:.1f}%</td>
            </tr>
          </table>

          <p style="margin-top:20px;">
            <a href="#" style="background:#E91E8C;color:white;padding:12px 28px;
               border-radius:50px;text-decoration:none;font-weight:bold;">View Your Dashboard</a>
          </p>
        </div>

        <p style="margin-top:20px;font-size:0.82rem;color:#7F8C8D;">
          <strong>Disclaimer:</strong> This is an AI screening tool and does not replace professional medical advice.
          Please consult a qualified healthcare provider for a definitive diagnosis.
        </p>
      </div>
    </body></html>
    """
    msg = Message(subject=subject, recipients=[user.email], html=body)
    thread = threading.Thread(target=_send_async, args=[app, msg], daemon=True)
    thread.start()


def send_high_risk_alert(app, user, prediction):
    """Send urgent alert email when malignant probability is high."""
    if not app.config.get("MAIL_USERNAME"):
        return
    subject = "⚠️ URGENT: High-Risk Result — Please Consult a Doctor Immediately"
    body = f"""
    <html><body style="font-family:Arial,sans-serif;color:#2C3E50;max-width:600px;margin:auto;">
      <div style="background:#FDECEB;border-left:6px solid #E74C3C;padding:32px;border-radius:12px;">
        <h1 style="color:#E74C3C;margin:0 0 8px;">⚠️ High-Risk Detection Alert</h1>
        <p style="color:#7F8C8D;margin:0 0 24px;">BreastGuard AI has flagged a high-risk result.</p>

        <div style="background:#fff;border-radius:10px;padding:24px;margin-bottom:20px;">
          <h2 style="color:#1A2940;">Hello, {user.username},</h2>
          <p>Your recent mammogram analysis returned a <strong style="color:#E74C3C;">high malignant probability
          ({prediction.malignant:.1f}%)</strong>. This does not confirm cancer, but we strongly recommend
          you consult a qualified medical professional as soon as possible.</p>

          <div style="background:#FDECEB;border-radius:8px;padding:16px;margin:16px 0;">
            <strong>Recommended Actions:</strong>
            <ul>
              <li>Schedule an appointment with your doctor or oncologist</li>
              <li>Request a professional mammogram reading</li>
              <li>Consider a biopsy if recommended by your doctor</li>
              <li>Do not panic — early detection greatly improves outcomes</li>
            </ul>
          </div>
        </div>

        <p style="font-size:0.82rem;color:#7F8C8D;">
          <strong>Disclaimer:</strong> BreastGuard AI is a screening assistance tool, not a diagnostic device.
          This alert is generated automatically and must be reviewed by a medical professional.
        </p>
      </div>
    </body></html>
    """
    msg = Message(subject=subject, recipients=[user.email], html=body)
    thread = threading.Thread(target=_send_async, args=[app, msg], daemon=True)
    thread.start()
