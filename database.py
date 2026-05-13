from datetime import datetime
from flask_login import UserMixin
from extensions import db


class User(db.Model, UserMixin):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="patient")  # patient | doctor | admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship("Prediction", backref="user", lazy=True)


class Prediction(db.Model):
    __tablename__ = "prediction"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    image_filename = db.Column(db.String(255))
    heatmap_filename = db.Column(db.String(255))
    benign = db.Column(db.Float)
    malignant = db.Column(db.Float)
    diagnosis = db.Column(db.String(20))
    detailed = db.Column(db.Text)   # JSON string
    ai_summary = db.Column(db.Text)
    doctor_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
