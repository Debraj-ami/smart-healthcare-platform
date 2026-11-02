-- =====================================================================
-- Smart Healthcare Platform — Database Schema DDL
-- Database: smart_healthcare_db
-- Phase 10: Spring Boot + MySQL Backend Foundation
-- =====================================================================

CREATE DATABASE IF NOT EXISTS smart_healthcare_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE smart_healthcare_db;

-- 1. users table
CREATE TABLE IF NOT EXISTS users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE,
    gender VARCHAR(50),
    role VARCHAR(50) NOT NULL DEFAULT 'PATIENT',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT chk_user_role CHECK (role IN ('PATIENT', 'DOCTOR', 'ADMIN'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. patient_profiles table
CREATE TABLE IF NOT EXISTS patient_profiles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE,
    blood_group VARCHAR(10),
    height_cm DECIMAL(5,2),
    weight_kg DECIMAL(5,2),
    lifestyle_context TEXT,
    medical_history TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_patient_profiles_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. disease_assessments table
CREATE TABLE IF NOT EXISTS disease_assessments (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    disease_type VARCHAR(50) NOT NULL,
    risk_score DOUBLE NOT NULL,
    risk_level VARCHAR(50) DEFAULT NULL,
    input_features TEXT,
    assessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_disease_assessments_patient FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT chk_assessment_disease_type CHECK (disease_type IN ('HEART_DISEASE', 'DIABETES', 'PARKINSONS', 'LIVER_DISEASE')),
    CONSTRAINT chk_assessment_risk_level CHECK (risk_level IS NULL OR risk_level IN ('LOW', 'MODERATE', 'HIGH'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. health_observations table
CREATE TABLE IF NOT EXISTS health_observations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value DOUBLE NOT NULL,
    unit VARCHAR(50),
    activity_level VARCHAR(50) DEFAULT 'UNKNOWN',
    activity_minutes INT DEFAULT NULL,
    logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_health_observations_patient FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. chatbot_conversations table
CREATE TABLE IF NOT EXISTS chatbot_conversations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT,
    session_id VARCHAR(100) NOT NULL,
    message_role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_chatbot_conversations_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT chk_message_role CHECK (message_role IN ('USER', 'ASSISTANT'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. personal_baselines table
CREATE TABLE IF NOT EXISTS personal_baselines (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    baseline_value DOUBLE NOT NULL,
    variability DOUBLE,
    observation_count INT DEFAULT 0,
    calculated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_personal_baselines_patient FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT uk_baseline_patient_metric UNIQUE (patient_id, metric_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. trend_analyses table
CREATE TABLE IF NOT EXISTS trend_analyses (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    baseline_id BIGINT,
    current_value DOUBLE NOT NULL,
    deviation_score DOUBLE,
    trend_direction VARCHAR(50),
    persistence VARCHAR(50),
    baseline_value DOUBLE,
    variability DOUBLE,
    absolute_deviation DOUBLE,
    deviation_from_baseline DOUBLE,
    previous_value DOUBLE,
    value_change DOUBLE,
    percentage_change DOUBLE,
    observation_count INT,
    historical_minimum DOUBLE,
    historical_maximum DOUBLE,
    historical_mean DOUBLE,
    recent_mean DOUBLE,
    diff_historical_mean DOUBLE,
    diff_recent_mean DOUBLE,
    observation_time TIMESTAMP NULL,
    activity_level VARCHAR(50),
    activity_minutes INT,
    analyzed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_trend_analyses_patient FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_trend_analyses_baseline FOREIGN KEY (baseline_id) REFERENCES personal_baselines(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8. alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    assessment_id BIGINT,
    trend_analysis_id BIGINT,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_alerts_patient FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_alerts_assessment FOREIGN KEY (assessment_id) REFERENCES disease_assessments(id) ON DELETE SET NULL,
    CONSTRAINT fk_alerts_trend FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_patient_profiles_user ON patient_profiles(user_id);
CREATE INDEX idx_disease_assessments_patient ON disease_assessments(patient_id);
CREATE INDEX idx_disease_assessments_disease ON disease_assessments(disease_type);
CREATE INDEX idx_health_observations_patient ON health_observations(patient_id);
CREATE INDEX idx_health_observations_metric ON health_observations(metric_type);
CREATE INDEX idx_chatbot_conversations_session ON chatbot_conversations(session_id);
CREATE INDEX idx_chatbot_conversations_user ON chatbot_conversations(user_id);
CREATE INDEX idx_personal_baselines_patient ON personal_baselines(patient_id);
CREATE INDEX idx_trend_analyses_patient ON trend_analyses(patient_id);
CREATE INDEX idx_alerts_patient ON alerts(patient_id);
