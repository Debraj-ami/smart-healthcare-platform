package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.ActivityLevel;

public class ActivityContext {

    private ActivityLevel activityLevel;
    private Integer activityMinutes;

    public ActivityContext() {
        this.activityLevel = ActivityLevel.UNKNOWN;
    }

    public ActivityContext(ActivityLevel activityLevel, Integer activityMinutes) {
        this.activityLevel = activityLevel != null ? activityLevel : ActivityLevel.UNKNOWN;
        this.activityMinutes = activityMinutes;
    }

    public ActivityLevel getActivityLevel() {
        return activityLevel;
    }

    public void setActivityLevel(ActivityLevel activityLevel) {
        this.activityLevel = activityLevel != null ? activityLevel : ActivityLevel.UNKNOWN;
    }

    public Integer getActivityMinutes() {
        return activityMinutes;
    }

    public void setActivityMinutes(Integer activityMinutes) {
        this.activityMinutes = activityMinutes;
    }
}
