package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.TimeOfDay;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

public class TemporalContext {

    private LocalDate observationDate;
    private LocalTime observationTime;
    private Integer hourOfDay;
    private String dayOfWeek;
    private TimeOfDay timeOfDay;

    public TemporalContext() {
    }

    public TemporalContext(LocalDate observationDate, LocalTime observationTime,
                           Integer hourOfDay, String dayOfWeek, TimeOfDay timeOfDay) {
        this.observationDate = observationDate;
        this.observationTime = observationTime;
        this.hourOfDay = hourOfDay;
        this.dayOfWeek = dayOfWeek;
        this.timeOfDay = timeOfDay;
    }

    public static TemporalContext fromDateTime(LocalDateTime dateTime) {
        if (dateTime == null) {
            return null;
        }

        LocalDate date = dateTime.toLocalDate();
        LocalTime time = dateTime.toLocalTime();
        int hour = dateTime.getHour();
        String dayOfWeek = dateTime.getDayOfWeek().name();

        TimeOfDay timeOfDay;
        if (hour < 6) {
            timeOfDay = TimeOfDay.NIGHT;
        } else if (hour < 12) {
            timeOfDay = TimeOfDay.MORNING;
        } else if (hour < 18) {
            timeOfDay = TimeOfDay.AFTERNOON;
        } else {
            timeOfDay = TimeOfDay.EVENING;
        }

        return new TemporalContext(date, time, hour, dayOfWeek, timeOfDay);
    }

    public LocalDate getObservationDate() {
        return observationDate;
    }

    public void setObservationDate(LocalDate observationDate) {
        this.observationDate = observationDate;
    }

    public LocalTime getObservationTime() {
        return observationTime;
    }

    public void setObservationTime(LocalTime observationTime) {
        this.observationTime = observationTime;
    }

    public Integer getHourOfDay() {
        return hourOfDay;
    }

    public void setHourOfDay(Integer hourOfDay) {
        this.hourOfDay = hourOfDay;
    }

    public String getDayOfWeek() {
        return dayOfWeek;
    }

    public void setDayOfWeek(String dayOfWeek) {
        this.dayOfWeek = dayOfWeek;
    }

    public TimeOfDay getTimeOfDay() {
        return timeOfDay;
    }

    public void setTimeOfDay(TimeOfDay timeOfDay) {
        this.timeOfDay = timeOfDay;
    }
}
