import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { User } from '../types/user';
import { PatientProfile } from '../types/patient';
import userService from '../services/userService';
import patientService from '../services/patientService';

interface PatientContextType {
  activePatientId: number;
  setActivePatientId: (id: number) => void;
  currentUser: User | null;
  patientProfile: PatientProfile | null;
  isLoading: boolean;
  error: string | null;
  refreshPatient: () => Promise<void>;
}

const PatientContext = createContext<PatientContextType | undefined>(undefined);

export const PatientProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Default to Patient 4 (which has verified real observations and trend records)
  const [activePatientId, setActivePatientIdState] = useState<number>(() => {
    const saved = localStorage.getItem('smarthealth_patient_id');
    return saved ? parseInt(saved, 10) : 4;
  });

  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [patientProfile, setPatientProfile] = useState<PatientProfile | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const setActivePatientId = (id: number) => {
    localStorage.setItem('smarthealth_patient_id', id.toString());
    setActivePatientIdState(id);
  };

  const loadPatientData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      // 1. Fetch User details
      const user = await userService.getUserById(activePatientId);
      setCurrentUser(user);

      // 2. Fetch Patient Profile (may return 404/null if profile not created yet)
      try {
        const profile = await patientService.getPatientProfile(activePatientId);
        setPatientProfile(profile);
      } catch {
        // Patient profile not yet saved for this user
        setPatientProfile(null);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch patient data from backend');
      setCurrentUser(null);
      setPatientProfile(null);
    } finally {
      setIsLoading(false);
    }
  }, [activePatientId]);

  useEffect(() => {
    loadPatientData();
  }, [loadPatientData]);

  return (
    <PatientContext.Provider
      value={{
        activePatientId,
        setActivePatientId,
        currentUser,
        patientProfile,
        isLoading,
        error,
        refreshPatient: loadPatientData,
      }}
    >
      {children}
    </PatientContext.Provider>
  );
};

export const usePatient = (): PatientContextType => {
  const context = useContext(PatientContext);
  if (!context) {
    throw new Error('usePatient must be used within a PatientProvider');
  }
  return context;
};

export default PatientContext;
