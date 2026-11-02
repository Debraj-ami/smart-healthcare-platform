import React, { useState } from 'react';
import { usePatient } from '../../context/PatientContext';
import { User, Mail, Calendar, Droplet, Ruler, Weight, FileText, CheckCircle2, Edit3, X, AlertCircle } from 'lucide-react';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import patientService from '../../services/patientService';

export const ProfilePage: React.FC = () => {
  const { activePatientId, currentUser, patientProfile, refreshPatient, isLoading } = usePatient();
  const [isEditing, setIsEditing] = useState<boolean>(false);

  // Edit form state
  const [formData, setFormData] = useState({
    bloodGroup: patientProfile?.bloodGroup || 'O+',
    heightCm: patientProfile?.heightCm?.toString() || '175',
    weightKg: patientProfile?.weightKg?.toString() || '75',
    lifestyleContext: patientProfile?.lifestyleContext || '',
    medicalHistory: patientProfile?.medicalHistory || '',
  });

  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const handleEditOpen = () => {
    setFormData({
      bloodGroup: patientProfile?.bloodGroup || 'O+',
      heightCm: patientProfile?.heightCm?.toString() || '',
      weightKg: patientProfile?.weightKg?.toString() || '',
      lifestyleContext: patientProfile?.lifestyleContext || '',
      medicalHistory: patientProfile?.medicalHistory || '',
    });
    setIsEditing(true);
    setSaveSuccess(false);
    setSaveError(null);
  };

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    setSaveError(null);

    try {
      await patientService.savePatientProfile(activePatientId, {
        bloodGroup: formData.bloodGroup,
        heightCm: formData.heightCm ? parseFloat(formData.heightCm) : undefined,
        weightKg: formData.weightKg ? parseFloat(formData.weightKg) : undefined,
        lifestyleContext: formData.lifestyleContext,
        medicalHistory: formData.medicalHistory,
      });

      await refreshPatient();
      setIsSaving(false);
      setIsEditing(false);
      setSaveSuccess(true);
    } catch (err: any) {
      setSaveError(err.message || 'Failed to save patient profile');
      setIsSaving(false);
    }
  };

  const fullName = currentUser
    ? `${currentUser.firstName} ${currentUser.lastName}`
    : `Patient #${activePatientId}`;
  const email = currentUser?.email || 'Not available';
  const dob = currentUser?.dateOfBirth || 'Not provided';
  const gender = currentUser?.gender || 'Not specified';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Patient Profile
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Manage your personal contact records and verified baseline biometrics.
          </p>
        </div>
        <Button variant="teal" size="sm" onClick={handleEditOpen} leftIcon={<Edit3 className="w-4 h-4" />}>
          Edit Profile
        </Button>
      </div>

      {saveSuccess && (
        <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
          <span>Patient profile saved successfully to database!</span>
        </div>
      )}

      {/* Main Grid: Left Avatar Card (4 cols) + Right Detail Cards (8 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Avatar Card (matching design reference #6) */}
        <div className="lg:col-span-4 bg-white rounded-3xl p-8 border border-slate-200 shadow-card text-center flex flex-col items-center justify-center space-y-4">
          <div className="w-28 h-28 rounded-full bg-gradient-to-tr from-teal-500 to-blue-600 text-white flex items-center justify-center font-bold text-3xl shadow-md border-4 border-slate-50">
            {currentUser?.firstName?.[0] || 'P'}
            {currentUser?.lastName?.[0] || activePatientId}
          </div>

          <div>
            <h2 className="text-lg font-bold text-slate-900">{fullName}</h2>
            <p className="text-xs text-slate-500">{email}</p>
            <span className="inline-block mt-2 text-[11px] font-bold px-3 py-1 rounded-full bg-slate-100 text-slate-700">
              Patient ID: #{activePatientId}
            </span>
          </div>

          <div className="w-full pt-4 border-t border-slate-100">
            <Button variant="outline" size="sm" className="w-full" onClick={handleEditOpen}>
              Update Profile Information
            </Button>
          </div>
        </div>

        {/* Right Column: Information Sections (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {/* Personal Information Card */}
          <Card className="p-6 space-y-4">
            <h3 className="text-sm font-bold text-slate-900 pb-3 border-b border-slate-100 flex items-center gap-2">
              <User className="w-4 h-4 text-teal-600" />
              <span>Personal Information</span>
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div>
                <span className="text-slate-400 block mb-0.5 font-medium">Full Name</span>
                <span className="text-sm font-semibold text-slate-800">{fullName}</span>
              </div>
              <div>
                <span className="text-slate-400 block mb-0.5 font-medium">Email Address</span>
                <span className="text-sm font-semibold text-slate-800">{email}</span>
              </div>
              <div>
                <span className="text-slate-400 block mb-0.5 font-medium">Date of Birth</span>
                <span className="text-sm font-semibold text-slate-800">{dob}</span>
              </div>
              <div>
                <span className="text-slate-400 block mb-0.5 font-medium">Gender</span>
                <span className="text-sm font-semibold text-slate-800">{gender}</span>
              </div>
            </div>
          </Card>

          {/* Health Information Card */}
          <Card className="p-6 space-y-4">
            <h3 className="text-sm font-bold text-slate-900 pb-3 border-b border-slate-100 flex items-center gap-2">
              <Droplet className="w-4 h-4 text-rose-600" />
              <span>Health & Biometric Information</span>
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
                <span className="text-slate-400 block mb-0.5 font-medium">Blood Group</span>
                <span className="text-base font-bold text-rose-700">
                  {patientProfile?.bloodGroup || 'Not specified'}
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
                <span className="text-slate-400 block mb-0.5 font-medium">Height</span>
                <span className="text-base font-bold text-slate-800">
                  {patientProfile?.heightCm ? `${patientProfile.heightCm} cm` : 'Not recorded'}
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
                <span className="text-slate-400 block mb-0.5 font-medium">Weight</span>
                <span className="text-base font-bold text-slate-800">
                  {patientProfile?.weightKg ? `${patientProfile.weightKg} kg` : 'Not recorded'}
                </span>
              </div>
            </div>

            <div className="space-y-3 pt-2 text-xs">
              <div>
                <span className="text-slate-400 block mb-1 font-medium">Lifestyle Context</span>
                <p className="p-3 bg-slate-50 rounded-xl text-slate-700 border border-slate-100">
                  {patientProfile?.lifestyleContext || 'No lifestyle context recorded yet.'}
                </p>
              </div>

              <div>
                <span className="text-slate-400 block mb-1 font-medium">Medical History</span>
                <p className="p-3 bg-slate-50 rounded-xl text-slate-700 border border-slate-100">
                  {patientProfile?.medicalHistory || 'No medical history recorded yet.'}
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Edit Profile Modal */}
      {isEditing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-xl border border-slate-200 animate-in fade-in">
            <div className="flex items-center justify-between pb-4 border-b border-slate-100">
              <h3 className="text-base font-bold text-slate-800">Edit Patient Profile</h3>
              <button
                onClick={() => setIsEditing(false)}
                className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {saveError && (
              <div className="mt-4 p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{saveError}</span>
              </div>
            )}

            <form onSubmit={handleSaveProfile} className="mt-4 space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Blood Group
                  </label>
                  <input
                    type="text"
                    value={formData.bloodGroup}
                    onChange={(e) => setFormData({ ...formData, bloodGroup: e.target.value })}
                    placeholder="e.g. O+"
                    className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Height (cm)
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={formData.heightCm}
                    onChange={(e) => setFormData({ ...formData, heightCm: e.target.value })}
                    placeholder="175"
                    className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Weight (kg)
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={formData.weightKg}
                    onChange={(e) => setFormData({ ...formData, weightKg: e.target.value })}
                    placeholder="75"
                    className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Lifestyle Context
                </label>
                <textarea
                  rows={2}
                  value={formData.lifestyleContext}
                  onChange={(e) => setFormData({ ...formData, lifestyleContext: e.target.value })}
                  placeholder="e.g. Moderate exercise 3x/week, desk job, non-smoker"
                  className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Medical History
                </label>
                <textarea
                  rows={2}
                  value={formData.medicalHistory}
                  onChange={(e) => setFormData({ ...formData, medicalHistory: e.target.value })}
                  placeholder="e.g. Family history of type 2 diabetes"
                  className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                />
              </div>

              <div className="pt-3 border-t border-slate-100 flex items-center justify-end gap-2.5">
                <Button type="button" variant="ghost" size="sm" onClick={() => setIsEditing(false)}>
                  Cancel
                </Button>
                <Button type="submit" variant="teal" size="sm" isLoading={isSaving}>
                  Save to Database
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProfilePage;
